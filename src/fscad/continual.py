"""Task-agnostic few-shot continual anomaly detection with a rejecting router.

Products arrive one by one. Each gets its own adapter + memory bank + prototype; nothing learned
before is modified. After every step all seen products are evaluated twice:

* oracle routing (true product known)   -> isolates detection error, FM is 0 by construction
* real routing (prototype router)        -> system error = detection error + routing error

With real routing an image rejected as "unknown product" gets no anomaly score. How it counts is set
by ``continual.reject_policy``: ``alarm`` (a rejection is an alert, ranked above every scored image)
or ``miss`` (a rejection passes unflagged, ranked below every scored image). Both are reported;
rejection rates are reported separately for normal and anomalous test images, because tau is
calibrated on normal images only and strong defects may be rejected instead of flagged.

Held-out products (never registered) measure the router's ability to reject unknown products; they
are only informative when they resemble registered ones (e.g. visa/pcb4 held out, pcb1-3 known).

    python -m fscad.continual continual.datasets=[mvtec_ad] shots=4
    python -m fscad.continual continual.datasets=[mvtec_ad,visa] continual.holdout=[] continual.order=random
"""

from __future__ import annotations

import csv
import json
import logging
import random
from contextlib import contextmanager
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score

from fscad.data.datasets import list_samples, load_mask, normalize
from fscad.data.splits import load_split
from fscad.metrics import all_metrics, forgetting_measure, image_auroc
from fscad.models.router import PrototypeRouter
from fscad.pipeline import (
    artifact_path,
    dataset_root,
    fit_from_split,
    load_images,
    resolve_classes,
    setup,
)
from fscad.product import ProductModel, score_product
from fscad.tracking import Tracker
from fscad.utils import seed_everything

log = logging.getLogger("fscad.continual")


def _nanmean(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")


REJECT_POLICIES = ("alarm", "miss")


def system_scores(scores: np.ndarray, rejected: np.ndarray, policy: str) -> np.ndarray:
    """Scores of the whole system: routed scores, with rejected images placed by the policy.

    ``scores`` holds the routed product's score for accepted images (any value where rejected).
    AUROC only uses the ranking, so rejected images get a value just above (alarm) or below (miss)
    every accepted score.
    """
    rejected = np.asarray(rejected, dtype=bool)
    out = np.asarray(scores, dtype=float).copy()
    acc = out[~rejected]
    hi, lo = (acc.max() + 1.0, acc.min() - 1.0) if acc.size else (1.0, 0.0)
    if policy == "alarm":
        out[rejected] = hi
    elif policy == "miss":
        out[rejected] = lo
    else:
        raise ValueError(f"unknown reject_policy {policy}; expected one of {REJECT_POLICIES}")
    return out


def routing_confusions(true: list[str], pred: list[str], top: int = 20) -> list[dict]:
    """Most frequent (true, predicted) routing mistakes, with the rate within the true class."""
    counts: dict[tuple[str, str], int] = {}
    per_class: dict[str, int] = {}
    for t, p in zip(true, pred, strict=True):
        per_class[t] = per_class.get(t, 0) + 1
        if t != p:
            counts[(t, p)] = counts.get((t, p), 0) + 1
    rows = [{"true": t, "pred": p, "n": n, "rate": n / per_class[t]} for (t, p), n in counts.items()]
    return sorted(rows, key=lambda r: -r["n"])[:top]


@contextmanager
def on_device(product: ProductModel, device: torch.device):
    product.memory.to(device)
    try:
        yield product
    finally:
        product.memory.to("cpu")


@torch.no_grad()
def frozen_cls(backbone, images: torch.Tensor, device, batch_size: int) -> torch.Tensor:
    out = []
    for i in range(0, len(images), batch_size):
        out.append(backbone(normalize(images[i : i + batch_size].to(device)), adapted=False).cls.cpu())
    return torch.cat(out) if out else torch.empty(0, backbone.embed_dim)


def task_order(cfg) -> tuple[list[str], list[str]]:
    holdout = set(cfg.continual.holdout or [])
    tasks = [f"{ds}/{c}" for ds in cfg.continual.datasets for c in resolve_classes(cfg, ds)]
    known = [t for t in tasks if t not in holdout]
    if cfg.continual.order == "alphabetical":
        known = sorted(known)
    elif cfg.continual.order == "random":
        random.Random(cfg.continual.order_seed).shuffle(known)
    else:
        raise ValueError(cfg.continual.order)
    if cfg.continual.max_tasks:
        known = known[: cfg.continual.max_tasks]
    return known, sorted(holdout)


class TaskData:
    """Test set of one product, with cached frozen CLS and per-(image, product) scores."""

    def __init__(self, cfg, name: str):
        self._size = cfg.model.img_size
        ds, cls = name.split("/", 1)
        samples = list_samples(ds, dataset_root(cfg, ds), cls, "test")
        if cfg.eval.max_test_images:
            idx = np.random.default_rng(0).permutation(len(samples))[: cfg.eval.max_test_images]
            samples = [samples[i] for i in sorted(idx)]
        self.name = name
        self.samples = samples
        self.labels = np.array([s.label for s in samples])
        self.cls: torch.Tensor | None = None
        self.scores: dict[str, np.ndarray] = {}  # product -> scores (NaN where not computed)

    def images(self, idx) -> torch.Tensor:
        return load_images([self.samples[i].path for i in idx], self._size)

    def scores_for(self, product: str) -> np.ndarray:
        if product not in self.scores:
            self.scores[product] = np.full(len(self.samples), np.nan)
        return self.scores[product]


def run(cfg: DictConfig) -> dict:
    seed_everything(cfg.seeds[0])
    seed = cfg.seeds[0]
    tracker = Tracker.from_cfg(cfg)
    backbone, device = setup(cfg)
    known, holdout = task_order(cfg)
    log.info("continual order (%d tasks): %s | holdout: %s", len(known), known, holdout)
    router = PrototypeRouter(n_prototypes=cfg.router.n_prototypes, seed=seed)
    products: dict[str, ProductModel] = {}
    tasks: dict[str, TaskData] = {}
    val_cls: dict[str, torch.Tensor] = {}
    T = len(known)
    policy = cfg.continual.reject_policy
    if policy not in REJECT_POLICIES:
        raise ValueError(f"continual.reject_policy must be one of {REJECT_POLICIES}")
    acc_real = {p: np.full((T, T), np.nan) for p in REJECT_POLICIES}
    acc_oracle = np.full((T, T), np.nan)
    oracle_final: dict[str, dict] = {}
    step_rows = []
    bs = cfg.eval.batch_size

    def score_missing(task: TaskData, routed: list[str], index) -> None:
        """Compute scores for (image index, routed product) pairs not seen before."""
        by_product: dict[str, list[int]] = {}
        for i, p in zip(index, routed, strict=True):
            if np.isnan(task.scores_for(p)[i]):
                by_product.setdefault(p, []).append(i)
        for p, idx in by_product.items():
            with on_device(products[p], device):
                for s in range(0, len(idx), bs):
                    chunk = idx[s : s + bs]
                    sc, _ = score_product(
                        backbone, products[p], task.images(chunk), cfg, device, with_maps=False
                    )
                    task.scores_for(p)[chunk] = sc

    tags = {
        "phase": cfg.phase,
        "dataset": "+".join(cfg.continual.datasets),
        "class": "all",
        "shots": cfg.shots,
        "seed": seed,
        "method": f"{cfg.method}-continual",
        "order": cfg.continual.order,
    }
    with tracker.run(
        f"continual-{tags['dataset']}-{cfg.continual.order}-k{cfg.shots}",
        tags,
        params=OmegaConf.to_container(cfg, resolve=True),
    ):
        for t, name in enumerate(known):
            ds, cls = name.split("/", 1)
            apath = artifact_path(cfg, ds, cls, cfg.shots, seed)
            if cfg.reuse_artifacts and apath.exists():
                product = ProductModel.load(apath, "cpu")
            else:
                product = fit_from_split(cfg, backbone, device, ds, cls, cfg.shots, seed)
                if cfg.save_artifacts:
                    product.save(apath)
            product.name = name
            product.memory.to("cpu")
            products[name] = product
            router.register(name, product.cls_feats)

            # router threshold from normal validation images of every registered product
            split = load_split(cfg.paths.splits_dir, ds, cls, cfg.shots, seed, root=dataset_root(cfg, ds))
            val_cls[name] = frozen_cls(
                backbone, load_images(split["val_paths"], cfg.model.img_size), device, bs
            )
            vc = torch.cat([v for v in val_cls.values() if len(v)])
            vl = [n for n, v in val_cls.items() for _ in range(len(v))]
            if len(vc):
                router.calibrate(vc, vl, cfg.router.tau_quantile)

            # new task: cache CLS, oracle scores and oracle pixel metrics (they never change)
            task = TaskData(cfg, name)
            cls_all = []
            maps, masks = [], []
            with on_device(product, device):
                for s in range(0, len(task.samples), bs):
                    idx = list(range(s, min(s + bs, len(task.samples))))
                    imgs = task.images(idx)
                    cls_all.append(frozen_cls(backbone, imgs, device, bs))
                    sc, mp = score_product(backbone, product, imgs, cfg, device, with_maps=cfg.eval.pixel)
                    task.scores_for(name)[idx] = sc
                    if cfg.eval.pixel:
                        maps.append(mp)
                        masks.append(
                            np.stack(
                                [load_mask(task.samples[i].mask, cfg.eval.mask_size).numpy() for i in idx]
                            )
                        )
            task.cls = torch.cat(cls_all)
            tasks[name] = task
            oracle_final[name] = all_metrics(
                task.labels,
                task.scores_for(name),
                np.concatenate(masks) if masks else None,
                np.concatenate(maps) if maps else None,
                pixel=cfg.eval.pixel,
            )

            # evaluate all seen tasks with real and oracle routing
            correct = correct_argmax = total = 0
            rej = {"normal": [0, 0], "anomalous": [0, 0]}  # [rejected, total]
            for j, prev in enumerate(known[: t + 1]):
                tk = tasks[prev]
                rr = router.route(tk.cls)
                rejected = rr.rejected.numpy()
                accepted = [p for p, r in zip(rr.best, rejected, strict=True) if not r]
                score_missing(tk, accepted, np.flatnonzero(~rejected))
                routed = np.array(
                    [
                        np.nan if r else tk.scores_for(p)[i]
                        for i, (p, r) in enumerate(zip(rr.best, rejected, strict=True))
                    ]
                )
                for p in REJECT_POLICIES:
                    acc_real[p][t, j] = image_auroc(tk.labels, system_scores(routed, rejected, p))
                acc_oracle[t, j] = image_auroc(tk.labels, tk.scores_for(prev))
                correct += sum(n == prev for n in rr.names)  # accepted and routed correctly
                correct_argmax += sum(p == prev for p in rr.best)
                total += len(rr.best)
                for key, sel in (("normal", tk.labels == 0), ("anomalous", tk.labels == 1)):
                    rej[key][0] += int(rejected[sel].sum())
                    rej[key][1] += int(sel.sum())
            row = {
                "step": t,
                "task": name,
                "n_classes": t + 1,
                "avg_i_auroc_real": float(np.nanmean(acc_real[policy][t, : t + 1])),
                **{
                    f"avg_i_auroc_real_{p}": float(np.nanmean(acc_real[p][t, : t + 1]))
                    for p in REJECT_POLICIES
                },
                "avg_i_auroc_oracle": float(np.nanmean(acc_oracle[t, : t + 1])),
                "routing_acc": correct / max(1, total),
                "routing_acc_argmax": correct_argmax / max(1, total),
                "reject_rate_normal": rej["normal"][0] / max(1, rej["normal"][1]),
                "reject_rate_anomalous": rej["anomalous"][0] / max(1, rej["anomalous"][1]),
                "tau": router.tau,
            }
            step_rows.append(row)
            tracker.log_metrics({k: v for k, v in row.items() if isinstance(v, float)}, step=t)
            log.info(
                "step %d/%d %-28s real=%.4f oracle=%.4f route=%.4f",
                t + 1,
                T,
                name,
                row["avg_i_auroc_real"],
                row["avg_i_auroc_oracle"],
                row["routing_acc"],
            )

        # rejection of unregistered products, overall and per held-out product
        rej, per_holdout = {}, {}
        if holdout:
            known_cls = torch.cat([tasks[n].cls for n in known])
            sim_k = router.route(known_cls).similarity.numpy()
            per_holdout, sim_u = {}, []
            for h in holdout:
                hd = TaskData(cfg, h)
                rh = router.route(frozen_cls(backbone, hd.images(range(len(hd.samples))), device, bs))
                s_h = rh.similarity.numpy()
                sim_u.append(s_h)
                nearest = max(set(rh.best), key=rh.best.count)
                per_holdout[h] = {
                    "true_reject_rate": float((s_h < router.tau).mean()),
                    "reject_auroc": float(
                        roc_auc_score(np.r_[np.zeros(len(sim_k)), np.ones(len(s_h))], -np.r_[sim_k, s_h])
                    ),
                    "nearest_known": nearest,
                }
            sim_u = np.concatenate(sim_u)
            y = np.r_[np.zeros(len(sim_k)), np.ones(len(sim_u))]
            rej = {
                "reject_auroc": float(roc_auc_score(y, -np.r_[sim_k, sim_u])),
                "true_reject_rate": float((sim_u < router.tau).mean()),
                "false_reject_rate_final": float((sim_k < router.tau).mean()),
            }

        # routing accuracy vs rejection rate (sweep tau) on known products
        curve = []
        all_cls = torch.cat([tasks[n].cls for n in known])
        own = [n for n in known for _ in range(len(tasks[n].samples))]
        rr = router.route(all_cls)
        sims = rr.similarity.numpy()
        hit = np.array([b == o for b, o in zip(rr.best, own, strict=True)])
        confusions = routing_confusions(own, rr.best)
        for q in np.linspace(0, 0.3, 31):
            thr = np.quantile(sims, q)
            acc_mask = sims >= thr
            curve.append(
                {
                    "reject_rate": float(1 - acc_mask.mean()),
                    "routing_acc_accepted": float(hit[acc_mask].mean()) if acc_mask.any() else float("nan"),
                }
            )

        final = {
            "reject_policy": policy,
            "avg_i_auroc_real": float(np.nanmean(acc_real[policy][T - 1])),
            "avg_i_auroc_oracle": float(np.nanmean(acc_oracle[T - 1])),
            "fm_real": forgetting_measure(acc_real[policy]),
            **{f"avg_i_auroc_real_{p}": float(np.nanmean(acc_real[p][T - 1])) for p in REJECT_POLICIES},
            **{f"fm_real_{p}": forgetting_measure(acc_real[p]) for p in REJECT_POLICIES},
            "fm_oracle": forgetting_measure(acc_oracle),
            "routing_acc": step_rows[-1]["routing_acc"],
            "routing_acc_argmax": step_rows[-1]["routing_acc_argmax"],
            "reject_rate_normal": step_rows[-1]["reject_rate_normal"],
            "reject_rate_anomalous": step_rows[-1]["reject_rate_anomalous"],
            "tau": router.tau,
            "mb_per_class": float(np.mean([p.nbytes() for p in products.values()]) / 2**20),
            "oracle_au_pro_mean": _nanmean([m.get("au_pro", np.nan) for m in oracle_final.values()]),
            **rej,
        }
        tracker.log_metrics({k: v for k, v in final.items() if isinstance(v, float)})
        result = {
            "final": final,
            "steps": step_rows,
            "order": known,
            "holdout": holdout,
            "acc_real": {p: a.tolist() for p, a in acc_real.items()},
            "acc_oracle": acc_oracle.tolist(),
            "oracle_per_task": oracle_final,
            "routing_vs_reject": curve,
            "routing_confusions": confusions,
            "holdout_rejection": per_holdout,
        }
        tracker.log_dict(result, "continual_result.json")
    log.info(
        "final: %s", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in final.items()})
    )
    return result


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    result = run(cfg)
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "continual_result.json").write_text(json.dumps(result, indent=2, default=float))
    with open(out / "continual_steps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result["steps"][0]))
        w.writeheader()
        w.writerows(result["steps"])


if __name__ == "__main__":
    main()
