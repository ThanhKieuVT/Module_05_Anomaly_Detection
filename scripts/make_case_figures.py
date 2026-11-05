"""Figures of correct and wrong cases on real data, for reports.

For every class: onboard the product from its fixed split (K shots), score the whole test set,
decide with the per-product threshold delta (99% quantile of validation normal scores), and draw one
example of each case

    TP  defect flagged          FN  defect missed
    TN  good image passed       FP  good image flagged

as image | ground truth | anomaly map. All maps of a class share one colour scale (0 to the 99.5%
quantile of all its maps), so a quiet map really means a low score; per-image normalisation would
make every good image look anomalous. With several classes it also routes every test image with the
prototype router and draws correct and wrong routing cases (wrong product or rejected).

    python scripts/make_case_figures.py dataset=visa 'classes=[pcb1,pcb2,pcb3,pcb4]' shots=4 \
        +fig.out=outputs/figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from fscad.data.datasets import list_samples, load_image, load_mask, normalize  # noqa: E402
from fscad.metrics import all_metrics  # noqa: E402
from fscad.models.router import PrototypeRouter  # noqa: E402
from fscad.pipeline import dataset_root, fit_from_split, load_images, resolve_classes, setup  # noqa: E402
from fscad.product import score_product  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
CASES = (("TP", "lỗi, báo đúng"), ("FN", "lỗi, bỏ sót"), ("TN", "tốt, cho qua đúng"), ("FP", "tốt, báo nhầm"))


def pick(labels: np.ndarray, scores: np.ndarray, thr: float) -> dict[str, int | None]:
    """Most typical example of each case: the clearest TP/TN and the worst FN/FP."""
    out: dict[str, int | None] = {}
    for case, lab, above in (("TP", 1, True), ("FN", 1, False), ("TN", 0, False), ("FP", 0, True)):
        idx = np.flatnonzero((labels == lab) & ((scores > thr) if above else (scores <= thr)))
        if not len(idx):
            out[case] = None
            continue
        # TP: highest score, TN: lowest; FN: lowest score (most clearly missed), FP: highest
        order = np.argsort(scores[idx])
        out[case] = int(idx[order[-1]] if case in ("TP", "FP") else idx[order[0]])
    return out


def show(ax, img, title=None):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def detection_figure(name, samples, labels, scores, maps, thr, size, out: Path) -> dict:
    chosen = pick(labels, scores, thr)
    vmax = float(np.quantile(maps, 0.995))
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 9.6))
    for r, (case, desc) in enumerate(CASES):
        i = chosen[case]
        row = axes[r]
        if i is None:
            for ax in row:
                ax.axis("off")
            row[1].text(0.5, 0.5, f"{case}: không có ảnh nào", ha="center", va="center", fontsize=9)
            continue
        img = load_image(samples[i].path, size).permute(1, 2, 0).numpy()
        mask = load_mask(samples[i].mask, size).numpy()
        show(row[0], img, f"{case} ({desc})\nscore {scores[i]:.3f} vs δ {thr:.3f}")
        show(row[1], img, "ground truth" if samples[i].mask else "ground truth (ảnh tốt)")
        if mask.any():
            row[1].contour(mask, levels=[0.5], colors="lime", linewidths=1.2)
        show(row[2], img, "anomaly map")
        row[2].imshow(maps[i], cmap="jet", alpha=0.45, vmin=0.0, vmax=vmax)
        if mask.any():
            row[2].contour(mask, levels=[0.5], colors="white", linewidths=0.8)
    fig.suptitle(f"{name}", fontsize=10)
    fig.tight_layout()
    path = out / f"cases_{name.replace('/', '_')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    counts = {
        "TP": int(((labels == 1) & (scores > thr)).sum()),
        "FN": int(((labels == 1) & (scores <= thr)).sum()),
        "TN": int(((labels == 0) & (scores <= thr)).sum()),
        "FP": int(((labels == 0) & (scores > thr)).sum()),
    }
    return {
        "figure": str(path),
        "counts": counts,
        "chosen": {k: (samples[v].path if v is not None else None) for k, v in chosen.items()},
    }


def routing_figure(entries, size, out: Path) -> str:
    """entries: list of (title, image path); up to 4 correct and 4 wrong cases."""
    n = len(entries)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, path, ok) in zip(axes.ravel(), entries, strict=False):
        show(ax, load_image(path, size).permute(1, 2, 0).numpy(), title)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("green" if ok else "red")
            s.set_linewidth(3)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    path = out / "routing_cases.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def main(overrides: list[str]) -> None:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["tracking.enabled=false", *overrides])
    out = Path(cfg.get("fig", {}).get("out", "outputs/figures"))
    out.mkdir(parents=True, exist_ok=True)
    backbone, device = setup(cfg)
    dataset, seed, size = cfg.dataset.name, cfg.seeds[0], cfg.model.img_size
    ms = cfg.eval.mask_size
    router = PrototypeRouter(n_prototypes=cfg.router.n_prototypes, seed=seed)
    report, val_cls, test = {}, {}, {}
    for cls in resolve_classes(cfg):
        name = f"{dataset}/{cls}"
        product = fit_from_split(cfg, backbone, device, dataset, cls, cfg.shots, seed)
        router.register(name, product.cls_feats)
        val_cls[name] = product.val_cls
        samples = list_samples(dataset, dataset_root(cfg, dataset), cls, "test")
        labels = np.array([s.label for s in samples])
        scores, maps, cls_feats = [], [], []
        bs = cfg.eval.batch_size * 4
        for i in range(0, len(samples), bs):
            imgs = load_images([s.path for s in samples[i : i + bs]], size)
            sc, mp = score_product(backbone, product, imgs, cfg, device, with_maps=True, map_size=ms)
            scores.append(sc)
            maps.append(mp)
            with torch.no_grad():
                cls_feats.append(backbone(normalize(imgs.to(device)), adapted=False).cls.cpu())
        scores, maps = np.concatenate(scores), np.concatenate(maps)
        masks = np.stack([load_mask(s.mask, ms).numpy() for s in samples])
        metrics = all_metrics(labels, scores, masks, maps, pixel=True)
        thr = float(product.score_threshold)
        report[name] = {
            "metrics": metrics,
            "threshold": thr,
            "n_mask_pixels_defects": int(masks[labels == 1].sum()),
            **detection_figure(name, samples, labels, scores, maps, thr, ms, out),
        }
        test[name] = (samples, torch.cat(cls_feats))
        print(
            name, json.dumps({k: round(v, 4) for k, v in metrics.items()}), report[name]["counts"], flush=True
        )

    if len(test) > 1:
        vc = torch.cat(list(val_cls.values()))
        router.calibrate(vc, [n for n, v in val_cls.items() for _ in range(len(v))], cfg.router.tau_quantile)
        good, bad = [], []
        confusion = {}
        for name, (samples, feats) in test.items():
            rr = router.route(feats)
            for s, routed, best, sim in zip(samples, rr.names, rr.best, rr.similarity.tolist(), strict=True):
                kind = "tốt" if s.label == 0 else "lỗi"
                short = name.split("/")[-1]
                if routed == name:
                    good.append((f"{short} ({kind}) → {short}\ncos {sim:.3f}", s.path, True))
                else:
                    got = f"từ chối (gần {best.split('/')[-1]})" if routed is None else routed.split("/")[-1]
                    bad.append((f"{short} ({kind}) → {got}\ncos {sim:.3f}", s.path, False))
                    confusion[f"{name} -> {routed}"] = confusion.get(f"{name} -> {routed}", 0) + 1
        rng = np.random.default_rng(seed)
        pick_good = [good[i] for i in rng.choice(len(good), min(4, len(good)), replace=False)]
        pick_bad = [bad[i] for i in rng.choice(len(bad), min(4, len(bad)), replace=False)] if bad else []
        report["routing"] = {
            "tau": router.tau,
            "accuracy": len(good) / (len(good) + len(bad)),
            "errors": confusion,
            "figure": routing_figure(pick_good + pick_bad, ms, out),
        }
        print(
            "routing", json.dumps({k: v for k, v in report["routing"].items() if k != "figure"}), flush=True
        )
    (out / "cases.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"figures in {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
