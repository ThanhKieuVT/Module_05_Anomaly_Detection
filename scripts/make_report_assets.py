"""Illustrations for the report, made from real VisA images and the project's own code.

    python scripts/make_report_assets.py --out ../report/figures

Writes PNGs plus ``assets.json`` with the exact image files used, so every caption can name them:

* dataset_samples.png  normal and defective test images of four classes, ground truth outlined
* fewshot_pcb1.png     the 4 shots of the fixed split visa/pcb1 k4_s0 and two of its validation images
* vit_patches.png      one shot cut into the 32x32 grid of 14-pixel patches DINOv2 sees at 448 px
* synthetic.png        Perlin and CutPaste anomalies made by fscad.data.synthetic on that shot
* masking.png          background masks from fscad.scoring.masking on frozen DINOv2 tokens
* vit_tokens.png       what the patch tokens of frozen DINOv2 encode: patch grid, PCA of the tokens,
                       similarity of one patch to every patch of another shot
* pipeline_detect.png  one defective image through the inference path of the code (frozen kNN, no
                       LoRA): tokens, mask, nearest memory patches, anomaly map, score vs delta
* pipeline_route.png   CLS similarity of three images to the registered products and tau
* compare_macaroni2.png  kNN and LoRA anomaly maps on the same macaroni2 test images, from the
                       products saved by the seed-2 evaluation run, read from the exported bundles
                       (--checkpoints, see docs/HUONG_DAN.md for the download), not refitted

These illustrate inputs and the mechanics of the code on single images; they are not results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from fscad.data.augment import apply as augment_apply  # noqa: E402
from fscad.data.augment import pool  # noqa: E402
from fscad.data.datasets import list_samples, load_image, load_mask, normalize  # noqa: E402
from fscad.data.splits import load_split  # noqa: E402
from fscad.data.synthetic import synthesize  # noqa: E402
from fscad.metrics import all_metrics  # noqa: E402
from fscad.models.router import PrototypeRouter  # noqa: E402
from fscad.pipeline import fit_from_split, load_images, setup  # noqa: E402
from fscad.product import ProductModel, score_product, view_foregrounds  # noqa: E402
from fscad.scoring.masking import foreground_mask  # noqa: E402
from fscad.scoring.scoring import anomaly_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
SIZE = 448
CLASSES = ("pcb1", "macaroni2", "capsules", "candle")


def rel(p: str | Path) -> str:
    return str(Path(p).relative_to(DATA))


def img(p) -> np.ndarray:
    return load_image(p, SIZE).permute(1, 2, 0).numpy()


def off(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def pick_defect(samples):
    """Defective test image whose defect is clearly visible: largest mask among the first 30."""
    bad = [s for s in samples if s.label == 1][:30]
    return max(bad, key=lambda s: float(load_mask(s.mask, 112).sum()))


def dataset_samples(out: Path) -> dict:
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(2.4 * len(CLASSES), 5.0))
    used = {}
    for j, cls in enumerate(CLASSES):
        test = list_samples("visa", DATA, cls, "test")
        good = next(s for s in test if s.label == 0)
        bad = pick_defect(test)
        axes[0, j].imshow(img(good.path))
        off(axes[0, j], f"{cls}: bình thường\n{Path(good.path).name}")
        axes[1, j].imshow(img(bad.path))
        axes[1, j].contour(load_mask(bad.mask, SIZE).numpy(), levels=[0.5], colors="red", linewidths=1.2)
        off(axes[1, j], f"{cls}: có lỗi\n{Path(bad.path).name}")
        used[cls] = {"normal": rel(good.path), "anomaly": rel(bad.path), "mask": rel(bad.mask)}
    fig.tight_layout()
    fig.savefig(out / "dataset_samples.png", dpi=150)
    plt.close(fig)
    return used


def fewshot(out: Path, splits_dir: Path) -> dict:
    split = load_split(splits_dir, "visa", "pcb1", 4, 0, root=DATA)
    shots, val = split["train_paths"], split["val_paths"][:2]
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
    for ax, p, tag in zip(axes, shots + val, ["shot"] * 4 + ["validation"] * 2, strict=True):
        ax.imshow(img(p))
        off(ax, f"{tag}\n{Path(p).name}")
        for s in ax.spines.values():
            s.set_edgecolor("tab:green" if tag == "shot" else "tab:gray")
            s.set_linewidth(2.5)
    fig.tight_layout()
    fig.savefig(out / "fewshot_pcb1.png", dpi=150)
    plt.close(fig)
    return {
        "split": "splits/visa/pcb1/k4_s0.json",
        "shots": [rel(p) for p in shots],
        "val": [rel(p) for p in val],
    }


def vit_patches(out: Path, path: str) -> None:
    x = img(path)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(x)
    for k in range(0, SIZE + 1, 14):
        ax.axhline(k - 0.5, color="white", lw=0.35, alpha=0.8)
        ax.axvline(k - 0.5, color="white", lw=0.35, alpha=0.8)
    r, c = 14, 9  # one patch highlighted
    ax.add_patch(plt.Rectangle((c * 14 - 0.5, r * 14 - 0.5), 14, 14, fill=False, ec="red", lw=1.5))
    ax.set_xlim(-0.5, SIZE - 0.5)
    ax.set_ylim(SIZE - 0.5, -0.5)
    ax.axis("off")
    fig.savefig(out / "vit_patches.png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def synthetic(out: Path, shots: list[str], backbone, device) -> dict:
    """Perlin and CutPaste on one shot, without and with the foreground mask used in training."""
    x = load_image(shots[0], SIZE)
    src = load_image(shots[1], SIZE)
    fg = view_foregrounds(backbone, x[None], ["identity"], device)[0, 0]
    cols = [
        ("perlin", None, "Perlin, không mặt nạ\n(trước d3a1313)"),
        ("perlin", fg, "Perlin, trong vật thể\n(hiện tại)"),
        ("cutpaste", fg, "CutPaste, tâm trong vật thể\n(hiện tại)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.4))
    axes[0, 0].imshow(x.permute(1, 2, 0))
    off(axes[0, 0], f"shot gốc\n{Path(shots[0]).name}")
    axes[1, 0].imshow(fg, cmap="gray")
    off(axes[1, 0], "mặt nạ vật thể\n(token đóng băng)")
    used = []
    for j, (kind, m_fg, title) in enumerate(cols, start=1):
        g = torch.Generator().manual_seed(4 if kind == "perlin" else 5)
        a, m = synthesize(x.clone(), src, kind, g, m_fg)
        axes[0, j].imshow(a.clamp(0, 1).permute(1, 2, 0))
        axes[0, j].contour(m.numpy(), levels=[0.5], colors="red", linewidths=0.8)
        off(axes[0, j], title)
        axes[1, j].imshow(m, cmap="gray")
        on_obj = float((m * fg).sum() / m.sum().clamp(min=1))
        off(axes[1, j], f"mask lỗi: {on_obj:.0%} nằm trên vật thể")
        used.append({"kind": kind, "foreground": m_fg is not None, "on_object": round(on_obj, 3)})
    fig.tight_layout()
    fig.savefig(out / "synthetic.png", dpi=150)
    plt.close(fig)
    return {"shot": rel(shots[0]), "texture_source": rel(shots[1]), "panels": used}


def knn_cfg(splits_dir: Path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "method=knn",
                "tracking.enabled=false",
                "dataset=visa",
                f"paths.data_root={ROOT / 'data'}",
                f"paths.splits_dir={splits_dir}",
            ],
        )


def pca_rgb(tokens: list[torch.Tensor], fgs: list[torch.Tensor], grid) -> list[np.ndarray]:
    """First three principal components of foreground tokens (fitted jointly) as RGB, background black."""
    fg_all = torch.cat([t[m] for t, m in zip(tokens, fgs, strict=True)]).float()
    mean = fg_all.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(fg_all - mean, full_matrices=False)
    proj = [((t.float() - mean) @ v[:3].T) for t in tokens]
    lo = torch.quantile(torch.cat([p[m] for p, m in zip(proj, fgs, strict=True)]), 0.01, dim=0)
    hi = torch.quantile(torch.cat([p[m] for p, m in zip(proj, fgs, strict=True)]), 0.99, dim=0)
    out = []
    for p, m in zip(proj, fgs, strict=True):
        rgb = ((p - lo) / (hi - lo)).clamp(0, 1) * m[:, None]
        out.append(rgb.reshape(*grid, 3).numpy())
    return out


def box(ax, r, c, color="red", lw=1.5, p=14):
    ax.add_patch(plt.Rectangle((c * p - 0.5, r * p - 0.5), p, p, fill=False, ec=color, lw=lw))


@torch.no_grad()
def vit_tokens(out: Path, cfg, backbone, device, shots: list[str]) -> dict:
    """Patch grid, PCA of patch tokens and patch-to-patch similarity between two shots."""
    a, b = shots[0], shots[1]
    x = load_images([a, b], SIZE)
    f = backbone(normalize(x.to(device)), adapted=False)
    tok, grid = f.patches.cpu(), f.grid
    fg = foreground_mask(f.patches, grid)
    rgb = pca_rgb([tok[0], tok[1]], [fg[0], fg[1]], grid)
    # query patch: the foreground patch closest to the centre of the object in image a
    ys, xs = np.unravel_index(np.flatnonzero(fg[0].numpy()), grid)
    cy, cx = ys.mean(), xs.mean()
    q = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    qr, qc = int(ys[q]), int(xs[q])
    z = torch.nn.functional.normalize(tok.float(), dim=-1)
    sim = (z[1] @ z[0, qr * grid[1] + qc]).reshape(grid).numpy()
    br, bc = np.unravel_index(int(sim.argmax()), grid)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    axes[0].imshow(x[0].permute(1, 2, 0))
    for k in range(0, SIZE + 1, 14):
        axes[0].axhline(k - 0.5, color="white", lw=0.3, alpha=0.7)
        axes[0].axvline(k - 0.5, color="white", lw=0.3, alpha=0.7)
    box(axes[0], qr, qc)
    off(axes[0], f"(a) {Path(a).name}: 32×32 patch 14×14 px\nô đỏ = patch truy vấn")
    axes[1].imshow(rgb[0], interpolation="nearest")
    off(axes[1], "(b) 3 thành phần PCA của 1024\npatch token (nền: đen)")
    axes[2].imshow(x[1].permute(1, 2, 0))
    axes[2].imshow(sim, cmap="jet", alpha=0.5, extent=(0, SIZE, SIZE, 0), vmin=0, vmax=1)
    box(axes[2], br, bc, color="white", lw=2)
    off(axes[2], f"(c) cos(patch đỏ, mọi patch của {Path(b).name})\nô trắng = gần nhất, cos {sim.max():.2f}")
    axes[3].imshow(rgb[1], interpolation="nearest")
    off(axes[3], f"(d) PCA token của {Path(b).name}\n(cùng phép chiếu với (b))")
    fig.tight_layout()
    fig.savefig(out / "vit_tokens.png", dpi=150)
    plt.close(fig)
    return {
        "a": rel(a),
        "b": rel(b),
        "query_patch": [qr, qc],
        "best": [int(br), int(bc)],
        "cos": float(sim.max()),
    }


@torch.no_grad()
def pipeline_detect(out: Path, cfg, backbone, device, cls: str = "pcb1") -> dict:
    """A defect of ``cls`` through the inference path: frozen kNN product fitted from the k4_s0 split.

    Shows a defect the product flags (S > delta) whose most anomalous patch lies on the ground truth,
    the largest such defect among the test images, and records the score of every defect tried.
    """
    product = fit_from_split(cfg, backbone, device, "visa", cls, 4, 0)
    split = load_split(cfg.paths.splits_dir, "visa", cls, 4, 0, root=DATA)
    shots = [str(p) for p in split["train_paths"]]
    thr = product.score_threshold
    grid_px = 14

    def run(path):
        x = load_images([str(path)], SIZE)
        f = backbone(normalize(x.to(device)), adapted=False)
        fg = foreground_mask(f.patches, f.grid)[0]
        d = product.memory.query(f.patches)[0].cpu()
        return x, f, fg, d, d * fg

    # same score as score_product (mean of top 1% masked patch distances)
    bad = [s_ for s_ in list_samples("visa", DATA, cls, "test") if s_.label == 1]
    tried = []
    for s_ in bad:
        x, f, fg, d, dm = run(s_.path)
        sc = float(score_product(backbone, product, x, cfg, device, with_maps=False)[0][0])
        gt_p = load_mask(s_.mask, SIZE)[grid_px // 2 :: grid_px, grid_px // 2 :: grid_px].flatten()
        hot_on_gt = bool(gt_p[int(dm.argmax())] > 0)
        tried.append((s_, sc, hot_on_gt, float(load_mask(s_.mask, 112).sum())))
    ok = [t for t in tried if t[1] > thr and t[2]]
    s_, score, _, _ = max(ok, key=lambda t: t[3])
    x, f, fg, d, dm = run(s_.path)
    grid, n = f.grid, f.grid[0] * f.grid[1]
    amap = anomaly_map(dm[None], grid, (SIZE, SIZE), cfg.eval.sigma)[0].numpy()
    gt = load_mask(s_.mask, SIZE).numpy()
    z = torch.nn.functional.normalize(f.patches[0].float(), dim=-1)
    bank = product.memory.bank
    views = pool(list(cfg.memory.augment))
    k = len(shots)

    def nearest(p):
        """Memory index -> (view, shot, patch); fit_product stacks views as [view][shot][patch]."""
        j = int((bank @ z[p]).argmax())
        v, rest = divmod(j, k * n)
        s, pp = divmod(rest, n)
        return views[v], s, pp

    hot = int(dm.argmax())
    cold_cands = torch.nonzero(fg & (dm < torch.quantile(dm[fg], 0.3))).flatten()
    cold = int(cold_cands[len(cold_cands) // 2])
    rgb = pca_rgb([f.patches[0].cpu()], [fg.cpu()], grid)[0]
    im = x[0].permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, 4, figsize=(11, 6.4))
    ax = axes[0, 0]
    ax.imshow(im)
    ax.contour(gt, levels=[0.5], colors="lime", linewidths=1)
    for p, col in ((hot, "red"), (cold, "cyan")):
        box(ax, *divmod(p, grid[1]), color=col, lw=1.8)
    off(ax, f"① ảnh vào {cls}/{Path(s_.path).name}\nxanh lá: ground truth")
    ax = axes[0, 1]
    ax.imshow(rgb, interpolation="nearest")
    off(ax, "② 1024 patch token DINOv2\n(PCA 3 thành phần → màu)")
    ax = axes[0, 2]
    keep = np.kron(fg.reshape(grid).float().numpy(), np.ones((grid_px, grid_px)))[..., None]
    ax.imshow(im * (0.25 + 0.75 * keep))
    off(ax, f"③ mặt nạ vật thể (nền tối):\ngiữ {fg.float().mean():.0%} patch")
    ax = axes[0, 3]
    tiles = [load_image(p, SIZE).permute(1, 2, 0).numpy() for p in shots]
    ax.imshow(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    off(ax, f"④ bộ nhớ: 4 shot × {len(views)} view\n= {len(product.memory)} patch token")
    for col_i, (p, col, tag) in enumerate(((hot, "red", "patch đỏ"), (cold, "cyan", "patch xanh"))):
        view, s, pp = nearest(p)
        ax = axes[1, col_i]
        ax.imshow(augment_apply(load_image(shots[s], SIZE), view).permute(1, 2, 0).numpy())
        box(ax, *divmod(pp, grid[1]), color=col, lw=2)
        off(ax, f"⑤ láng giềng gần nhất của {tag}:\n{Path(shots[s]).name} ({view}), d = {float(d[p]):.3f}")
    ax = axes[1, 2]
    ax.imshow(dm.reshape(grid).numpy(), cmap="jet", interpolation="nearest", vmin=0)
    off(ax, "⑥ khoảng cách cos tới bộ nhớ\ntừng patch (nền = 0)")
    ax = axes[1, 3]
    ax.imshow(im)
    ax.imshow(amap, cmap="jet", alpha=0.45, vmin=0)
    ax.contour(gt, levels=[0.5], colors="white", linewidths=0.8)
    off(ax, f"⑦ bản đồ lỗi; S = {score:.3f} > δ = {thr:.3f}\n→ báo LỖI")
    fig.tight_layout()
    fig.savefig(out / "pipeline_detect.png", dpi=150)
    plt.close(fig)
    return {
        "image": rel(s_.path),
        "shots": [rel(p) for p in shots],
        "memory_patches": len(product.memory),
        "kept": round(float(fg.float().mean()), 3),
        "score": round(score, 4),
        "delta": round(float(thr), 4),
        "hot": {"patch": divmod(hot, grid[1]), "d": round(float(d[hot]), 3), "nn": nearest(hot)},
        "cold": {"patch": divmod(cold, grid[1]), "d": round(float(d[cold]), 3), "nn": nearest(cold)},
        "defects_tried": {rel(t[0].path): {"score": round(t[1], 4), "hot_on_gt": t[2]} for t in tried},
    }


KNOWN = ("pcb1", "pcb2", "pcb3", "macaroni2", "capsules")


@torch.no_grad()
def pipeline_route(out: Path, cfg, backbone, device, inputs: list[tuple[str, Path]]) -> dict:
    """Router of the code on real CLS tokens: products registered from their k4_s0 shots."""
    router = PrototypeRouter(n_prototypes=cfg.router.n_prototypes, seed=0)
    val_cls, labels, thumbs = [], [], {}
    for cls in KNOWN:
        split = load_split(cfg.paths.splits_dir, "visa", cls, 4, 0, root=DATA)
        sh = load_images(split["train_paths"], SIZE)
        router.register(f"visa/{cls}", backbone(normalize(sh.to(device)), adapted=False).cls.cpu())
        v = load_images(split["val_paths"], SIZE)
        val_cls.append(
            torch.cat(
                [
                    backbone(normalize(v[i : i + 8].to(device)), adapted=False).cls.cpu()
                    for i in range(0, len(v), 8)
                ]
            )
        )
        labels += [f"visa/{cls}"] * len(v)
        thumbs[cls] = split["train_paths"][0]
    tau = router.calibrate(torch.cat(val_cls), labels, cfg.router.tau_quantile)
    x = load_images([str(p) for _, p in inputs], SIZE)
    z = backbone(normalize(x.to(device)), adapted=False).cls.cpu()
    sims, names = router.similarities(z)
    rr = router.route(z)
    fig = plt.figure(figsize=(9, 1.75 * (len(inputs) + 1)))
    gs = fig.add_gridspec(len(inputs) + 1, 5)
    for j, cls in enumerate(KNOWN):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img(thumbs[cls]))
        off(ax, f"đã đăng ký: {cls}")
    lo = min(float(sims.min()), tau) - 0.05  # one x scale for all rows
    res = []
    for i, (tag, p) in enumerate(inputs):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.imshow(x[i].permute(1, 2, 0))
        routed = rr.names[i]
        dec = f"→ {routed.split('/')[-1]}" if routed else "→ TỪ CHỐI"
        off(ax, f"{tag}\n{p.parts[-5]}/{p.name}\n{dec}")
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color("tab:red" if routed is None else "tab:green")
            s.set_linewidth(3)
        ax = fig.add_subplot(gs[i + 1, 1:5])
        vals = sims[i].numpy()
        short = [n.split("/")[-1] for n in names]
        colors = ["tab:green" if (routed and n == routed) else "tab:gray" for n in names]
        ax.barh(short, vals, color=colors, height=0.7)
        ax.axvline(tau, color="tab:red", ls="--", lw=1.2, label=f"τ = {tau:.3f}")
        ax.legend(loc="lower right", fontsize=9, frameon=False)
        ax.set_xlim(lo, 1.08)
        ax.invert_yaxis()
        ax.tick_params(labelsize=9)
        if i == len(inputs) - 1:
            ax.set_xlabel("cos(CLS ảnh, prototype)", fontsize=9)
        else:
            ax.set_xticklabels([])
        for k_, v_ in enumerate(vals):
            ax.text(max(v_, 0), k_, f" {v_:.3f}", va="center", fontsize=8)
        res.append(
            {
                "input": rel(p),
                "tag": tag,
                "routed": routed,
                "sims": {n: round(float(v_), 4) for n, v_ in zip(short, vals, strict=True)},
            }
        )
    fig.tight_layout()
    fig.savefig(out / "pipeline_route.png", dpi=150)
    plt.close(fig)
    return {"known": list(KNOWN), "tau": round(tau, 4), "inputs": res}


@torch.no_grad()
def compare_maps(out: Path, ckpts: Path, cls: str, seed: int, names: list[str]) -> dict:
    """Saved kNN and LoRA products of one split, scored on the same test images.

    Each method's maps share one colour scale (0 to the 99.5% quantile over the whole test set), so
    colours are comparable within a column, and every title gives the score against that method's delta.
    """
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="config", overrides=["method=lora", "tracking.enabled=false", "dataset=visa"]
        )
    backbone, device = setup(cfg)
    prods = {
        m: ProductModel.load(ckpts / f"visa-{m}-k4-s{seed}" / f"visa__{cls}" / "model.pt", device)
        for m in ("knn", "lora")
    }
    test = list_samples("visa", DATA, cls, "test")
    allx = load_images([s_.path for s_ in test], SIZE)
    res = {}
    for m, prod in prods.items():
        sc, mp = score_product(backbone, prod, allx, cfg, device, with_maps=True, map_size=SIZE)
        res[m] = (sc, mp, float(np.quantile(mp, 0.995)), prod.score_threshold)
    idx = [next(i for i, s_ in enumerate(test) if Path(s_.path).name == n) for n in names]
    fig, axes = plt.subplots(len(idx), 3, figsize=(7.4, 2.55 * len(idx)))
    used = []
    for r, i in enumerate(idx):
        s_ = test[i]
        im = allx[i].permute(1, 2, 0).numpy()
        gt = load_mask(s_.mask, SIZE).numpy()
        kind = "lỗi" if s_.label else "tốt"
        axes[r, 0].imshow(im)
        if gt.any():
            axes[r, 0].contour(gt, levels=[0.5], colors="lime", linewidths=1.2)
        off(axes[r, 0], f"{Path(s_.path).name} ({kind})")
        row = {"image": rel(s_.path), "label": int(s_.label)}
        for j, m in enumerate(("knn", "lora"), start=1):
            sc, mp, vmax, thr = res[m]
            axes[r, j].imshow(im)
            axes[r, j].imshow(mp[i], cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
            if gt.any():
                axes[r, j].contour(gt, levels=[0.5], colors="white", linewidths=0.8)
            flag = "báo lỗi" if sc[i] > thr else "cho qua"
            off(axes[r, j], f"{'kNN' if m == 'knn' else 'LoRA'}: S = {sc[i]:.3f}, δ = {thr:.3f} → {flag}")
            row[m] = {"score": round(float(sc[i]), 4), "delta": round(float(thr), 4)}
        used.append(row)
    fig.tight_layout()
    fig.savefig(out / f"compare_{cls}.png", dpi=150)
    plt.close(fig)
    labels = np.array([s_.label for s_ in test])
    return {
        "seed": seed,
        "images": used,
        **{
            f"i_auroc_{m}": round(all_metrics(labels, res[m][0], None, None, pixel=False)["i_auroc"], 4)
            for m in res
        },
    }


@torch.no_grad()
def masking(out: Path, used: dict, backbone, device) -> dict:
    paths = [DATA / used[c]["anomaly"] for c in ("pcb1", "macaroni2", "capsules")]
    x = torch.stack([load_image(p, SIZE) for p in paths])
    f = backbone(normalize(x.to(device)), adapted=False)
    m = foreground_mask(f.patches, f.grid).reshape(-1, *f.grid).float()
    fig, axes = plt.subplots(2, len(paths), figsize=(2.6 * len(paths), 5.0))
    for j, p in enumerate(paths):
        axes[0, j].imshow(x[j].permute(1, 2, 0))
        off(axes[0, j], f"{p.parts[-5]} / {p.name}")
        axes[1, j].imshow(x[j].permute(1, 2, 0))
        axes[1, j].imshow(m[j], cmap="Greens", alpha=0.45, extent=(0, SIZE, SIZE, 0))
        off(axes[1, j], f"giữ {m[j].mean():.0%} patch (xanh)")
    fig.tight_layout()
    fig.savefig(out / "masking.png", dpi=150)
    plt.close(fig)
    return {"images": [rel(p) for p in paths], "kept": [round(float(v), 3) for v in m.mean((1, 2))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/report_assets")
    ap.add_argument("--splits-dir", default=str(ROOT / "splits"))
    ap.add_argument("--checkpoints", default=str(ROOT / "checkpoints"), help="exported bundles")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assets = {"dataset_samples": dataset_samples(out)}
    assets["fewshot"] = fewshot(out, Path(a.splits_dir))
    shots = [str(DATA / p) for p in assets["fewshot"]["shots"]]
    vit_patches(out, shots[0])
    assets["vit_patches"] = assets["fewshot"]["shots"][0]
    cfg = knn_cfg(Path(a.splits_dir))
    backbone, device = setup(cfg)
    assets["synthetic"] = synthetic(out, shots, backbone, device)
    assets["masking"] = masking(out, assets["dataset_samples"], backbone, device)
    assets["vit_tokens"] = vit_tokens(out, cfg, backbone, device, shots)
    assets["pipeline_detect"] = pipeline_detect(out, cfg, backbone, device)

    def test(c, k):
        return next(s.path for s in list_samples("visa", DATA, c, "test") if s.label == k)

    assets["pipeline_route"] = pipeline_route(
        out,
        cfg,
        backbone,
        device,
        [
            ("đã đăng ký, tốt", Path(test("pcb1", 0))),
            ("đã đăng ký, có lỗi", DATA / assets["pipeline_detect"]["image"]),
            ("chưa đăng ký, giống", Path(test("pcb4", 0))),
            ("chưa đăng ký, khác", Path(test("candle", 0))),
        ],
    )
    assets["compare_macaroni2"] = compare_maps(
        out, Path(a.checkpoints), "macaroni2", 2, ["027.JPG", "071.JPG", "028.JPG", "0648.JPG"]
    )
    (out / "assets.json").write_text(json.dumps(assets, indent=2, ensure_ascii=False))
    print(json.dumps(assets, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
