"""Overview illustration of the problem for the report's introduction, from real VisA images.

    python scripts/make_report_overview.py --out ../report/figures

Writes ``problem_scenario.png`` and prints the image files used, so the caption can name them:

* top row: products added to the line one after another, each registered from the 4 shots of its
  fixed split (k4_s0); one shot is shown per product,
* bottom row: what the system meets in operation: a defective image (ground truth outlined), an
  image of a product that was never registered, and a registered product under changed lighting
  (``fscad.data.photometric.relight``, kind ``mixed``, severity 1.0).

This illustrates the setting; it is not a result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from fscad.data.datasets import list_samples, load_image, load_mask  # noqa: E402
from fscad.data.photometric import relight  # noqa: E402
from fscad.data.splits import load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
SIZE = 448
INK, MUTED, RED, AQUA, YELLOW = "#222222", "#6b6b6b", "#d62728", "#1baf7a", "#eda100"
ARRIVALS = ("candle", "pcb1", "capsules", "macaroni1")


def rel(p) -> str:
    return str(Path(p).relative_to(DATA))


def show(ax, x, title, color=None):
    ax.imshow(x)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, color=INK)
    for s in ax.spines.values():
        s.set_edgecolor(color or "#cccccc")
        s.set_linewidth(2.5 if color else 0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/report_assets"))
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "splits")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.0, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.45, wspace=0.12)
    used = {}
    for j, cls in enumerate(ARRIVALS):
        shot = load_split(a.splits_dir, "visa", cls, 4, 0, root=DATA)["train_paths"][0]
        used[f"arrival_{cls}"] = rel(shot)
        show(
            fig.add_subplot(gs[0, j]),
            load_image(shot, SIZE).permute(1, 2, 0),
            f"đợt {j + 1}: {cls}\n(4 ảnh tốt)",
        )
    test = list_samples("visa", DATA, "pcb1", "test")
    bad = next(s for s in test if Path(s.path).name == "020.JPG" and s.label == 1)
    good = next(s for s in test if s.label == 0)
    unknown = next(s for s in list_samples("visa", DATA, "pcb4", "test") if s.label == 0)
    used.update(defect=rel(bad.path), unknown=rel(unknown.path), drift=rel(good.path))
    ax = fig.add_subplot(gs[1, 0:1])
    show(ax, load_image(bad.path, SIZE).permute(1, 2, 0), "ảnh lỗi cần phát hiện\n(pcb1)", RED)
    ax.contour(load_mask(bad.mask, SIZE).numpy(), levels=[0.5], colors=RED, linewidths=1.2)
    show(
        fig.add_subplot(gs[1, 1:2]),
        load_image(unknown.path, SIZE).permute(1, 2, 0),
        "sản phẩm chưa đăng ký\n(pcb4)",
        YELLOW,
    )
    x = load_image(good.path, SIZE)
    show(fig.add_subplot(gs[1, 2]), x.permute(1, 2, 0), "pcb1, ánh sáng\nban đầu")
    show(
        fig.add_subplot(gs[1, 3]),
        relight(x, "mixed", 1.0).permute(1, 2, 0),
        "pcb1, ánh sáng\nđã thay đổi",
        AQUA,
    )
    arrow = FancyArrowPatch(
        (0.13, 0.515),
        (0.90, 0.515),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=14,
        color=MUTED,
        lw=1.4,
    )
    fig.patches.append(arrow)
    fig.text(
        0.515,
        0.525,
        "thời gian: sản phẩm mới lần lượt được đưa lên chuyền",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    fig.savefig(a.out / "problem_scenario.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    for k, v in used.items():
        print(k, v)


if __name__ == "__main__":
    main()
