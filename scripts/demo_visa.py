"""Live demo on real VisA images: register products from 4 good images, inspect, reject, watch drift.

    python scripts/demo_visa.py                          # kNN on frozen DINOv2, about 1 minute in all
    python scripts/demo_visa.py --method lora            # LoRA onboarding, about 3 minutes per product
    python scripts/demo_visa.py --products checkpoints/visa-lora-k4-s0   # LoRA bundles from Drive
    python scripts/demo_visa.py --no-pause               # run straight through (rehearsal, recording)

Four steps, each waits for Enter so the presenter can talk over it:

1. Onboard pcb1, candle and capsules, each from the 4 shots of its fixed split ``k4_s0`` (30 more good
   images of the split set the thresholds). With ``--products`` the saved products are loaded instead.
2. Inspect eight test images: good and defective ones of the registered products, a product that was
   never registered (pcb4) and a registered product under changed lighting. One line per image:
   decision, routed product, CLS similarity to the nearest prototype, anomaly score and threshold.
3. Save ``<out>/demo_panel.png``: every image next to its anomaly map, with the decision in the title.
4. Drift: stream 60 good pcb1 test images, first as they are, then with the lighting changed
   (``relight`` kind ``mixed``, severity 0.6), and print the monitor's KS p-values after each stream.

Decisions: ``ok``, ``LỖI`` (score above the product's threshold delta) or ``TỪ CHỐI`` (the router
matched no registered product; under the report's ``alarm`` policy this goes to a person).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from fscad.data.datasets import list_samples, load_image, load_mask  # noqa: E402
from fscad.data.photometric import relight  # noqa: E402
from fscad.data.splits import load_split  # noqa: E402
from fscad.pipeline import load_images  # noqa: E402
from fscad.scoring.masking import masking_default  # noqa: E402
from fscad.service import AnomalyService  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "visa"
PRODUCTS = ("pcb1", "candle", "capsules")
# (label shown on screen, class, file under <class>/Data/Images/, relit?)
CASES = [
    ("pcb1, ảnh tốt", "pcb1", "Normal/0107.JPG", False),
    ("pcb1, lỗi nhỏ ở chân cắm", "pcb1", "Anomaly/020.JPG", False),
    ("pcb1, lỗi lớn ở hai màng", "pcb1", "Anomaly/043.JPG", False),
    ("candle, ảnh tốt", "candle", "Normal/0790.JPG", False),
    ("candle, lỗi ở mép nến", "candle", "Anomaly/006.JPG", False),
    ("capsules, lỗi trên một viên", "capsules", "Anomaly/018.JPG", False),
    ("pcb4, sản phẩm chưa đăng ký", "pcb4", "Normal/0107.JPG", False),
    ("pcb1, ánh sáng đã đổi", "pcb1", "Normal/0107.JPG", True),
]


def step(title: str, pause: bool) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    if pause:
        input("  [Enter để chạy] ")


def demo_cfg(method: str, products_dir: Path, device: str, steps: int):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="serve")
        mcfg = compose(config_name="config", overrides=[f"method={method}"])
    OmegaConf.set_struct(cfg, False)
    # adapter and training settings of the report's runs (configs/method, configs/train)
    cfg.method, cfg.lora, cfg.train = mcfg.method, mcfg.lora, mcfg.train
    cfg.device, cfg.products_dir = device, str(products_dir)
    cfg.onboard_steps = steps if method == "lora" else 0
    cfg.drift = {"window": 50, "min_samples": 30}  # as in the H3 experiment
    return cfg


def onboard(svc: AnomalyService, splits: Path) -> None:
    for cls in PRODUCTS:
        split = load_split(splits, "visa", cls, 4, 0, root=DATA)
        size = svc.cfg.model.img_size
        shots = load_images(split["train_paths"], size)
        val = load_images(split["val_paths"], size)
        t0 = time.perf_counter()
        r = svc.onboard(f"visa/{cls}", shots, val, masking=masking_default("visa", cls, "auto"))
        names = ", ".join(Path(p).name for p in split["train_paths"])
        print(
            f"  {cls:<9} 4 ảnh tốt ({names})  ->  {time.perf_counter() - t0:5.1f} s, "
            f"{r['bytes'] / 2**20:.1f} MB, ngưỡng δ = {r['threshold']:.3f}"
        )
    print(f"  Bộ định tuyến: {len(svc.router)} sản phẩm, ngưỡng từ chối τ = {svc.router.tau:.3f}")


def inspect(svc: AnomalyService) -> list[dict]:
    size = svc.cfg.model.img_size
    rows = []
    print(f"  {'ảnh':<32}{'quyết định':<11}{'sản phẩm':<11}{'cos CLS':>8}{'điểm S':>9}{'δ':>8}{'ms':>7}")
    for label, cls, rel, relit in CASES:
        path = DATA / cls / "Data" / "Images" / rel
        x = load_image(str(path), size)
        if relit:
            x = relight(x, "mixed", 1.0)
        p = svc.predict(x, return_map=True)
        decision = "TỪ CHỐI" if p.rejected else ("LỖI" if p.is_anomaly else "ok")
        product = (p.product or "").split("/")[-1] or "-"
        score = "-" if p.score is None else f"{p.score:.3f}"
        thr = "-" if p.threshold is None else f"{p.threshold:.3f}"
        print(
            f"  {label:<32}{decision:<11}{product:<11}{p.similarity:>8.3f}{score:>9}{thr:>8}"
            f"{sum(v for k, v in p.timings_ms.items() if k != 'total'):>7.0f}"
        )
        mask = None
        if "Anomaly" in rel:
            mpath = DATA / cls / "Data" / "Masks" / rel.replace("Images/", "").replace(".JPG", ".png")
            if mpath.exists():
                mask = load_mask(str(mpath), size).numpy()
        rows.append({"label": label, "x": x, "map": p.anomaly_map, "mask": mask, "decision": decision})
    return rows


def panel(rows: list[dict], path: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2))
    for j, r in enumerate(rows):
        img = r["x"].permute(1, 2, 0).numpy()
        axes[0, j].imshow(img)
        if r["mask"] is not None:
            axes[0, j].contour(r["mask"], levels=[0.5], colors="red", linewidths=1)
        axes[0, j].set_title(r["label"].replace(", ", "\n", 1), fontsize=9)
        axes[1, j].imshow(img)
        if r["map"] is not None:
            axes[1, j].imshow(r["map"], cmap="jet", alpha=0.45)
        color = {"ok": "tab:green", "LỖI": "tab:red", "TỪ CHỐI": "tab:orange"}[r["decision"]]
        axes[1, j].set_title(r["decision"], fontsize=11, color=color, fontweight="bold")
        for a in axes[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def drift(svc: AnomalyService) -> None:
    size = svc.cfg.model.img_size
    good = [s.path for s in list_samples("visa", DATA, "pcb1", "test") if s.label == 0][:60]
    for name, level in (("luồng sạch", 0.0), ("ánh sáng đổi, mức 0,6", 0.6)):
        for p in good:
            x = relight(load_image(p, size), "mixed", level)
            pred = svc.predict(x, product="visa/pcb1")  # stream of one product, router skipped
            del pred
        st = svc.drift()["visa/pcb1"]
        flag = "CẢNH BÁO DRIFT" if st["drift"] else "bình thường"
        print(f"  {name:<24} p(CLS) = {st['cls_p']:.2g}   p(điểm) = {st['score_p']:.2g}   ->  {flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["knn", "lora"], default="knn")
    ap.add_argument("--steps", type=int, default=200, help="LoRA training steps when onboarding")
    ap.add_argument("--products", type=Path, help="load saved products (<product>/model.pt) instead")
    ap.add_argument("--splits-dir", type=Path, default=ROOT / "splits")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-pause", action="store_true")
    a = ap.parse_args()
    pause = not a.no_pause
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = None
    if a.products:
        products_dir = a.products
    else:
        tmp = Path(tempfile.mkdtemp(prefix="fscad_demo_"))
        products_dir = tmp

    print("Nạp DINOv2 ViT-S/14 (đóng băng, dùng chung cho mọi sản phẩm) ...")
    svc = AnomalyService(demo_cfg(a.method, products_dir, a.device, a.steps))
    print(f"  thiết bị: {svc.device}, phương pháp: {a.method if not a.products else 'theo checkpoint'}")
    try:
        if a.products:
            step(f"1. Nạp {len(svc.router)} sản phẩm đã đăng ký từ {products_dir}", pause)
            print("  " + ", ".join(svc.router.classes) + f"; τ = {svc.router.tau:.3f}")
        else:
            step(f"1. Đăng ký 3 sản phẩm, mỗi sản phẩm chỉ từ 4 ảnh tốt ({a.method})", pause)
            onboard(svc, a.splits_dir)
        step("2. Kiểm tra 8 ảnh: tốt, lỗi, sản phẩm lạ, ánh sáng đổi", pause)
        rows = inspect(svc)
        step("3. Bản đồ lỗi", pause)
        panel(rows, a.out / "demo_panel.png")
        print(f"  đã lưu {a.out / 'demo_panel.png'}")
        step("4. Giám sát drift trên luồng 60 ảnh tốt của pcb1", pause)
        drift(svc)
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\nXong.")


if __name__ == "__main__":
    main()
