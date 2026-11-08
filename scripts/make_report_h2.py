"""H2 table and figure for the report, from the continual runs' result files.

    python scripts/make_report_h2.py --runs outputs --out ../report

Reads every ``continual_result.json`` under ``--runs`` whose Hydra config says
``phase=continual`` and ``tracking.final=true``, groups runs by (method, order) and reports the mean and
standard deviation over seeds. Writes ``tables/res_h2.tex``, ``figures/res_h2_steps.png`` and
``h2.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#222222",
    "#6b6b6b",
    "#e4e3df",
)
COLS = [
    ("avg_i_auroc_oracle", "oracle", 100),
    ("avg_i_auroc_real_alarm", "thực (alarm)", 100),
    ("avg_i_auroc_real_miss", "thực (miss)", 100),
    ("fm_real_alarm", "FM (điểm)", 100),
    ("routing_acc_argmax", "chọn đúng (\\%)", 100),
    ("reject_rate_normal", "ảnh tốt", 100),
    ("reject_rate_anomalous", "ảnh lỗi", 100),
    ("true_reject_rate", "SP lạ", 100),
]


def load_runs(root: Path) -> list[dict]:
    runs = []
    for res in sorted(root.glob("**/continual_result.json")):
        cfg = OmegaConf.load(res.parent / ".hydra" / "config.yaml")
        if cfg.phase != "continual" or not cfg.tracking.final or "visa" not in cfg.continual.datasets:
            continue
        r = json.loads(res.read_text())
        runs.append(
            {
                **r,
                "task_order": r["order"],
                "method": cfg.method,
                "order": cfg.continual.order,
                "seed": int(cfg.seeds[0]),
                "dir": str(res.parent),
            }
        )
    return runs


def fmt(v: list[float], scale: float) -> str:
    v = np.asarray(v, dtype=float) * scale
    if len(v) < 2:
        return f"{v.mean():.1f}"
    return f"{v.mean():.1f} $\\pm$ {v.std(ddof=1):.1f}"


def table(groups: dict) -> tuple[str, dict]:
    rows, out = [], {}
    for (method, order), runs in groups.items():
        label = {"knn": "kNN", "lora": "LoRA"}[method] + (
            "" if order == "alphabetical" else ", thứ tự ngẫu nhiên"
        )
        label += f" ({len(runs)} seed)" if len(runs) > 1 else " (seed 0)"
        cells = [fmt([r["final"][k] for r in runs], s) for k, _, s in COLS]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
        out[f"{method}-{order}"] = {k: [r["final"][k] for r in runs] for k, _, _ in COLS} | {
            "seeds": [r["seed"] for r in runs],
            "holdout": [r["holdout_rejection"] for r in runs],
        }
    head = " & ".join(f"\\textbf{{{n}}}" for _, n, _ in COLS)
    tex = "\n".join(
        [
            "\\begin{tabular}{l ccc c c ccc}",
            "\\toprule",
            "\\headerrowcolor",
            "& \\multicolumn{3}{c}{\\textbf{I-AUROC trung bình (\\%)}} & \\textbf{Quên} & "
            "\\textbf{Router} & \\multicolumn{3}{c}{\\textbf{Router từ chối (\\%)}} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){7-9}",
            f"\\textbf{{Phương pháp}} & {head} \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return re.sub(r"(\d)\.(\d)", r"\1{,}\2", tex), out


def steps_figure(groups: dict, path: Path) -> None:
    """Left: mean I-AUROC over seen products, real (alarm) vs oracle, kNN and LoRA.
    Right: rejection rate of normal and anomalous test images as products are added (the router reads
    frozen CLS tokens, so it is the same for kNN and LoRA)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    ax = axes[0]
    for method, color, name in (("knn", BLUE, "kNN"), ("lora", ORANGE, "LoRA")):
        runs = groups.get((method, "alphabetical"), [])
        if not runs:
            continue
        x = np.arange(1, len(runs[0]["steps"]) + 1)
        for key, ls, lab in (
            ("avg_i_auroc_oracle", "--", "oracle"),
            ("avg_i_auroc_real_alarm", "-", "router thực"),
        ):
            y = 100 * np.mean([[s[key] for s in r["steps"]] for r in runs], axis=0)
            ax.plot(
                x,
                y,
                ls=ls,
                color=color,
                lw=2,
                marker="o" if ls == "-" else None,
                ms=4,
                mec="white",
                label=f"{name}, {lab}",
            )
    ax.set_xlabel("số sản phẩm đã đăng ký", fontsize=8.5)
    ax.set_title("I-AUROC trung bình các SP đã đăng ký (%)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    runs = groups.get(("lora", "alphabetical"), [])
    ax = axes[1]
    if runs:
        x = np.arange(1, len(runs[0]["steps"]) + 1)
        for key, color, lab in (
            ("reject_rate_anomalous", YELLOW, "ảnh lỗi"),
            ("reject_rate_normal", AQUA, "ảnh tốt"),
        ):
            y = 100 * np.mean([[s[key] for s in r["steps"]] for r in runs], axis=0)
            ax.plot(x, y, color=color, lw=2, marker="o", ms=4, mec="white", label=lab)
            ax.annotate(
                f"{y[-1]:.1f}".replace(".", ","),
                (x[-1], y[-1]),
                textcoords="offset points",
                xytext=(6, 0),
                va="center",
                fontsize=7.5,
                color=INK,
            )
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_xlabel("số sản phẩm đã đăng ký", fontsize=8.5)
    ax.set_title("Tỷ lệ ảnh test bị router từ chối (%)", fontsize=9, loc="left", color=INK)
    for a in axes:
        a.set_xticks(range(1, 11))
        a.tick_params(labelsize=8, colors=INK)
        a.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/report_results"))
    a = ap.parse_args()
    runs = load_runs(a.runs)
    groups: dict = {}
    for r in sorted(runs, key=lambda r: (r["method"], r["order"], r["seed"])):
        groups.setdefault((r["method"], r["order"]), []).append(r)
    (a.out / "tables").mkdir(parents=True, exist_ok=True)
    (a.out / "figures").mkdir(parents=True, exist_ok=True)
    tex, data = table(groups)
    (a.out / "tables" / "res_h2.tex").write_text(tex + "\n")
    steps_figure(groups, a.out / "figures" / "res_h2_steps.png")
    data["runs"] = [{k: r[k] for k in ("method", "order", "seed", "dir")} for r in runs]
    (a.out / "h2.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(tex)
    print(json.dumps(data, indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
