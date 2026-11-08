"""H3 table and figures for the report, from the drift runs' CSV files.

    python scripts/make_report_h3.py --runs outputs --out ../report --example visa/pcb1/Data/Images/Normal/0000.JPG

Reads every ``drift_levels.csv`` / ``drift_summary.csv`` under ``--runs`` whose Hydra config says
``phase=drift`` and ``tracking.final=true``. Writes ``tables/res_h3.tex``, ``figures/res_h3.png``
(I-AUROC change per severity, and alarm level against quality-drop level per class),
``figures/drift_example.png`` (one image at a few severities) and ``h3.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from fscad.data.datasets import load_image  # noqa: E402
from fscad.data.photometric import relight  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#222222",
    "#6b6b6b",
    "#e4e3df",
)


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load(root: Path) -> tuple[list[dict], list[dict], dict]:
    """Rows and summaries of the latest final drift run (one run covers every class)."""
    found = []
    for summ in sorted(root.glob("**/drift_summary.csv")):
        cfg = OmegaConf.load(summ.parent / ".hydra" / "config.yaml")
        if cfg.phase == "drift" and cfg.tracking.final:
            found.append((summ.parent, cfg))
    if not found:
        raise SystemExit("no final drift run found")
    run, cfg = found[-1]
    rows = read_csv(run / "drift_levels.csv")
    for r in rows:
        r["level"], r["i_auroc"] = float(r["level"]), float(r["i_auroc"])
        r["drift"] = r["drift"] == "True"
    summ = read_csv(run / "drift_summary.csv")
    for s in summ:
        s["clean_i_auroc"] = float(s["clean_i_auroc"])
        for k in ("alarm_level", "drop_level"):
            s[k] = float(s[k]) if s[k] not in ("", "None") else None
        for k in ("false_alarm_clean", "alarm_before_drop"):
            s[k] = s[k] == "True"
    meta = {
        "dir": str(run),
        "method": cfg.method,
        "seed": int(cfg.seeds[0]),
        "shots": int(cfg.shots),
        "kind": cfg.drift.kind,
        "levels": list(cfg.drift.levels),
        "delta": float(cfg.drift.delta),
        "window": int(cfg.drift.window),
        "alpha": float(cfg.drift.alpha),
        "anomaly_rate": float(cfg.drift.anomaly_rate),
    }
    return rows, summ, meta


def lv(v: float | None) -> str:
    return "--" if v is None else f"{v:.1f}"


def signed(v: float) -> str:
    v = round(v, 1)
    return f"{'$+$' if v > 0 else '$-$' if v < 0 else ''}{abs(v):.1f}"


def table(rows: list[dict], summ: list[dict]) -> str:
    out = []
    for s in summ:
        cls = s["class"]
        r = {x["level"]: x["i_auroc"] for x in rows if x["class"] == cls}
        top = max(r)
        verdict = "\\yes" if s["alarm_before_drop"] else "\\no"
        out.append(
            f"{cls.replace('_', chr(92) + '_')} & {100 * s['clean_i_auroc']:.1f} & {100 * r[top]:.1f} & "
            f"{signed(100 * (r[top] - s['clean_i_auroc']))} & {'có' if s['false_alarm_clean'] else 'không'} & "
            f"{lv(s['alarm_level'])} & {lv(s['drop_level'])} & {verdict} \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{tabular}{l ccc c cc c}",
            "\\toprule",
            "\\headerrowcolor",
            "& \\multicolumn{3}{c}{\\textbf{I-AUROC (\\%)}} & \\textbf{Cảnh báo} & "
            "\\multicolumn{2}{c}{\\textbf{Mức drift đầu tiên}} & \\textbf{Cảnh báo} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){6-7}",
            "\\textbf{Lớp} & \\textbf{sạch} & \\textbf{mức 1,0} & \\textbf{chênh} & \\textbf{trên luồng sạch} & "
            "\\textbf{cảnh báo} & \\textbf{giảm $>$2 điểm} & \\textbf{trước khi giảm} \\\\",
            "\\midrule",
            *out,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return re.sub(r"(\d)\.(\d)", r"\1{,}\2", tex)


def figure(rows: list[dict], summ: list[dict], meta: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    levels = sorted({r["level"] for r in rows})
    curves = []
    for s in summ:
        y = [
            100 * (r["i_auroc"] - s["clean_i_auroc"])
            for lv_ in levels
            for r in rows
            if r["class"] == s["class"] and r["level"] == lv_
        ]
        curves.append(y)
        ax.plot(levels, y, color=MUTED, lw=0.8, alpha=0.55, zorder=2)
    ax.plot(
        levels,
        np.median(curves, axis=0),
        color=ORANGE,
        lw=2.2,
        marker="o",
        ms=4,
        mec="white",
        zorder=3,
        label="trung vị 12 lớp",
    )
    ax.axhline(-100 * meta["delta"], color=INK, lw=0.9, ls="--", zorder=1)
    ax.annotate(
        f"$-${100 * meta['delta']:.0f} điểm",
        (levels[0], -100 * meta["delta"]),
        textcoords="offset points",
        xytext=(2, -10),
        fontsize=7.5,
        color=INK,
    )
    ax.set_xlabel("mức drift ánh sáng", fontsize=8.5)
    ax.set_title("Thay đổi I-AUROC so với ảnh sạch (điểm)", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = axes[1]
    order = sorted(summ, key=lambda s: (s["drop_level"] or 9, s["class"]))
    top = max(levels)
    for i, s in enumerate(order):
        a, d = s["alarm_level"], s["drop_level"]
        dd = d if d is not None else top + 0.12
        if a is not None:
            ax.plot([a, dd], [i, i], color=GRID, lw=2.5, zorder=1, solid_capstyle="round")
            ax.scatter([a], [i], color=AQUA, s=34, zorder=3, edgecolor="white", lw=0.8)
        ax.scatter(
            [dd],
            [i],
            color=YELLOW,
            s=34,
            zorder=3,
            edgecolor="white",
            lw=0.8,
            marker="o" if d is not None else ">",
        )
    ax.scatter([], [], color=AQUA, s=34, label="monitor cảnh báo")
    ax.scatter([], [], color=YELLOW, s=34, label=f"I-AUROC giảm $>${100 * meta['delta']:.0f} điểm")
    ax.scatter(
        [], [], color=YELLOW, s=34, marker=">", label=f"không giảm tới mức {top:.1f}".replace(".", ",")
    )
    ax.set_yticks(range(len(order)), [s["class"] for s in order], fontsize=7.5)
    ax.set_xlim(-0.05, top + 0.22)
    ax.set_xticks(np.linspace(0, top, 6))
    ax.set_xlabel("mức drift ánh sáng", fontsize=8.5)
    ax.set_title("Mức drift: cảnh báo và chất lượng giảm", fontsize=9, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2)
    for a_ in axes:
        a_.tick_params(labelsize=8, colors=INK)
        a_.grid(axis="x" if a_ is ax else "y", color=GRID, lw=0.6, zorder=0)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def example(image: Path, kind: str, levels: list[float], path: Path, size: int = 448) -> None:
    x = load_image(str(image), size)
    fig, axes = plt.subplots(1, len(levels), figsize=(7.4, 7.4 / len(levels) + 0.35))
    for a, s in zip(axes, levels, strict=True):
        a.imshow(relight(x, kind, s).permute(1, 2, 0).numpy())
        a.set_title(f"mức {s:.1f}".replace(".", ","), fontsize=8.5, color=INK)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("outputs"))
    ap.add_argument("--out", type=Path, default=Path("outputs/report_results"))
    ap.add_argument("--example", type=Path, default=None, help="image to show at several drift levels")
    a = ap.parse_args()
    rows, summ, meta = load(a.runs)
    (a.out / "tables").mkdir(parents=True, exist_ok=True)
    (a.out / "figures").mkdir(parents=True, exist_ok=True)
    tex = table(rows, summ)
    (a.out / "tables" / "res_h3.tex").write_text(tex + "\n")
    figure(rows, summ, meta, a.out / "figures" / "res_h3.png")
    if a.example:
        example(a.example, meta["kind"], [0.0, 0.3, 0.6, 1.0], a.out / "figures" / "drift_example.png")
    (a.out / "h3.json").write_text(
        json.dumps({"meta": meta, "summary": summ, "levels": rows}, indent=2, ensure_ascii=False)
    )
    print(tex)
    for s in summ:
        print(s)


if __name__ == "__main__":
    main()
