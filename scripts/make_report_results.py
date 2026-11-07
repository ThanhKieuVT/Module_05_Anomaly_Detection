"""Result tables and figures for the report, read from MLflow runs tagged final=true.

    python scripts/make_report_results.py --out ../report

Writes ``tables/res_*.tex``, ``figures/res_*.png`` and ``results.json`` (every number in the
tables). Configurations are selected by their logged parameters, not by run name:

* kNN        frozen DINOv2 + memory bank (phase baseline)
* LoRA       current default: lr 1e-4, 200 steps, synthetic anomalies on the object
* the development runs on pcb1-4 (seed 0) that led to that default

Only (class, seed) pairs where both kNN and LoRA finished enter a comparison, so a partial seed
does not bias the means.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#222222", "#6b6b6b", "#e4e3df"
PCB = ["pcb1", "pcb2", "pcb3", "pcb4"]


def runs() -> pd.DataFrame:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    df = mlflow.search_runs(
        experiment_names=["fscad"], filter_string="tags.final='true' and tags.dataset='visa'"
    )
    df = df[df.status == "FINISHED"].copy()
    df["cls"], df["seed"], df["shots"] = (
        df["tags.class"],
        df["tags.seed"].astype(int),
        df["tags.shots"].astype(int),
    )
    p = lambda c: df.get(f"params.{c}", pd.Series(index=df.index, dtype=object)).astype(str)  # noqa: E731
    lora = df["tags.method"] == "lora"
    fg = p("train.synthetic_on_foreground") == "True"
    syn = p("train.use_synthetic") == "True"
    lr, steps = p("train.lr"), p("train.steps")
    df["config"] = None
    df.loc[df["tags.method"] == "knn", "config"] = "knn"
    df.loc[lora & (lr == "0.0001") & (steps == "200") & syn & fg, "config"] = "lora"
    df.loc[lora & (lr == "0.001") & (steps == "200") & syn & ~fg, "config"] = "lora_lr1e-3"
    df.loc[lora & (lr == "0.001") & (steps == "200") & syn & fg, "config"] = "lora_lr1e-3_fg"
    df.loc[lora & (lr == "0.001") & (steps == "50") & syn & fg, "config"] = "lora_lr1e-3_fg_50steps"
    df.loc[lora & (lr == "0.001") & (steps == "200") & ~syn & fg, "config"] = "lora_lr1e-3_fg_nosynth"
    return df[df.config.notna()]


def paired(df: pd.DataFrame, shots: int, a="knn", b="lora") -> pd.DataFrame:
    """(cls, seed) rows with metrics of both configs, suffixes _a/_b."""
    cols = [
        "cls",
        "seed",
        "metrics.i_auroc",
        "metrics.au_pro",
        "metrics.recall_at_delta",
        "metrics.fpr_at_delta",
    ]
    x = df[(df.config == a) & (df.shots == shots)][cols]
    y = df[(df.config == b) & (df.shots == shots)][cols]
    m = x.merge(y, on=["cls", "seed"], suffixes=("_a", "_b"))
    return m.sort_values(["cls", "seed"])


def pm(v: pd.Series) -> str:
    v = 100 * v
    return f"{v.mean():.1f}" if len(v) < 2 else f"{v.mean():.1f} $\\pm$ {v.std(ddof=1):.1f}"


def signed(x: float) -> str:
    x = round(x, 1) + 0.0  # no "-0.0"
    return f"{'+' if x >= 0 else '$-$'}{abs(x):.1f}"


def main_table(m: pd.DataFrame) -> tuple[str, dict]:
    rows, out = [], {}
    for cls, g in m.groupby("cls"):
        d = 100 * (g["metrics.i_auroc_b"] - g["metrics.i_auroc_a"])
        win = "\\textbf{" if d.mean() >= 1 else "{"
        name = cls.replace("_", "\\_")
        rows.append(
            f"{name} & {len(g)} & {pm(g['metrics.i_auroc_a'])} & {pm(g['metrics.i_auroc_b'])} & "
            f"{win}{signed(d.mean())}}} & {pm(g['metrics.au_pro_a'])} & {pm(g['metrics.au_pro_b'])} \\\\"
        )
        out[cls] = {
            "seeds": g.seed.tolist(),
            "i_auroc_knn": g["metrics.i_auroc_a"].round(4).tolist(),
            "i_auroc_lora": g["metrics.i_auroc_b"].round(4).tolist(),
            "au_pro_knn": g["metrics.au_pro_a"].round(4).tolist(),
            "au_pro_lora": g["metrics.au_pro_b"].round(4).tolist(),
        }
    # mean over classes of the per-seed class means; +- is the spread of that mean across seeds
    seed_means = m.groupby("seed")[[c for c in m.columns if c.startswith("metrics.")]].mean()
    full = m.groupby("seed").size() == m.cls.nunique()
    sm = seed_means[full.reindex(seed_means.index).values]
    d = 100 * (sm["metrics.i_auroc_b"] - sm["metrics.i_auroc_a"])
    rows.append("\\midrule")
    rows.append(
        f"\\textbf{{Trung bình 12 lớp}} & {len(sm)} & {pm(sm['metrics.i_auroc_a'])} & {pm(sm['metrics.i_auroc_b'])} & "
        f"\\textbf{{{signed(d.mean())}}} & {pm(sm['metrics.au_pro_a'])} & {pm(sm['metrics.au_pro_b'])} \\\\"
    )
    out["mean_over_complete_seeds"] = {
        "seeds": sm.index.tolist(),
        **{k.replace("metrics.", ""): sm[k].round(4).tolist() for k in sm.columns},
    }
    tex = "\n".join(
        [
            "\\begin{tabular}{l c cc c cc}",
            "\\toprule",
            "\\headerrowcolor",
            "& & \\multicolumn{3}{c}{\\textbf{I-AUROC (\\%)}} & \\multicolumn{2}{c}{\\textbf{AU-PRO (\\%)}} \\\\",
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
            "\\textbf{Lớp} & \\textbf{Seed} & \\textbf{kNN} & \\textbf{LoRA} & \\textbf{$\\Delta$} & \\textbf{kNN} & \\textbf{LoRA} \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return tex, out


def dev_table(df: pd.DataFrame) -> tuple[str, dict]:
    configs = [
        ("knn", "kNN, DINOv2 đóng băng"),
        ("lora_lr1e-3", "(A) LoRA ban đầu: lr $10^{-3}$, 200 bước, lỗi tổng hợp trên cả ảnh"),
        ("lora_lr1e-3_fg", "(B) = (A), lỗi tổng hợp chỉ trên vật thể"),
        ("lora_lr1e-3_fg_nosynth", "(C) = (B), bỏ lỗi tổng hợp (chỉ compactness)"),
        ("lora_lr1e-3_fg_50steps", "(D) = (B), 50 bước"),
        ("lora", "(E) = (B), lr $10^{-4}$ (\\textbf{mặc định hiện tại})"),
    ]
    d = df[(df.shots == 4) & (df.seed == 0) & df.cls.isin(PCB)]
    rows, out = [], {}
    for key, label in configs:
        g = d[d.config == key].set_index("cls")["metrics.i_auroc"]
        if not set(PCB) <= set(g.index):
            continue
        vals = [100 * g[c] for c in PCB]
        rows.append(f"{label} & " + " & ".join(f"{v:.1f}" for v in vals) + f" & {np.mean(vals):.1f} \\\\")
        out[key] = dict(zip(PCB, [round(v / 100, 4) for v in vals], strict=True))
    knn_row = rows[0]
    rows[1:1] = ["\\midrule"]
    tex = "\n".join(
        [
            "\\begin{tabularx}{\\textwidth}{X ccccc}",
            "\\toprule",
            "\\headerrowcolor",
            "\\textbf{Cấu hình} & \\textbf{pcb1} & \\textbf{pcb2} & \\textbf{pcb3} & \\textbf{pcb4} & \\textbf{TB} \\\\",
            "\\midrule",
            knn_row,
            *rows[1:],
            "\\bottomrule",
            "\\end{tabularx}",
        ]
    )
    return tex, out


def ops_table(df: pd.DataFrame, m: pd.DataFrame) -> tuple[str, dict]:
    keys = m[["cls", "seed"]]
    rows, out = [], {}
    for key, label in (("knn", "kNN"), ("lora", "LoRA")):
        g = df[(df.config == key) & (df.shots == 4)].merge(keys, on=["cls", "seed"])
        r = {
            "recall_at_delta": g["metrics.recall_at_delta"].mean(),
            "fpr_at_delta": g["metrics.fpr_at_delta"].mean(),
            "latency_ms": g["metrics.latency_ms_per_image"].median(),
            "mb_per_class": g["metrics.mb_per_class"].mean(),
            "fit_seconds": g["metrics.fit_seconds"].median(),
            "train_seconds": g["metrics.train_seconds"].median() if "metrics.train_seconds" in g else np.nan,
            "runs": len(g),
        }
        out[key] = {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}
        train = "--" if pd.isna(r["train_seconds"]) else f"{r['train_seconds']:.0f}"
        rows.append(
            f"{label} & {100 * r['recall_at_delta']:.1f} & {100 * r['fpr_at_delta']:.1f} & {r['latency_ms']:.0f} & "
            f"{r['mb_per_class']:.1f} & {train} & {r['fit_seconds']:.0f} \\\\"
        )
    tex = "\n".join(
        [
            "\\begin{tabular}{l cc c c cc}",
            "\\toprule",
            "\\headerrowcolor",
            "& \\multicolumn{2}{c}{\\textbf{Tại ngưỡng $\\delta$ (\\%)}} & \\textbf{Độ trễ} & \\textbf{Bộ nhớ} & "
            "\\multicolumn{2}{c}{\\textbf{Onboard (giây)}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}",
            "\\textbf{Phương pháp} & \\textbf{Recall} & \\textbf{FPR} & \\textbf{ms/ảnh} & \\textbf{MB/lớp} & "
            "\\textbf{train} & \\textbf{tổng} \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return tex, out


def delta_figure(m: pd.DataFrame, path: Path) -> None:
    """Per-class I-AUROC gain of LoRA over kNN: one dot per seed, a bar for the mean."""
    m = m.assign(d=100 * (m["metrics.i_auroc_b"] - m["metrics.i_auroc_a"]))
    order = m.groupby("cls").d.mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(order))
    means = [m[m.cls == c].d.mean() for c in order]
    ax.barh(y, means, height=0.55, color=ORANGE, alpha=0.35, zorder=2, label="trung bình các seed")
    for i, c in enumerate(order):
        g = m[m.cls == c]
        ax.scatter(g.d, np.full(len(g), i), s=22, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.scatter([], [], s=22, color=ORANGE, label="từng seed")
    ax.axvline(0, color=INK, lw=0.8, zorder=1)
    ax.axvline(1, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.text(1.2, -0.75, "+1 điểm (tiêu chí H1)", color=MUTED, fontsize=7.5, va="center")
    ax.set_yticks(y, order, fontsize=8.5)
    ax.set_xlabel("I-AUROC LoRA − kNN (điểm %)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def seed_figure(df: pd.DataFrame, m: pd.DataFrame, path: Path) -> None:
    """kNN vs LoRA I-AUROC per class, seeds as separate marks, both methods on one axis."""
    order = m.groupby("cls")["metrics.i_auroc_a"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for col, color, label, off in (
        ("metrics.i_auroc_a", BLUE, "kNN", -0.13),
        ("metrics.i_auroc_b", ORANGE, "LoRA", 0.13),
    ):
        for i, c in enumerate(order):
            v = 100 * m[m.cls == c][col]
            ax.plot([v.min(), v.max()], [i + off, i + off], color=color, lw=2, alpha=0.5, zorder=2)
            ax.scatter(
                v,
                np.full(len(v), i + off),
                s=20,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label=label if i == 0 else None,
            )
    ax.set_yticks(range(len(order)), order, fontsize=8.5)
    ax.set_xlabel("I-AUROC (%), mỗi chấm là một seed", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/report_results")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = runs()
    m = paired(df, 4)
    res = {}
    for name, (tex, data) in {
        "main": main_table(m),
        "dev": dev_table(df),
        "ops": ops_table(df, m),
    }.items():
        (out / "tables" / f"res_{name}.tex").write_text(tex + "\n")
        res[name] = data
    delta_figure(m, out / "figures" / "res_delta.png")
    seed_figure(df, m, out / "figures" / "res_seeds.png")
    res["pairs"] = {"n": len(m), "seeds_per_class": m.groupby("cls").size().to_dict()}
    res["commits"] = df.groupby("config")["tags.git_commit"].unique().apply(sorted).to_dict()
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
