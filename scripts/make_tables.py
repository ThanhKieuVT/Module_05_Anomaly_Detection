"""Build LaTeX tables straight from MLflow (only runs tagged final=true by default).

python scripts/make_tables.py --phase single_task --metric i_auroc --out tables/single_task.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd


def load_runs(experiment: str, phase: str | None, final_only: bool) -> pd.DataFrame:
    filt = []
    if phase:
        filt.append(f"tags.phase = '{phase}'")
    if final_only:
        filt.append("tags.final = 'true'")
    df = mlflow.search_runs(experiment_names=[experiment], filter_string=" and ".join(filt))
    if df.empty:
        raise SystemExit("no matching runs (did you tag them final=true?)")
    return df


def table(df: pd.DataFrame, metric: str) -> str:
    col = f"metrics.{metric}"
    g = df.groupby(["tags.dataset", "tags.method", "tags.shots"])[col].agg(["mean", "std"]).reset_index()
    g["cell"] = g.apply(
        lambda r: f"{100 * r['mean']:.1f} $\\pm$ {100 * (r['std'] if r['std'] == r['std'] else 0):.1f}",
        axis=1,
    )
    g["shots"] = g["tags.shots"].astype(int)
    pivot = g.pivot_table(
        index=["tags.dataset", "tags.method"], columns="shots", values="cell", aggfunc="first"
    )
    shots = sorted(pivot.columns)
    lines = [
        "\\begin{tabular}{ll" + "c" * len(shots) + "}",
        "\\toprule",
        "Dataset & Method & " + " & ".join(f"{k}-shot" for k in shots) + " \\\\",
        "\\midrule",
    ]
    for (ds, method), row in pivot.iterrows():
        lines.append(f"{ds} & {method} & " + " & ".join(str(row.get(k, "--")) for k in shots) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="fscad")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--metric", default="i_auroc")
    ap.add_argument("--all", action="store_true", help="include runs not tagged final=true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tex = table(load_runs(a.experiment, a.phase, not a.all), a.metric)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(tex)
    print(tex)
