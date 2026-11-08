"""Tables for the report from ``scripts/rescore_variants.py`` output.

    python scripts/make_report_rescore.py --csv outputs/rescore/visa_k4.csv --out ../report

Writes ``tables/res_variants.tex`` (I-AUROC of each score / mask variant: all 12 classes, the 11
classes without macaroni2, and macaroni2 alone), ``tables/res_delta.tex`` (FPR and recall of the
two threshold rules for the default variant) and ``rescore.json``. Values are the mean over seeds
of the per-seed mean over classes, with the standard deviation over seeds.

For the q99 rule, the expected FPR on exchangeable normal images follows from order statistics:
``np.quantile`` at 0.99 of n = 30 values interpolates between the two largest validation scores,
which a new normal image exceeds with probability 2/(n+1) and 1/(n+1); the expected FPR is taken
as the linear interpolation of the two (about 4.2% for n = 30).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

VARIANTS = {
    "default": "mặt nạ giãn, top 1\\% (đang dùng)",
    "nodilate": "mặt nạ không giãn, top 1\\%",
    "max": "mặt nạ giãn, patch lớn nhất",
    "nodilate_max": "mặt nạ không giãn, patch lớn nhất",
}
METHODS = {"knn": "kNN", "lora": "LoRA"}


def load(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        for k in ("i_auroc", "fpr_q99", "recall_q99", "fpr_mean3sd", "recall_mean3sd"):
            r[k] = float(r[k])
    return rows


def agg(rows: list[dict], key: str, keep=lambda r: True) -> tuple[float, float, int]:
    """Mean and std over seeds of the per-seed class mean (x100)."""
    seeds = sorted({r["seed"] for r in rows})
    per = [100 * np.mean([r[key] for r in rows if r["seed"] == s and keep(r)]) for s in seeds]
    return float(np.mean(per)), float(np.std(per, ddof=1)) if len(per) > 1 else 0.0, len(per)


def cell(m: float, s: float) -> str:
    return f"{m:.1f} $\\pm$ {s:.1f}"


def expected_fpr_q99(n: int) -> float:
    pos = 0.99 * (n - 1)  # 0-based position of np.quantile's linear interpolation
    frac = pos - np.floor(pos)
    # exceed X_(n-1) with prob 2/(n+1), X_(n) with prob 1/(n+1); interpolate on the threshold position
    return float((1 - frac) * 2 / (n + 1) + frac * 1 / (n + 1))


def variants_table(rows: list[dict], out: dict) -> str:
    lines = []
    for v, name in VARIANTS.items():
        cells = []
        for m in METHODS:
            sub = [r for r in rows if r["method"] == m and r["variant"] == v]
            for label, keep in (
                ("all", lambda r: True),
                ("no_mac2", lambda r: r["cls"] != "macaroni2"),
                ("mac2", lambda r: r["cls"] == "macaroni2"),
            ):
                mean, sd, _ = agg(sub, "i_auroc", keep)
                out.setdefault("i_auroc", {}).setdefault(m, {}).setdefault(v, {})[label] = [mean, sd]
                cells.append(cell(mean, sd))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    sub = "\\textbf{12 lớp} & \\textbf{bỏ macaroni2} & \\textbf{macaroni2}"
    tex = "\n".join(
        [
            "\\begin{tabular}{l ccc ccc}",
            "\\toprule",
            "\\headerrowcolor",
            "& \\multicolumn{3}{c}{\\textbf{kNN}} & \\multicolumn{3}{c}{\\textbf{LoRA}} \\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
            f"\\textbf{{Điểm ảnh}} & {sub} & {sub} \\\\",
            "\\midrule",
            *lines,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return re.sub(r"(\d)\.(\d)", r"\1{,}\2", tex)


def delta_table(rows: list[dict], out: dict) -> str:
    lines = []
    for rule, name in (("q99", "phân vị 99\\% (đang dùng)"), ("mean3sd", "trung bình + 3 độ lệch chuẩn")):
        cells = []
        for m in METHODS:
            sub = [r for r in rows if r["method"] == m and r["variant"] == "default"]
            for key in (f"fpr_{rule}", f"recall_{rule}"):
                mean, sd, _ = agg(sub, key)
                out.setdefault("delta", {}).setdefault(m, {})[key] = [mean, sd]
                cells.append(cell(mean, sd))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    sub = "\\textbf{FPR} & \\textbf{recall}"
    tex = "\n".join(
        [
            "\\begin{tabular}{l cc cc}",
            "\\toprule",
            "\\headerrowcolor",
            "& \\multicolumn{2}{c}{\\textbf{kNN (\\%)}} & \\multicolumn{2}{c}{\\textbf{LoRA (\\%)}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
            f"\\textbf{{Quy tắc đặt $\\delta$}} & {sub} & {sub} \\\\",
            "\\midrule",
            *lines,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    return re.sub(r"(\d)\.(\d)", r"\1{,}\2", tex)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("outputs/rescore/visa_k4.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/report_results"))
    ap.add_argument("--n-val", type=int, default=30)
    a = ap.parse_args()
    rows = load(a.csv)
    (a.out / "tables").mkdir(parents=True, exist_ok=True)
    out: dict = {"n_rows": len(rows), "expected_fpr_q99": expected_fpr_q99(a.n_val)}
    t1 = variants_table(rows, out)
    t2 = delta_table(rows, out)
    (a.out / "tables" / "res_variants.tex").write_text(t1 + "\n")
    (a.out / "tables" / "res_delta.tex").write_text(t2 + "\n")
    per_class = {}
    for m in METHODS:
        for cls in sorted({r["cls"] for r in rows}):
            per_class[f"{m}/{cls}"] = {
                v: round(
                    100
                    * float(
                        np.mean(
                            [
                                r["i_auroc"]
                                for r in rows
                                if r["method"] == m and r["cls"] == cls and r["variant"] == v
                            ]
                        )
                    ),
                    2,
                )
                for v in VARIANTS
            }
    out["per_class"] = per_class
    (a.out / "rescore.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(t1, t2, sep="\n\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_class"}, indent=1))
    for k, v in per_class.items():
        print(k, v)


if __name__ == "__main__":
    main()
