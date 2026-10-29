"""Print (or run) the Hydra multirun commands of the experiment matrix in the project plan.

    python scripts/run_matrix.py              # print the commands
    python scripts/run_matrix.py --run main   # run one group
Groups: baseline, main, ablation, continual.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess

SHOTS = "1,4,16"  # 2-week plan; the full plan uses 1,2,4,8,16
SEEDS = "[0,1,2]"
ABLATION_CLASSES = "[bottle,screw,carpet,transistor,pill]"

GROUPS = {
    "baseline": [
        f"python -m fscad.evaluate -m phase=baseline method=knn dataset=mvtec_ad,visa shots={SHOTS} seeds={SEEDS}",
    ],
    "main": [
        f"python -m fscad.evaluate -m phase=single_task method=lora dataset=mvtec_ad,visa shots={SHOTS} seeds={SEEDS}",
    ],
    "ablation": [
        f"python -m fscad.evaluate -m phase=ablation dataset=mvtec_ad classes={ABLATION_CLASSES} shots=4 lora.r=4,8,16 lora.blocks=2,4",
        f"python -m fscad.evaluate -m phase=ablation dataset=mvtec_ad classes={ABLATION_CLASSES} shots=4 train.use_synthetic=false",
        f"python -m fscad.evaluate -m phase=ablation dataset=mvtec_ad classes={ABLATION_CLASSES} shots=4 train.synthetic=perlin,cutpaste",
        f"python -m fscad.evaluate -m phase=ablation dataset=mvtec_ad classes={ABLATION_CLASSES} shots=4 model=dinov2_vitb14",
    ],
    "continual": [
        # reuse the single-task adapters (same run_name) so continual costs only routing + scoring
        "python -m fscad.continual -m phase=continual reuse_artifacts=true shots=4 continual.datasets=[mvtec_ad] "
        "continual.holdout=[] continual.order=alphabetical,random",
        "python -m fscad.continual phase=continual reuse_artifacts=true shots=4 continual.datasets=[mvtec_ad,visa] "
        "continual.holdout=[visa/candle,visa/fryum,visa/pcb1]",
    ],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(GROUPS))
    a = ap.parse_args()
    for g, cmds in GROUPS.items():
        if a.run and g != a.run:
            continue
        print(f"# {g}")
        for c in cmds:
            print(c)
            if a.run:
                subprocess.run(shlex.split(c), check=True)
