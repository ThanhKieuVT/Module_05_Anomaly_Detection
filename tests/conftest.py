import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from fscad.data.toy import make_toy_dataset
from fscad.models.backbone import DinoBackbone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


@pytest.fixture(scope="session")
def toy_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    make_toy_dataset(root / "toy", classes=("alpha", "beta", "gamma"), size=112, n_train=8)
    return root


@pytest.fixture()
def tiny_backbone() -> DinoBackbone:
    torch.manual_seed(0)
    return DinoBackbone("dinov2_vits14", pretrained=False, img_size=56, adapt_blocks=2, lora_r=4)


def make_cfg(toy_root: Path, tmp_path: Path, *overrides: str):
    base = [
        "dataset=toy",
        "model=tiny",
        "device=cpu",
        "tracking.enabled=false",
        f"paths.data_root={toy_root}",
        f"paths.splits_dir={tmp_path}/splits",
        f"paths.artifacts_dir={tmp_path}/artifacts",
        "train.steps=3",
        "train.batch_size=4",
        "eval.batch_size=4",
        "eval.num_workers=0",
        "eval.mask_size=56",
        "shots=2",
        "lora.blocks=2",
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=base + list(overrides))
