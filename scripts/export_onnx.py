"""Export the shared (never adapted) part of the backbone to ONNX, optionally quantised to INT8.

The first ``n_blocks - lora.blocks`` blocks are identical for every product, so they can run in
ONNX Runtime while the last adapted blocks stay in PyTorch with the selected LoRA.

    python scripts/export_onnx.py model=dinov2_vits14 lora.blocks=4 +out=artifacts/onnx/prefix.onnx +int8=true
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from fscad.models.backbone import build_backbone


class Prefix(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.b = backbone

    def forward(self, x):
        h, _ = self.b.forward_prefix(x)
        return h


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out = Path(cfg.get("out", "artifacts/onnx/prefix.onnx"))
    out.parent.mkdir(parents=True, exist_ok=True)
    backbone = build_backbone(cfg).eval()
    size = cfg.model.img_size
    x = torch.randn(1, 3, size, size)
    torch.onnx.export(
        Prefix(backbone),
        x,
        str(out),
        input_names=["image"],
        output_names=["tokens"],
        dynamic_axes={"image": {0: "batch"}, "tokens": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported {out}")
    if cfg.get("int8", False):
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q = out.with_name(out.stem + "_int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QInt8)
        print(f"quantised {q}")


if __name__ == "__main__":
    main()
