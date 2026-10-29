import numpy as np
import torch

from fscad.data.datasets import list_samples
from fscad.data.splits import load_split, make_split
from fscad.data.synthetic import patch_labels, synthesize
from fscad.metrics import au_pro, f1_max, forgetting_measure, image_auroc, pixel_auroc
from fscad.training.losses import compactness_margin_loss


def test_image_metrics():
    y = np.array([0, 0, 1, 1])
    assert image_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert f1_max(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_pixel_metrics_perfect_and_random():
    masks = np.zeros((4, 32, 32))
    masks[0, 5:10, 5:10] = 1
    masks[1, 20:30, 2:6] = 1
    assert pixel_auroc(masks, masks.copy()) == 1.0
    assert au_pro(masks, masks + 0.0) > 0.99
    rnd = np.random.default_rng(0).random(masks.shape)
    assert 0.0 < au_pro(masks, rnd) < 0.4  # random maps give ~ fpr_limit / 2


def test_forgetting_measure():
    acc = np.array([[0.9, np.nan], [0.7, 0.8]])
    assert abs(forgetting_measure(acc) - 0.2) < 1e-9
    assert forgetting_measure(np.array([[0.9, np.nan], [0.9, 0.8]])) == 0.0


def test_toy_dataset_and_splits(toy_root, tmp_path):
    root = toy_root / "toy"
    test = list_samples("toy", root, "alpha", "test")
    assert {s.label for s in test} == {0, 1}
    assert all(s.mask for s in test if s.label == 1)
    s1 = make_split("toy", root, "alpha", 2, seed=0)
    s2 = make_split("toy", root, "alpha", 4, seed=0)
    assert s2["train"][:2] == s1["train"]  # nested shots
    assert not set(s1["train"]) & set(s1["val"])
    loaded = load_split(tmp_path, "toy", "alpha", 2, 0, root=root)
    assert loaded["train"] == s1["train"]


def test_synthetic_anomalies_and_patch_labels():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 112, 112)
    for kind in ("perlin", "cutpaste"):
        out, mask = synthesize(img, img.flip(-1), kind, g)
        assert out.shape == img.shape and 0 < mask.mean() < 0.5
        assert (out.min() >= 0) and (out.max() <= 1)
    lab = patch_labels(mask[None], (8, 8))
    assert lab.shape == (1, 64) and (lab == 1).any() and (lab == 0).any()


def test_loss_separates_normal_and_anomalous():
    torch.manual_seed(0)
    feats = torch.randn(4, 16, 8, requires_grad=True)
    labels = torch.zeros(4, 16)
    labels[3, :4] = 1
    out = compactness_margin_loss(feats, labels, torch.tensor([0, 1, 2, -1]), center="knn")
    out.loss.backward()
    assert out.loss.item() > 0 and feats.grad is not None
    out_mean = compactness_margin_loss(feats, labels, torch.tensor([0, 1, 2, -1]), center="mean")
    assert torch.isfinite(out_mean.loss)
