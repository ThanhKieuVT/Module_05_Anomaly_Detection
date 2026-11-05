import torch

from fscad.models.lora import (
    expected_lora_param_count,
    load_lora_state_dict,
    lora_parameters,
    lora_state_dict,
    reset_lora,
    set_lora_enabled,
)
from fscad.models.memory_bank import MemoryBank
from fscad.models.router import PrototypeRouter


def test_lora_param_count_matches_formula(tiny_backbone):
    n = sum(p.numel() for p in lora_parameters(tiny_backbone.vit))
    assert n == expected_lora_param_count(tiny_backbone.embed_dim, r=4, n_blocks=2)


def test_only_lora_is_trainable(tiny_backbone):
    for p in lora_parameters(tiny_backbone.vit):
        p.requires_grad_(True)
    trainable = [n for n, p in tiny_backbone.named_parameters() if p.requires_grad]
    assert trainable and all(".lora_" in n for n in trainable)


def test_fresh_adapter_is_identity(tiny_backbone):
    x = torch.randn(2, 3, 56, 56)
    a = tiny_backbone(x, adapted=True)
    b = tiny_backbone(x, adapted=False)
    assert torch.allclose(a.patches, b.patches, atol=1e-5)
    assert a.patches.shape == (2, 16, tiny_backbone.embed_dim) and a.grid == (4, 4)


def test_lora_changes_q_v_only_and_roundtrips(tiny_backbone):
    with torch.no_grad():
        for p in lora_parameters(tiny_backbone.vit):
            p.add_(0.05 * torch.randn_like(p))
    qkv = tiny_backbone.vit.blocks[-1].attn.qkv
    x = torch.randn(1, 5, tiny_backbone.embed_dim)
    d = tiny_backbone.embed_dim
    on = qkv(x)
    qkv.enabled = False
    off = qkv(x)
    qkv.enabled = True
    assert torch.allclose(on[..., d : 2 * d], off[..., d : 2 * d])  # K untouched
    assert not torch.allclose(on[..., :d], off[..., :d])
    state = lora_state_dict(tiny_backbone.vit)
    img = torch.randn(1, 3, 56, 56)
    ref = tiny_backbone(img).patches
    reset_lora(tiny_backbone.vit)
    load_lora_state_dict(tiny_backbone.vit, state)
    assert torch.allclose(tiny_backbone(img).patches, ref, atol=1e-6)


def test_shared_prefix_matches_full_forward(tiny_backbone):
    with torch.no_grad():
        for p in lora_parameters(tiny_backbone.vit):
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(2, 3, 56, 56)
    frozen, adapted = tiny_backbone.forward_both(x)
    assert torch.allclose(frozen.cls, tiny_backbone(x, adapted=False).cls, atol=1e-5)
    assert torch.allclose(adapted.patches, tiny_backbone(x, adapted=True).patches, atol=1e-5)
    set_lora_enabled(tiny_backbone.vit, True)


def test_memory_bank_knn():
    mb = MemoryBank(k=1)
    bank = torch.nn.functional.normalize(torch.randn(50, 8).abs(), dim=-1)  # positive orthant
    mb.add(bank)
    assert torch.allclose(mb.query(bank), torch.zeros(50), atol=1e-5)
    assert (mb.query(-bank) >= 1.0 - 1e-5).all()  # opposite orthant: cosine <= 0
    mb.subsample(max_size=10)
    assert len(mb) == 10


def test_router_routes_and_rejects():
    torch.manual_seed(0)
    a, b = torch.randn(16), torch.randn(16)
    r = PrototypeRouter()
    r.register("a", a + 0.05 * torch.randn(5, 16))
    r.register("b", b + 0.05 * torch.randn(5, 16))
    val = torch.cat([a + 0.05 * torch.randn(5, 16), b + 0.05 * torch.randn(5, 16)])
    r.calibrate(val, ["a"] * 5 + ["b"] * 5, quantile=0.0)
    res = r.route(torch.stack([a, b, -a - b]))
    assert res.best[:2] == ["a", "b"]
    assert res.rejected.tolist() == [False, False, True]
    k = PrototypeRouter(n_prototypes=2)
    k.register("a", torch.cat([a + 0.01 * torch.randn(4, 16), b + 0.01 * torch.randn(4, 16)]))
    assert k.prototypes["a"].shape == (2, 16)


def test_normal_views_are_distinct_and_ids_follow_shots():
    from fscad.product import sample_normal_views

    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        # K >= 2: no duplicate (shot, view); id = shot; several shots per batch
        v = sample_normal_views(k=4, n_views=3, n=4, g=g)
        pairs = [(i, a) for i, a, _ in v]
        assert len(set(pairs)) == len(pairs) == 4
        assert all(ref == i for i, _, ref in v) and len({i for i, _, _ in v}) == 4
        v = sample_normal_views(k=2, n_views=3, n=4, g=g)
        assert len({(i, a) for i, a, _ in v}) == 4 and {i for i, _, _ in v} == {0, 1}
        # K = 1: capped at the number of distinct views, each view its own id
        v = sample_normal_views(k=1, n_views=3, n=4, g=g)
        assert len(v) == 3 and len({ref for _, _, ref in v}) == 3


def test_relight_severity_zero_is_identity():
    from fscad.data.photometric import KINDS, relight

    x = torch.rand(2, 3, 8, 8)
    for kind in KINDS:
        assert torch.allclose(relight(x, kind, 0.0), x)
        y = relight(x, kind, 1.0)
        assert y.shape == x.shape and not torch.allclose(y, x) and y.min() >= 0 and y.max() <= 1
