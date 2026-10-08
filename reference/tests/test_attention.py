import torch

from gptoss_ref import model, rope


def openai_sdpa(cfg, x, aw, window, cos, sin):
    """OpenAI's reference sdpa (gpt_oss/torch/model.py), transcribed: full mask matrix, sinks as an extra column."""
    n = x.shape[0]
    hkv, g, d = cfg.num_key_value_heads, cfg.q_per_kv, cfg.head_dim
    q = (x @ aw["q_w"].t() + aw["q_b"]).view(n, hkv, g, d)
    k = (x @ aw["k_w"].t() + aw["k_b"]).view(n, hkv, d)
    v = (x @ aw["v_w"].t() + aw["v_b"]).view(n, hkv, d)
    q = rope.apply(q.view(n, -1, d), cos, sin).view(n, hkv, g, d)
    k = rope.apply(k, cos, sin)
    K = k[:, :, None, :].expand(-1, -1, g, -1)
    V = v[:, :, None, :].expand(-1, -1, g, -1)
    S = aw["sinks"].reshape(hkv, g, 1, 1).expand(-1, -1, n, -1)
    mask = torch.triu(x.new_full((n, n), -float("inf")), diagonal=1)
    if window > 0:
        mask += torch.tril(mask.new_full((n, n), -float("inf")), diagonal=-window)
    QK = torch.einsum("qhmd,khmd->hmqk", q, K) / d ** 0.5 + mask[None, None]
    W = torch.softmax(torch.cat([QK, S], dim=-1), dim=-1)[..., :-1]
    out = torch.einsum("hmqk,khmd->qhmd", W, V).reshape(n, -1)
    return out @ aw["o_w"].t() + aw["o_b"]


def _layer_inputs(cfg, tensors, layer, n, seed=3):
    from gptoss_ref.weights import DictWeights
    ref = model.Reference(cfg, DictWeights(tensors))
    aw = ref._attn_weights(layer)
    x = torch.randn(n, cfg.hidden_size, generator=torch.Generator().manual_seed(seed))
    cos, sin = rope.cos_sin(cfg, torch.arange(n))
    return x, aw, cos, sin


def test_chunked_attention_matches_reference_formula(tiny_cfg, tiny_weights):
    for layer, n in ((0, 37), (1, 37), (0, 5), (1, 1)):
        x, aw, cos, sin = _layer_inputs(tiny_cfg, tiny_weights, layer, n)
        want = openai_sdpa(tiny_cfg, x, aw, tiny_cfg.window(layer), cos, sin)
        for max_scores in (1 << 26, 50):       # one chunk, and many tiny chunks
            got = model.attention(tiny_cfg, x, aw, tiny_cfg.window(layer), cos, sin, max_scores=max_scores)
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_window_sees_exactly_window_keys(tiny_cfg, tiny_weights):
    # Changing a key exactly `window` positions back must not change the output; one position closer must.
    layer, n = 0, 20
    w = tiny_cfg.window(layer)
    x, aw, cos, sin = _layer_inputs(tiny_cfg, tiny_weights, layer, n)
    base = model.attention(tiny_cfg, x, aw, w, cos, sin)
    q = n - 1
    far, near = x.clone(), x.clone()
    far[q - w] += 5.0
    near[q - w + 1] += 5.0
    assert torch.equal(model.attention(tiny_cfg, far, aw, w, cos, sin)[q], base[q])
    assert not torch.allclose(model.attention(tiny_cfg, near, aw, w, cos, sin)[q], base[q])


def test_sinks_take_mass_but_add_nothing(tiny_cfg, tiny_weights):
    layer, n = 1, 9
    x, aw, cos, sin = _layer_inputs(tiny_cfg, tiny_weights, layer, n)
    no_sink = dict(aw, sinks=torch.full_like(aw["sinks"], -float("inf")))
    big_sink = dict(aw, sinks=torch.full_like(aw["sinks"], 30.0))
    a = model.attention(tiny_cfg, x, no_sink, 0, cos, sin)
    b = model.attention(tiny_cfg, x, big_sink, 0, cos, sin)
    # A dominant sink drives every head's attention output to ~0, leaving only the o_proj bias.
    torch.testing.assert_close(b, aw["o_b"].expand_as(b), rtol=0, atol=1e-6)
    assert not torch.allclose(a, b)
