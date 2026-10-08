"""Kernel tests: each CUDA kernel against a float64 torch implementation of the reference semantics.

    GPTOSS_LIB=build/libgptoss.so PYTHONPATH=reference python -m pytest -q runtime/tests
"""
import ctypes
import math
import os

import pytest
import torch

from gptoss_ref import mxfp4, rope
from gptoss_ref.config import GptOssConfig

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
LIB = ctypes.CDLL(os.environ.get("GPTOSS_LIB", "build/libgptoss.so"))
LIB.gptoss_last_error.restype = ctypes.c_char_p
P, I32, F32 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
for name, args in {
    "gptoss_kernel_gemv_bf16": [P, P, P, P, P, I32, I32],
    "gptoss_kernel_rmsnorm": [P, P, P, I32, F32],
    "gptoss_kernel_rope_append": [P, P, P, I32, P, P, I32],
    "gptoss_kernel_attention": [P, P, P, P, I32, I32, I32, P],
    "gptoss_kernel_router": [P, P, P, I32, I32, I32, P, P],
    "gptoss_kernel_moe": [P, P, P, P, P, P, P, P, P, P, I32, I32, I32, F32, F32],
}.items():
    getattr(LIB, name).argtypes = args

DEV = "cuda"
REAL = GptOssConfig.from_dict({
    "vocab_size": 201088, "hidden_size": 2880, "intermediate_size": 2880, "num_hidden_layers": 24,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64, "num_local_experts": 32,
    "num_experts_per_tok": 4, "sliding_window": 128, "rope_theta": 150000, "swiglu_limit": 7.0,
    "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                     "truncate": False, "original_max_position_embeddings": 4096}})


def p(t):
    return None if t is None else t.data_ptr()


def ok(rc):
    assert rc == 0, LIB.gptoss_last_error().decode()


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


@cuda
@pytest.mark.parametrize("N,K", [(5120, 2880), (2880, 4096), (37, 64), (3142, 2880)])
def test_gemv_bf16(N, K):
    g = gen(N)
    W = torch.randn(N, K, device=DEV, generator=g).to(torch.bfloat16)
    x, b, r = (torch.randn(n, device=DEV, generator=g) for n in (K, N, N))
    want = W.double() @ x.double()
    y = torch.empty(N, device=DEV)
    ok(LIB.gptoss_kernel_gemv_bf16(p(W), p(x), None, None, p(y), N, K))
    torch.testing.assert_close(y.double(), want, rtol=1e-5, atol=2e-4)
    ok(LIB.gptoss_kernel_gemv_bf16(p(W), p(x), p(b), p(r), p(y), N, K))
    torch.testing.assert_close(y.double(), want + b.double() + r.double(), rtol=1e-5, atol=2e-4)
    y2 = r.clone()                                  # resid may alias y
    ok(LIB.gptoss_kernel_gemv_bf16(p(W), p(x), p(b), p(y2), p(y2), N, K))
    torch.testing.assert_close(y2.double(), want + b.double() + r.double(), rtol=1e-5, atol=2e-4)


@cuda
def test_rmsnorm():
    g = gen(1)
    x, w = torch.randn(2880, device=DEV, generator=g) * 3, torch.randn(2880, device=DEV, generator=g)
    y = torch.empty_like(x)
    ok(LIB.gptoss_kernel_rmsnorm(p(x), p(w), p(y), 2880, 1e-5))
    xd = x.double()
    torch.testing.assert_close(y.double(), xd * torch.rsqrt(xd.pow(2).mean() + 1e-5) * w.double(), rtol=1e-5, atol=1e-6)


@cuda
@pytest.mark.parametrize("pos,window", [(0, 0), (77, 0), (131071, 0), (300, 128), (127, 128)])
def test_rope_append(pos, window):
    g = gen(pos + window)
    qkv = torch.randn(5120, device=DEV, generator=g)
    cos_t, sin_t = rope.cos_sin(REAL, torch.arange(pos + 1))
    cos_t, sin_t = cos_t.to(DEV).contiguous(), sin_t.to(DEV).contiguous()
    slots = window if window else pos + 1
    kc = torch.zeros(slots, 8, 64, device=DEV, dtype=torch.float16)
    vc = torch.zeros_like(kc)
    out = qkv.clone()
    ok(LIB.gptoss_kernel_rope_append(p(out), p(cos_t), p(sin_t), pos, p(kc), p(vc), window))
    c, s = cos_t[pos:pos + 1].cpu(), sin_t[pos:pos + 1].cpu()
    q_ref = rope.apply(qkv[:4096].view(1, 64, 64).cpu(), c, s).view(-1)
    k_ref = rope.apply(qkv[4096:4608].view(1, 8, 64).cpu(), c, s).view(8, 64)
    torch.testing.assert_close(out[:4096].cpu(), q_ref, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out[4096:4608].cpu(), k_ref.view(-1), rtol=1e-6, atol=1e-6)
    slot = pos % window if window else pos
    torch.testing.assert_close(kc[slot].cpu(), k_ref.to(torch.float16))
    torch.testing.assert_close(vc[slot].cpu(), qkv[4608:].view(8, 64).cpu().to(torch.float16))


def attention_ref(q, kc, vc, sinks, pos, window):
    """Visible keys of a query at `pos`; sinks as an extra softmax column (float64)."""
    first = max(0, pos - window + 1) if window else 0
    positions = torch.arange(first, pos + 1)
    slots = positions % window if window else positions
    K = kc[slots].double()                       # [n, 8, 64]
    V = vc[slots].double()
    qh = q.double().view(8, 8, 64)               # [kv head, group, dim]
    s = torch.einsum("hgd,nhd->hgn", qh, K) / 8.0
    s = torch.cat([s, sinks.double().view(8, 8, 1)], dim=-1)
    w = torch.softmax(s, dim=-1)[..., :-1]
    return torch.einsum("hgn,nhd->hgd", w, V).reshape(-1)


@cuda
@pytest.mark.parametrize("pos,window,max_ctx", [(0, 0, 4096), (5, 0, 4096), (300, 0, 4096), (8199, 0, 131072),
                                                (20000, 0, 32768), (127, 128, 4096), (128, 128, 4096),
                                                (1000, 128, 4096)])
def test_attention(pos, window, max_ctx):
    g = gen(pos * 7 + window)
    slots = window if window else pos + 1
    q = torch.randn(64 * 64, device=DEV, generator=g)
    kc = torch.randn(slots, 8, 64, device=DEV, generator=g).to(torch.float16)
    vc = torch.randn(slots, 8, 64, device=DEV, generator=g).to(torch.float16)
    sinks = torch.randn(64, device=DEV, generator=g) * 2
    out = torch.empty(64 * 64, device=DEV)
    ok(LIB.gptoss_kernel_attention(p(q), p(kc), p(vc), p(sinks), pos, window, max_ctx, p(out)))
    torch.testing.assert_close(out.double(), attention_ref(q, kc, vc, sinks, pos, window), rtol=1e-4, atol=1e-5)


@cuda
def test_router():
    g = gen(3)
    W = (torch.randn(32, 2880, device=DEV, generator=g) * 0.05).to(torch.bfloat16)
    b, x = torch.randn(32, device=DEV, generator=g) * 0.1, torch.randn(2880, device=DEV, generator=g)
    ids = torch.empty(4, device=DEV, dtype=torch.int32)
    w = torch.empty(4, device=DEV)
    ok(LIB.gptoss_kernel_router(p(W), p(b), p(x), 32, 2880, 4, p(ids), p(w)))
    logits = W.double() @ x.double() + b.double()
    v, i = torch.topk(logits, 4)
    assert ids.tolist() == i.tolist()
    torch.testing.assert_close(w.double(), torch.softmax(v, 0), rtol=1e-5, atol=1e-6)


@cuda
def test_moe_real_shape():
    g = torch.Generator().manual_seed(4)
    E, H, I, k = 6, 2880, 2880, 4
    gu_b = torch.randint(0, 256, (E, 2 * I, H // 32, 16), generator=g, dtype=torch.uint8)
    gu_s = torch.randint(118, 124, (E, 2 * I, H // 32), generator=g, dtype=torch.uint8)
    dn_b = torch.randint(0, 256, (E, H, I // 32, 16), generator=g, dtype=torch.uint8)
    dn_s = torch.randint(118, 124, (E, H, I // 32), generator=g, dtype=torch.uint8)
    gu_bias, dn_bias = torch.randn(E, 2 * I, generator=g) * 0.1, torch.randn(E, H, generator=g) * 0.1
    x = torch.randn(H, generator=g)
    resid = torch.randn(H, generator=g)
    ids = torch.tensor([5, 0, 3, 2], dtype=torch.int32)
    weights = torch.softmax(torch.randn(k, generator=g), 0)
    # Keep every device tensor referenced until the call returns: a temporary's memory is freed (and reused by
    # the caching allocator) as soon as its last Python reference goes away.
    d = [t.to(DEV).contiguous() for t in (gu_b, gu_s, gu_bias, dn_b, dn_s, dn_bias, x, ids, weights)]
    out = resid.to(DEV).clone()
    ok(LIB.gptoss_kernel_moe(*(p(t) for t in d), p(out), k, H, I, 1.702, 7.0))
    want = resid.double().clone()
    for slot, e in enumerate(ids.tolist()):
        w1 = mxfp4.dequant(gu_b[e], gu_s[e]).double()
        w2 = mxfp4.dequant(dn_b[e], dn_s[e]).double()
        t = w1 @ x.double() + gu_bias[e].double()
        glu, lin = t[0::2].clamp(max=7.0), t[1::2].clamp(-7.0, 7.0)
        a = glu * torch.sigmoid(1.702 * glu) * (lin + 1)
        want += weights[slot].double() * (w2 @ a + dn_bias[e].double())
    torch.testing.assert_close(out.cpu().double(), want, rtol=1e-4, atol=1e-3)
