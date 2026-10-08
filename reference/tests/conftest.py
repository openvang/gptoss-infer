import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptoss_ref import mxfp4  # noqa: E402
from gptoss_ref.config import GptOssConfig  # noqa: E402

# A tiny gpt-oss: same structure as the real model (alternating window/full layers, sinks, biases, YaRN with
# truncate=false, interleaved clamped SwiGLU, top-4 of 8 experts in MXFP4), small enough for CPU tests.
TINY = {
    "vocab_size": 97, "hidden_size": 64, "intermediate_size": 32, "num_hidden_layers": 4,
    "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 16,
    "num_local_experts": 8, "num_experts_per_tok": 4, "sliding_window": 6,
    "layer_types": ["sliding_attention", "full_attention"] * 2,
    "rms_norm_eps": 1e-5, "rope_theta": 150000.0, "swiglu_limit": 7.0, "max_position_embeddings": 4096,
    "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                     "truncate": False, "original_max_position_embeddings": 4096},
}


def tiny_tensors(cfg, seed=0):
    """Random weights in Hugging Face naming, experts stored as MXFP4 blocks + scales like the real checkpoint."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, std=1.0: torch.randn(*s, generator=g) * std
    H, I, E = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts
    hq, hkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    t = {"model.embed_tokens.weight": r(cfg.vocab_size, H, std=0.5),
         "model.norm.weight": 1 + r(H, std=0.1),
         "lm_head.weight": r(cfg.vocab_size, H, std=0.1)}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        t |= {p + "input_layernorm.weight": 1 + r(H, std=0.1),
              p + "post_attention_layernorm.weight": 1 + r(H, std=0.1),
              p + "self_attn.q_proj.weight": r(hq * d, H, std=0.08), p + "self_attn.q_proj.bias": r(hq * d, std=0.05),
              p + "self_attn.k_proj.weight": r(hkv * d, H, std=0.08), p + "self_attn.k_proj.bias": r(hkv * d, std=0.05),
              p + "self_attn.v_proj.weight": r(hkv * d, H, std=0.08), p + "self_attn.v_proj.bias": r(hkv * d, std=0.05),
              p + "self_attn.o_proj.weight": r(H, hq * d, std=0.08), p + "self_attn.o_proj.bias": r(H, std=0.05),
              p + "self_attn.sinks": r(hq, std=1.0),
              p + "mlp.router.weight": r(E, H, std=0.2), p + "mlp.router.bias": r(E, std=0.1),
              p + "mlp.experts.gate_up_proj_bias": r(E, 2 * I, std=0.05),
              p + "mlp.experts.down_proj_bias": r(E, H, std=0.05)}
        for name, rows, k in (("gate_up_proj", 2 * I, H), ("down_proj", H, I)):
            t[p + f"mlp.experts.{name}_blocks"] = torch.randint(0, 256, (E, rows, k // 32, 16), generator=g,
                                                               dtype=torch.uint8)
            t[p + f"mlp.experts.{name}_scales"] = (127 + torch.randint(-7, -3, (E, rows, k // 32), generator=g)
                                                  ).to(torch.uint8)
    return t


@pytest.fixture
def tiny_cfg():
    return GptOssConfig.from_dict(TINY)


@pytest.fixture
def tiny_weights(tiny_cfg):
    return tiny_tensors(tiny_cfg)


def dense_experts(tensors, cfg):
    """MXFP4 experts -> Hugging Face dense layout ([E, in, out]) for loading into transformers."""
    out = {}
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}.mlp.experts."
        for name in ("gate_up_proj", "down_proj"):
            w = mxfp4.dequant(tensors[p + name + "_blocks"], tensors[p + name + "_scales"])  # [E, out, in]
            out[p + name] = w.transpose(1, 2).contiguous()
    return out
