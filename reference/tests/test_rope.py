import dataclasses
import math

import pytest
import torch

from gptoss_ref import rope
from gptoss_ref.config import GptOssConfig

REAL = {  # the rope-relevant part of openai/gpt-oss-20b config.json
    "vocab_size": 201088, "hidden_size": 2880, "intermediate_size": 2880, "num_hidden_layers": 24,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64, "num_local_experts": 32,
    "num_experts_per_tok": 4, "sliding_window": 128, "rope_theta": 150000,
    "rope_scaling": {"rope_type": "yarn", "factor": 32.0, "beta_fast": 32.0, "beta_slow": 1.0,
                     "truncate": False, "original_max_position_embeddings": 4096},
}


@pytest.fixture
def real_cfg():
    return GptOssConfig.from_dict(REAL)


def test_correction_range_is_untruncated(real_cfg):
    low, high = rope.yarn_range(64, 150000, 4096, 32.0, 1.0, truncate=False)
    assert low == pytest.approx(8.0928, abs=1e-4) and high == pytest.approx(17.3980, abs=1e-4)
    assert rope.yarn_range(64, 150000, 4096, 32.0, 1.0, truncate=True) == (8, 18)
    assert real_cfg.rope_truncate is False


def test_truncation_changes_exactly_pairs_9_to_17(real_cfg):
    a, _ = rope.yarn_inv_freq(real_cfg, truncate=False)
    b, _ = rope.yarn_inv_freq(real_cfg, truncate=True)
    changed = [i for i in range(32) if a[i] != b[i]]
    assert changed == list(range(9, 18))


def test_concentration(real_cfg):
    _, c = rope.yarn_inv_freq(real_cfg)
    assert c == pytest.approx(0.1 * math.log(32) + 1)


def test_matches_transformers_yarn(real_cfg):
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    hf_cfg = transformers.GptOssConfig(
        hidden_size=2880, num_attention_heads=64, num_key_value_heads=8, head_dim=64, num_hidden_layers=2,
        max_position_embeddings=131072,
        rope_parameters={"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0, "beta_fast": 32.0,
                         "beta_slow": 1.0, "truncate": False, "original_max_position_embeddings": 4096})
    hf_inv, hf_scale = ROPE_INIT_FUNCTIONS["yarn"](hf_cfg, "cpu")
    ours, conc = rope.yarn_inv_freq(real_cfg)
    assert torch.equal(ours, hf_inv.to(torch.float32))
    assert conc == pytest.approx(hf_scale)


def test_rotation_is_neox_half_split():
    cfg = GptOssConfig.from_dict(REAL)
    cfg = dataclasses.replace(cfg, rope_factor=1.0)       # plain RoPE, concentration 1
    x = torch.zeros(1, 1, 64)
    x[0, 0, 0] = 1.0                                         # pair 0 is (dim 0, dim 32)
    cos, sin = rope.cos_sin(cfg, torch.tensor([1]))
    y = rope.apply(x, cos, sin)
    assert y[0, 0, 0].item() == pytest.approx(math.cos(1.0)) and y[0, 0, 32].item() == pytest.approx(math.sin(1.0))
