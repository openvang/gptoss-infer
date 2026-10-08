"""YaRN rotary embeddings for gpt-oss (NeoX half-split rotation).

Numerics deliberately follow OpenAI's reference (gpt_oss/torch/model.py) and Hugging Face's implementation: the
frequencies, the angles (position * inv_freq) and cos/sin are all computed in float32. Computing angles in float64
would be "more exact" but would move the golden away from the implementations the model is served and verified
with; at 131k positions the two differ by up to ~0.008 rad on the fastest pair.
"""
import math

import torch


def yarn_range(head_dim, theta, original_max_pos, beta_fast, beta_slow, truncate):
    """Correction range [low, high] in rotary-pair index units.

    gpt-oss ships truncate=False: low = 8.0928, high = 17.3980 for its config. Truncating (HF's default, and what
    llama.cpp does for every model) gives [8, 18] and changes the frequencies of pairs 9..17.
    """
    def dim_for(rotations):
        return head_dim * math.log(original_max_pos / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low, high = dim_for(beta_fast), dim_for(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, head_dim - 1)


def yarn_inv_freq(cfg, truncate=None):
    """(inv_freq [head_dim/2] fp32, concentration). `truncate` overrides the config, for comparisons only."""
    truncate = cfg.rope_truncate if truncate is None else truncate
    d = cfg.head_dim
    freq = cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d)
    extrapolation = 1.0 / freq
    interpolation = 1.0 / (cfg.rope_factor * freq)
    low, high = yarn_range(d, cfg.rope_theta, cfg.rope_original_max_pos,
                           cfg.rope_beta_fast, cfg.rope_beta_slow, truncate)
    if high == low:
        high += 0.001
    ramp = ((torch.arange(d // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    keep = 1 - ramp                          # 1 = keep the original (extrapolated) frequency
    inv_freq = interpolation * (1 - keep) + extrapolation * keep
    concentration = 0.1 * math.log(cfg.rope_factor) + 1.0 if cfg.rope_factor > 1 else 1.0
    return inv_freq, concentration


def cos_sin(cfg, positions):
    """cos, sin [n, head_dim/2] fp32 for integer positions, both scaled by the YaRN concentration."""
    inv_freq, concentration = yarn_inv_freq(cfg)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    return angles.cos() * concentration, angles.sin() * concentration


def apply(x, cos, sin):
    """Rotate x [n, heads, head_dim] by NeoX half-split: (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin)."""
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
