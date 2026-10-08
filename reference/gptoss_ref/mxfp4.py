"""MXFP4 (OCP microscaling FP4) decoding, exactly as gpt-oss stores its expert weights.

A block is 32 E2M1 values in 16 bytes plus one E8M0 scale byte. Byte j of a block holds element 2j in its LOW
nibble and element 2j+1 in its HIGH nibble. The value is e2m1 * 2**(scale - 127). Every MXFP4 value is exactly
representable in fp32 (and in bf16), so decoding is lossless.
"""
import torch
import torch.nn.functional as F

# E2M1 code -> value. Codes 8..15 are the negatives of 0..7 (code 8 is -0.0).
FP4_VALUES = (+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
BLOCK = 32  # values per block
BYTES_PER_BLOCK = 16


def dequant(blocks, scales, dtype=torch.float32):
    """Decode MXFP4 tensors.

    blocks: uint8 [..., G, 16]; scales: uint8 [..., G]  ->  dtype [..., G * 32]
    For gpt-oss experts the leading dims are [rows] (or [experts, rows]) and G * 32 is the input dimension.
    """
    if blocks.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("blocks and scales must be uint8")
    if blocks.shape[-1] != BYTES_PER_BLOCK or blocks.shape[:-1] != scales.shape:
        raise ValueError(f"shape mismatch: blocks {tuple(blocks.shape)} scales {tuple(scales.shape)}")
    if bool((scales == 255).any()):
        raise ValueError("E8M0 scale 0xFF encodes NaN in the OCP MX spec; refusing to decode it")
    # One lookup per byte -> (low-nibble value, high-nibble value); ~3x faster than two nibble gathers on CPU.
    vals = F.embedding(blocks.long(), _byte_lut(dtype)).flatten(-2)    # [..., G, 32], low nibble first
    scale = torch.exp2(scales.to(dtype) - 127)                         # exact powers of two
    return (vals * scale.unsqueeze(-1)).flatten(-2)


def _byte_lut(dtype):
    return torch.tensor([[FP4_VALUES[b & 0x0F], FP4_VALUES[b >> 4]] for b in range(256)], dtype=dtype)


def quantize(weights):
    """Round fp32 weights [..., K] (K % 32 == 0) to MXFP4. Returns (blocks, scales).

    Test helper only: picks the smallest power-of-two scale that keeps the block's max |w| within 6.0, then rounds
    each value to the nearest E2M1 code. dequant(quantize(w)) is the MXFP4 grid point closest to w.
    """
    *lead, k = weights.shape
    if k % BLOCK:
        raise ValueError("last dim must be a multiple of 32")
    w = weights.reshape(*lead, k // BLOCK, BLOCK).to(torch.float32)
    amax = w.abs().amax(dim=-1).clamp_min(2.0 ** -120)
    exp = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scales = (exp + 127).to(torch.uint8)
    scaled = w / torch.exp2(exp).unsqueeze(-1)
    grid = torch.tensor(FP4_VALUES[:8])
    mag = scaled.abs().unsqueeze(-1)
    code = (mag - grid).abs().argmin(dim=-1)                     # 0..7
    code = code + 8 * (scaled < 0).long()                       # sign bit
    code = code.to(torch.uint8)
    blocks = code[..., 0::2] | (code[..., 1::2] << 4)
    return blocks.contiguous(), scales.contiguous()
