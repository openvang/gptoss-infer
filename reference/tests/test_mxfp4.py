import pytest
import torch

from gptoss_ref import mxfp4


def naive_dequant(blocks, scales):
    """Element-by-element decoding straight from the spec, for comparison."""
    out = []
    for row_b, row_s in zip(blocks.reshape(-1, 16).tolist(), scales.reshape(-1).tolist()):
        for byte in row_b:
            for code in (byte & 0x0F, byte >> 4):          # low nibble is the even element
                out.append(mxfp4.FP4_VALUES[code] * 2.0 ** (row_s - 127))
    return torch.tensor(out, dtype=torch.float32).reshape(*scales.shape[:-1], scales.shape[-1] * 32)


def test_matches_naive_decoding():
    g = torch.Generator().manual_seed(1)
    blocks = torch.randint(0, 256, (3, 5, 4, 16), generator=g, dtype=torch.uint8)
    scales = torch.randint(100, 150, (3, 5, 4), generator=g, dtype=torch.uint8)
    assert torch.equal(mxfp4.dequant(blocks, scales), naive_dequant(blocks, scales))


def test_every_code_and_nibble_order():
    # byte 0x21: low nibble 1 (+0.5) is element 0, high nibble 2 (+1.0) is element 1
    blocks = torch.zeros(1, 16, dtype=torch.uint8)
    blocks[0, 0] = 0x21
    out = mxfp4.dequant(blocks, torch.tensor([127], dtype=torch.uint8))
    assert out[0].item() == 0.5 and out[1].item() == 1.0
    codes = torch.arange(16, dtype=torch.uint8)
    blocks = (codes[0::2] | (codes[1::2] << 4)).reshape(1, 8).repeat(1, 2)
    out = mxfp4.dequant(blocks, torch.tensor([128], dtype=torch.uint8))     # scale 2**1
    assert out[:16].tolist() == [2 * v for v in mxfp4.FP4_VALUES]


def test_scale_extremes_are_exact():
    blocks = torch.full((1, 16), 0x77, dtype=torch.uint8)                    # every element is +6.0
    lo = mxfp4.dequant(blocks, torch.tensor([1], dtype=torch.uint8))
    assert lo[0].item() == 6.0 * 2.0 ** -126
    hi = mxfp4.dequant(blocks, torch.tensor([254], dtype=torch.uint8))
    assert hi[0].item() == 6.0 * 2.0 ** 127 or hi[0].item() == float("inf")  # 6 * 2**127 overflows fp32


def test_every_valid_scale_is_an_exact_power_of_two():
    blocks = torch.full((255, 1, 16), 0x22, dtype=torch.uint8)               # 255 rows of one block, all +1.0
    scales = torch.arange(255, dtype=torch.uint8).reshape(255, 1)
    out = mxfp4.dequant(blocks, scales)                                      # [255, 32]
    want = torch.tensor([2.0 ** (s - 127) for s in range(255)], dtype=torch.float64).to(torch.float32)
    assert torch.equal(out[:, 0], want) and torch.equal(out[:, 31], want)


def test_nan_scale_rejected():
    with pytest.raises(ValueError):
        mxfp4.dequant(torch.zeros(1, 16, dtype=torch.uint8), torch.tensor([255], dtype=torch.uint8))


def test_quantize_roundtrip_of_representable_values():
    g = torch.Generator().manual_seed(2)
    blocks = torch.randint(0, 256, (4, 3, 16), generator=g, dtype=torch.uint8)
    scales = torch.randint(110, 130, (4, 3), generator=g, dtype=torch.uint8)
    w = mxfp4.dequant(blocks, scales)
    assert torch.equal(mxfp4.dequant(*mxfp4.quantize(w)), w)
