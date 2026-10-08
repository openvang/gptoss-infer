import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy  # noqa: E402

GOOD = {"gen_top1": 0.997, "gen_kl_mean": 3.7e-4, "gen_kl_p99": 0.003, "kl_mean": 0.0086}


def test_interval_brackets_the_true_ratio():
    rnd = random.Random(0)
    base = [100 * (1 + rnd.gauss(0, 0.005)) for _ in range(5)]
    cand = [b * 1.10 * (1 + rnd.gauss(0, 0.005)) for b in base]
    est, lo, hi = policy.paired_interval(base, cand)
    assert lo < 1.10 < hi and lo > 1.05 and abs(est - 1.10) < 0.02


def test_no_false_tier_under_noise():
    """A/A: identical code measured with 1 % run-to-run noise must almost never earn a tier."""
    rnd = random.Random(1)
    tiers = 0
    for _ in range(2000):
        base = [100 * (1 + rnd.gauss(0, 0.01)) for _ in range(5)]
        cand = [100 * (1 + rnd.gauss(0, 0.01)) for _ in range(5)]
        _, lo, _ = policy.paired_interval(base, cand)
        tiers += policy.tier_for(lo - 1) is not None
    assert tiers / 2000 < 0.005


def test_bands_and_regression():
    assert policy.tier_for(0.25) == "XL" and policy.tier_for(0.12) == "L" and policy.tier_for(0.07) == "M"
    assert policy.tier_for(0.04) == "S" and policy.tier_for(0.021) == "XS" and policy.tier_for(0.019) is None
    flat = [100.0, 100.2, 99.9, 100.1, 100.0]
    v = policy.decide(True, [], 15000, {"decode@128": (flat, [x * 0.90 for x in flat])}, 0.0)
    assert v["label"] == "REJECT" and "regressed" in v["reasons"][0]
    v = policy.decide(True, [], 15000, {"decode@128": (flat, [x * 1.30 for x in flat])}, 0.0)
    assert v["label"] == "XL"


def test_best_axis_wins_and_gates_reject():
    flat = [100.0, 100.2, 99.9, 100.1, 100.0]
    axes = {"decode@128": (flat, list(flat)), "prefill@4k": (flat, [x * 1.08 for x in flat])}
    assert policy.decide(True, [], 15000, axes, 0.0)["label"] == "M"
    assert policy.decide(True, [], policy.VRAM_BUDGET_MIB + 1, axes, 0.0)["label"] == "REJECT"
    assert policy.decide(True, ["decode: bad"], 15000, axes, 0.0)["label"] == "REJECT"
    assert policy.decide(True, [], 15000, axes, 0.5)["label"] == "REJECT"
    assert policy.decide(False, [], 0, {}, 0.0)["label"] == "REJECT"


def test_correctness_thresholds():
    ok = {"decode": GOOD, "prefill": GOOD}
    assert policy.correctness_reasons(ok, ok) == []
    bf16_kv = dict(GOOD, gen_top1=0.988, gen_kl_mean=1.5e-3, gen_kl_p99=0.025, kl_mean=0.023)
    assert len(policy.correctness_reasons({"decode": bf16_kv, "prefill": GOOD}, ok)) >= 3
    worse = dict(GOOD, gen_kl_mean=GOOD["gen_kl_mean"] * 1.6)          # within absolute bounds, worse than base
    assert any("worse than base" in r for r in policy.correctness_reasons({"decode": worse, "prefill": GOOD}, ok))
    nan = dict(GOOD, kl_mean=math.nan)
    assert policy.correctness_reasons({"decode": nan, "prefill": GOOD}, ok)


def test_needs_pairs():
    with pytest.raises(ValueError):
        policy.paired_interval([1.0], [1.0])
