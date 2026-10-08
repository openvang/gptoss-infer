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


PROOF = """## RTX 5090 results

- [x] Tested on RTX 5090

| | decode@128 | decode@4k | prefill@4k |
|---|---:|---:|---:|
| before (main) | 283.0 | 269.7 | 280.1 |
| after (this PR) | 283.1 | 1,269.5 tok/s | |
"""


def test_proof_needs_the_ticked_box_and_a_gain():
    assert policy.proof(PROOF) == "ok"                                   # decode@128 283.1 > 283.0
    assert policy.proof(PROOF.replace("[x]", "[X]")) == "ok"
    assert policy.proof(PROOF.replace("[x]", "[ ]")) == "unticked"
    assert policy.proof("") == "unticked" and policy.proof(None) == "unticked"
    assert policy.proof(f"<!-- - [x] Tested on RTX 5090 -->\n{PROOF.replace('[x]', '[ ]')}") == "unticked"
    flat = PROOF.replace("283.1", "283.0").replace("1,269.5 tok/s", "269.7")
    assert policy.proof(flat) == "no-gain"
    assert policy.proof(flat.replace("| after (this PR) | 283.0 | 269.7 | |", "| after (this PR) | | | 290 |")) == "ok"
    assert policy.proof(PROOF.replace("before (main)", "baseline")) == "no-gain"     # row labels are what is read


def test_lanes():
    assert policy.lane(["runtime/src/kernels.cu", "CMakeLists.txt"]) == "scored"
    assert policy.lane(["runtime/src/kernels.cu", "README.md"]) == "mixed"
    assert policy.lane(["README.md", "bench/llama_cpp_baseline.sh"]) == "manual"
    assert policy.lane([]) == "manual"
    assert policy.lane(["runtime/src/kernels.cu", "eval/policy.py"]) == "protected"
    assert policy.lane(["bench/baselines/x.json"]) == "protected"
    assert policy.lane(["runtime/a.cu"] * policy.MAX_LISTED_FILES) == "manual"


def test_merge_order_is_largest_conservative_gain_then_oldest():
    assert policy.merge_order([(3, 1.04), (5, 1.07), (2, 1.07)]) == [2, 5, 3]
    axes = {"decode@128": {"low": 1.021}, "prefill@4k": {"low": 1.064}}
    assert policy.speedup_score(axes) == 1.064


def test_the_pr_template_is_what_the_proof_check_reads():
    template = (Path(__file__).resolve().parents[2] / ".github/PULL_REQUEST_TEMPLATE.md").read_text()
    assert policy.proof(template) == "unticked"
    filled = (template.replace("- [ ] Tested on RTX 5090", "- [x] Tested on RTX 5090")
              .replace("| before (main) | | | |", "| before (main) | 283.0 | 269.7 | 280.1 |")
              .replace("| after (this PR) | | | |", "| after (this PR) | 283.0 | 269.7 | 290.4 |"))
    assert policy.proof(filled) == "ok"
    assert policy.proof(filled.replace("290.4", "280.1")) == "no-gain"
