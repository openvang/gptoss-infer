"""Scoring policy: every threshold the evaluator applies, in one place.

Changing this file changes what pull requests are paid for, so it is maintainer-owned (a PR that touches it is
not evaluated) and versioned (POLICY_VERSION goes into every verdict).
"""
import math

POLICY_VERSION = 1

# The LLM has to leave room on the RTX 5090 (32 GiB) for the planned STT and TTS models on the same card, so the
# whole engine process (weights, KV cache, workspaces, CUDA context) must stay under this measured peak.
VRAM_BUDGET_MIB = 24 * 1024

# Correctness against reference/goldens/golden_v1, on the positions the model generates (gen_*) and on all
# positions, for both the step-by-step decode path and gptoss_prefill. Calibrated 2026-10-08 with
# reference/scripts/reference_candidate.py: the fp32 reference with an FP16 KV cache scores gen top-1 0.996,
# gen KL mean 5.2e-4, gen KL p99 0.0057 and all-position KL mean 0.0095. The bounds sit near twice that, so
# FP16-KV-level precision passes and BF16-KV-level degradation (gen KL 1.5e-3, p99 0.025) fails.
GOLDEN_GATE = {"gen_top1_min": 0.993, "gen_kl_mean_max": 1.0e-3, "gen_kl_p99_max": 0.012, "kl_mean_max": 0.019}
# Inside those bounds, a candidate may still not be meaningfully less accurate than the base it would replace.
DIFFERENTIAL_KL_FACTOR = 1.25
DIFFERENTIAL_KL_SLACK = 2e-5

# Speed axes, higher is better, each measured in interleaved base/candidate pairs on real tokens.
AXES = ("decode@128", "decode@4k", "prefill@4k")
# Tiers are earned on the conservative gain: the low end of the paired ratio's confidence interval.
BANDS = ((0.18, "XL"), (0.10, "L"), (0.06, "M"), (0.035, "S"), (0.02, "XS"))
TIERS = tuple(name for _, name in BANDS)
# An axis whose interval lies entirely below 1 - REGRESSION is a measured regression: REJECT.
REGRESSION = 0.02
# 99 % per axis; with three axes the family-wise false-positive rate stays under 3 %.
CONFIDENCE = 0.99
# The candidate's log-probs during the timed runs must match the base's: timing a different computation is not
# a speedup.
INTEGRITY_MAX_MEAN_ABS_LP = 0.05

LABELS = {name: f"eval:{name}" for name in TIERS + ("none", "REJECT", "skipped")}

# Two-sided 99 % Student-t critical values by degrees of freedom.
_T99 = {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499, 8: 3.355, 9: 3.250, 10: 3.169,
        11: 3.106, 12: 3.055, 13: 3.012, 14: 2.977, 15: 2.947, 16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861,
        20: 2.845, 25: 2.787, 30: 2.750}


def _t99(df):
    if df in _T99:
        return _T99[df]
    if df > 30:
        return 2.576
    return _T99[max(k for k in _T99 if k <= df)]          # conservative: the next smaller df has a wider t


def paired_interval(base, cand):
    """Ratio cand/base from paired runs: (estimate, low, high) of exp(mean log ratio) with a 99 % t-interval."""
    if len(base) != len(cand) or len(base) < 2:
        raise ValueError("need at least two paired measurements")
    logs = [math.log(c / b) for b, c in zip(base, cand)]
    n = len(logs)
    mean = sum(logs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in logs) / (n - 1))
    half = _t99(n - 1) * sd / math.sqrt(n)
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def tier_for(gain):
    for edge, name in BANDS:
        if gain >= edge:
            return name
    return None


def correctness_reasons(cand, base):
    """cand/base: {"decode": summary, "prefill": summary} from gptoss_ref.compare. Returns failure reasons."""
    g = GOLDEN_GATE
    reasons = []
    for path in ("decode", "prefill"):
        s = cand[path]
        if s["gen_top1"] < g["gen_top1_min"]:
            reasons.append(f"{path}: gen top-1 {s['gen_top1']:.4f} < {g['gen_top1_min']}")
        if s["gen_kl_mean"] > g["gen_kl_mean_max"]:
            reasons.append(f"{path}: gen KL mean {s['gen_kl_mean']:.2e} > {g['gen_kl_mean_max']:.0e}")
        if s["gen_kl_p99"] > g["gen_kl_p99_max"]:
            reasons.append(f"{path}: gen KL p99 {s['gen_kl_p99']:.2e} > {g['gen_kl_p99_max']}")
        if s["kl_mean"] > g["kl_mean_max"]:
            reasons.append(f"{path}: KL mean {s['kl_mean']:.2e} > {g['kl_mean_max']}")
        if not all(math.isfinite(s[key]) for key in ("kl_mean", "gen_kl_mean")):
            reasons.append(f"{path}: non-finite KL")
        limit = base[path]["gen_kl_mean"] * DIFFERENTIAL_KL_FACTOR + DIFFERENTIAL_KL_SLACK
        if s["gen_kl_mean"] > limit:
            reasons.append(f"{path}: gen KL mean {s['gen_kl_mean']:.2e} is worse than base "
                           f"{base[path]['gen_kl_mean']:.2e} (limit {limit:.2e})")
    return reasons


def decide(build_ok, correct_reasons, peak_mib, axes, integrity_lp):
    """Verdict from measurements.

    axes: {axis: (base_values, cand_values)}; integrity_lp: mean |lp_cand - lp_base| over timed tokens.
    Returns {"label", "tier", "reasons", "axes"}; label is one of LABELS' keys.
    """
    if not build_ok:
        return {"label": "REJECT", "tier": None, "reasons": ["build failed"], "axes": {}}
    reasons = list(correct_reasons)
    if peak_mib > VRAM_BUDGET_MIB:
        reasons.append(f"VRAM peak {peak_mib} MiB > budget {VRAM_BUDGET_MIB} MiB")
    if integrity_lp > INTEGRITY_MAX_MEAN_ABS_LP:
        reasons.append(f"timed runs compute different log-probs than base (mean |dlp| {integrity_lp:.3f})")
    stats, best = {}, None
    for axis, (b, c) in axes.items():
        est, lo, hi = paired_interval(b, c)
        stats[axis] = {"ratio": est, "low": lo, "high": hi, "base_median": sorted(b)[len(b) // 2],
                       "cand_median": sorted(c)[len(c) // 2], "pairs": len(b)}
        if hi < 1 - REGRESSION:
            reasons.append(f"{axis} regressed: ratio {est:.3f} (99 % CI {lo:.3f}-{hi:.3f})")
        tier = tier_for(lo - 1)
        stats[axis]["tier"] = tier
        if tier and (best is None or TIERS.index(tier) < TIERS.index(best)):
            best = tier
    if reasons:
        return {"label": "REJECT", "tier": None, "reasons": reasons, "axes": stats}
    return {"label": best or "none", "tier": best, "reasons": [], "axes": stats}
