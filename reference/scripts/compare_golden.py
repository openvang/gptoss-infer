"""Gate an engine's candidate dump against a golden file.

    python reference/scripts/compare_golden.py --golden reference/goldens/golden_v1.safetensors \
        --candidate engine_dump.safetensors [--top1-min 0.99 --kl-mean-max 0.01 --kl-p99-max 0.1] [--json out.json]

Without thresholds it only reports. Thresholds must come from calibration (an exact engine against this golden,
repeated), not from guesses; see reference/README.md. Exit status: 0 pass or report-only, 1 fail, 2 bad input.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptoss_ref import compare, golden  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--top1-min", type=float)
    ap.add_argument("--kl-mean-max", type=float)
    ap.add_argument("--kl-p99-max", type=float)
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    g_records, _ = golden.load(args.golden)
    c_records, c_meta = compare.load_candidate(args.candidate)
    g_sha = golden.file_sha256(args.golden)
    if c_meta.get("golden_sha256") != g_sha:
        print(f"candidate was computed against golden {c_meta.get('golden_sha256', '?')[:16]}…, "
              f"not {g_sha[:16]}…", file=sys.stderr)
        sys.exit(2)
    if len(c_records) != len(g_records):
        print(f"candidate has {len(c_records)} sequences, golden has {len(g_records)}", file=sys.stderr)
        sys.exit(2)
    try:
        summary, per_seq = compare.compare(g_records, c_records)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    print(json.dumps(summary, indent=2))
    if args.json:
        Path(args.json).write_text(compare.report(summary, per_seq))
    thresholds = (args.top1_min, args.kl_mean_max, args.kl_p99_max)
    if all(t is None for t in thresholds):
        print("report only: no thresholds given")
        return
    if any(t is None for t in thresholds):
        print("give all three thresholds or none", file=sys.stderr)
        sys.exit(2)
    ok, reasons = compare.verdict(summary, *thresholds)
    print("PASS" if ok else "FAIL: " + "; ".join(reasons))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
