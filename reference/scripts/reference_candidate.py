"""Run the reference itself as a candidate engine against a golden file, optionally with an engine-like
KV-cache format, and report compare.py's metrics.

    python reference/scripts/reference_candidate.py --model-dir /path/to/gpt-oss-20b \
        --golden reference/goldens/golden_v1 --device cuda --kv-dtype bf16 [--out cand.safetensors]

This is the calibration instrument for the gate's thresholds:
  * --kv-dtype fp32 on another device measures pure floating-point noise (the floor);
  * --kv-dtype bf16 measures what a bf16 KV cache alone costs, which is the target an exact bf16-KV engine
    should reproduce (an engine scoring much worse than this has a bug, not a precision problem).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptoss_ref import GptOssConfig, Reference, SafetensorsWeights, compare, golden  # noqa: E402

KV = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--golden", required=True, help="golden path without extension")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--kv-dtype", choices=sorted(KV), default="fp32")
    ap.add_argument("--out", help="also write the candidate dump here")
    args = ap.parse_args()

    manifest = json.loads(Path(args.golden + ".json").read_text())
    gfile = args.golden + ".safetensors"
    if golden.file_sha256(gfile) != manifest["sha256"]:
        sys.exit("golden file does not match its manifest")
    records, meta = golden.load(gfile)
    ref = Reference(GptOssConfig.from_model_dir(args.model_dir), SafetensorsWeights(args.model_dir),
                    device=args.device, kv_dtype=KV[args.kv_dtype])
    t0 = time.time()
    hidden = ref.final_hidden([r["tokens"].long() for r in records])
    cands = compare.candidate_from_hidden(records, hidden, ref.w.get("lm_head.weight").to(ref.device))
    elapsed = time.time() - t0
    if args.out:
        compare.save_candidate(args.out, cands, golden_sha256=manifest["sha256"],
                               engine=f"reference device={args.device} kv={args.kv_dtype}")
    summary, per_seq = compare.compare(records, cands)
    print(json.dumps({"device": args.device, "kv_dtype": args.kv_dtype, "seconds": round(elapsed, 1),
                      "summary": summary}, indent=2))
    for row, item in zip(per_seq, meta["corpus"]["ids"]):
        print(f"  {item:16s} top1 {row['top1']:.4f} kl_mean {row['kl_mean']:.2e} kl_max {row['kl_max']:.2e} "
              f"gen_top1 {row.get('gen_top1', float('nan')):.4f} gen_kl_mean {row.get('gen_kl_mean', float('nan')):.2e}")


if __name__ == "__main__":
    main()
