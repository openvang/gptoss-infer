#!/usr/bin/env bash
# llama.cpp baseline for gpt-oss-20b on the local GPU, with the provenance needed to compare runs.
#
#   bench/llama_cpp_baseline.sh <llama.cpp dir with build/bin/llama-bench> <model.gguf> <out.json>
#
# Records: llama.cpp commit, GGUF sha256, GPU, driver, enforced/default power limit, clocks; then runs
#   1. prefill pp512/pp2048/pp8192 and decode tg128 at depth 0      (5 reps)
#   2. decode tg128 at context depth 4k/16k/32k                     (3 reps)
#   3. (2) and the depth-0 decode again with GGML_CUDA_GRAPH_OPT=1  (llama.cpp's opt-in graph optimisation)
# Flags are stated explicitly because llama-bench defaults change between releases.
set -euo pipefail
LLAMA=$1 GGUF=$2 OUT=$3
BENCH="$LLAMA/build/bin/llama-bench"
COMMON=(-m "$GGUF" -ngl 99 -fa 1 -b 2048 -ub 2048 -o json)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

gpu() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits | head -1 | xargs; }

"$BENCH" "${COMMON[@]}" -p 512,2048,8192 -n 128 -r 5 > "$tmp/std.json"
"$BENCH" "${COMMON[@]}" -p 0 -n 128 -d 4096,16384,32768 -r 3 > "$tmp/depth.json"
GGML_CUDA_GRAPH_OPT=1 "$BENCH" "${COMMON[@]}" -p 0 -n 128 -d 0,4096,16384,32768 -r 3 > "$tmp/graphopt.json"

python3 - "$tmp" "$OUT" <<EOF
import json, sys
from pathlib import Path
tmp, out = Path(sys.argv[1]), Path(sys.argv[2])
def rows(name, variant):
    return [{"variant": variant, "test": f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}",
             "depth": r.get("n_depth", 0), "tok_s": r["avg_ts"], "stddev": r["stddev_ts"], "reps": len(r["samples_ts"])}
            for r in json.loads((tmp / name).read_text())]
result = {
    "llama_cpp_commit": "$(git -C "$LLAMA" rev-parse HEAD)",
    "gguf": "$(basename "$GGUF")", "gguf_sha256": "$(sha256sum "$GGUF" | cut -d' ' -f1)",
    "gpu": "$(gpu name)", "driver": "$(gpu driver_version)",
    "power_limit_w": "$(gpu power.limit)", "power_default_limit_w": "$(gpu power.default_limit)",
    "clocks_max_sm_mhz": "$(gpu clocks.max.sm)", "date_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "flags": "-ngl 99 -fa 1 -b 2048 -ub 2048",
    "results": rows("std.json", "default") + rows("depth.json", "default") + rows("graphopt.json", "GGML_CUDA_GRAPH_OPT=1"),
}
out.write_text(json.dumps(result, indent=2) + "\n")
for r in result["results"]:
    print(f"{r['variant']:22s} {r['test']:7s} depth {r['depth']:6d}  {r['tok_s']:9.1f} ± {r['stddev']:.1f} tok/s")
EOF
