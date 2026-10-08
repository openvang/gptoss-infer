"""Gate the engine against a golden file: teacher-force every golden sequence through decode, write a candidate
dump bound to the golden's sha256, and report reference/gptoss_ref/compare.py's metrics.

    PYTHONPATH=reference python runtime/tools/score_golden.py --lib build/libgptoss.so \
        --model-dir /path/to/gpt-oss-20b --golden reference/goldens/golden_v1 --out cand.safetensors [--graph]
"""
import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from gptoss_ref import compare, golden


def load_lib(path):
    lib = ctypes.CDLL(path)
    P, I32 = ctypes.c_void_p, ctypes.c_int
    lib.gptoss_last_error.restype = ctypes.c_char_p
    lib.gptoss_create.restype = P
    lib.gptoss_create.argtypes = [ctypes.c_char_p, I32, ctypes.c_longlong]
    lib.gptoss_destroy.argtypes = [P]
    for name in ("gptoss_reset",):
        getattr(lib, name).argtypes = [P]
    lib.gptoss_use_graph.argtypes = [P, I32]
    lib.gptoss_step.argtypes = [P, I32]
    lib.gptoss_score.argtypes = [P, ctypes.POINTER(ctypes.c_int32), I32, I32, ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float),
                                 ctypes.POINTER(ctypes.c_float)]
    return lib


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--golden", required=True, help="golden path without extension")
    ap.add_argument("--out", required=True, help="candidate file to write")
    ap.add_argument("--graph", action="store_true", help="replay a captured CUDA graph per step")
    args = ap.parse_args()

    lib = load_lib(args.lib)

    def ok(rc):
        if rc != 0:
            sys.exit(f"engine error: {lib.gptoss_last_error().decode()}")

    manifest = json.loads(Path(args.golden + ".json").read_text())
    gfile = args.golden + ".safetensors"
    if golden.file_sha256(gfile) != manifest["sha256"]:
        sys.exit("golden file does not match its manifest")
    records, meta = golden.load(gfile)
    max_ctx = max(len(r["tokens"]) for r in records)
    engine = lib.gptoss_create(args.model_dir.encode(), max_ctx, 0)
    if not engine:
        sys.exit(f"create failed: {lib.gptoss_last_error().decode()}")
    ok(lib.gptoss_use_graph(engine, int(args.graph)))

    k = records[0]["top_ids"].shape[1]
    ids = (ctypes.c_int32 * k)()
    lp = (ctypes.c_float * k)()
    am, tl, lse = ctypes.c_int32(), ctypes.c_float(), ctypes.c_float()
    cands, t0 = [], time.time()
    for rec in records:
        toks = rec["tokens"].tolist()
        n = len(toks)
        lp_at, top1, tlp = torch.empty(n - 1, k), torch.empty(n - 1, dtype=torch.int32), torch.empty(n - 1)
        ok(lib.gptoss_reset(engine))
        for pos in range(n - 1):
            ok(lib.gptoss_step(engine, toks[pos]))
            ids[:] = rec["top_ids"][pos].tolist()
            ok(lib.gptoss_score(engine, ids, k, toks[pos + 1], lp, ctypes.byref(am), ctypes.byref(tl), ctypes.byref(lse)))
            lp_at[pos] = torch.tensor(lp[:])
            top1[pos] = am.value
            tlp[pos] = tl.value
        cands.append({"lp_at_ref": lp_at, "top1": top1, "target_lp": tlp})
    elapsed = time.time() - t0
    lib.gptoss_destroy(engine)

    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        rev = ""
    compare.save_candidate(args.out, cands, golden_sha256=manifest["sha256"],
                           engine=f"gptoss-infer {rev} graph={args.graph}")
    summary, per_seq = compare.compare(records, cands)
    positions = sum(len(r["tokens"]) - 1 for r in records)
    print(json.dumps({"positions": positions, "seconds": round(elapsed, 1), "summary": summary}, indent=2))
    for row, item in zip(per_seq, meta["corpus"]["ids"]):
        print(f"  {item:16s} top1 {row['top1']:.4f} kl_mean {row['kl_mean']:.2e} kl_max {row['kl_max']:.2e} "
              f"gen_top1 {row.get('gen_top1', float('nan')):.4f} gen_kl_mean {row.get('gen_kl_mean', float('nan')):.2e}")


if __name__ == "__main__":
    main()
