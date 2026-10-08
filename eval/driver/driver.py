#!/usr/bin/env python3
"""Runs inside the isolated eval container (no network, read-only weights and data, no credentials).

Drives one build of libgptoss through its public C API and writes raw results to --out. It judges nothing:
run_eval.py compares and scores outside the container. Dependencies: python3 and numpy only.

  score: teacher-force every golden sequence through gptoss_step (decode path) and through gptoss_prefill
         (prefill path), recording log-probs at the golden's top-k ids, argmax and target log-probs.
  bench: for each scenario, time gptoss_prefill on a real prompt, then time teacher-forced decode steps,
         recording the target log-probs of the timed steps (an integrity fingerprint).
"""
import argparse
import ctypes
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path

import numpy as np

I32P = ctypes.POINTER(ctypes.c_int32)
F32P = ctypes.POINTER(ctypes.c_float)


def load(path):
    lib = ctypes.CDLL(path)
    P, I32 = ctypes.c_void_p, ctypes.c_int
    sig = {
        "gptoss_create": ([ctypes.c_char_p, I32, ctypes.c_longlong], P),
        "gptoss_destroy": ([P], None),
        "gptoss_last_error": ([], ctypes.c_char_p),
        "gptoss_device_bytes": ([P], ctypes.c_longlong),
        "gptoss_reset": ([P], I32),
        "gptoss_step": ([P, I32], I32),
        "gptoss_prefill": ([P, I32P, I32, I32P, I32, F32P, I32P, F32P], I32),
        "gptoss_score": ([P, I32P, I32, I32, F32P, I32P, F32P, F32P], I32),
    }
    for name, (args, res) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, res
    return lib


def ptr(a, kind):
    return a.ctypes.data_as(kind) if a is not None else None


class Engine:
    def __init__(self, lib, model, max_ctx, budget_bytes):
        self.lib = lib
        self.h = lib.gptoss_create(model.encode(), max_ctx, budget_bytes)
        if not self.h:
            raise RuntimeError(f"create: {lib.gptoss_last_error().decode()}")

    def ok(self, rc, what):
        if rc != 0:
            raise RuntimeError(f"{what}: {self.lib.gptoss_last_error().decode()}")

    def reset(self):
        self.ok(self.lib.gptoss_reset(self.h), "reset")

    def step(self, tok):
        self.ok(self.lib.gptoss_step(self.h, int(tok)), "step")

    def prefill(self, toks, ids=None, k=0, lp=None, am=None, tl=None):
        self.ok(self.lib.gptoss_prefill(self.h, ptr(toks, I32P), len(toks), ptr(ids, I32P), k, ptr(lp, F32P),
                                        ptr(am, I32P), ptr(tl, F32P)), "prefill")

    def score(self, ids, k, target, lp, am, tl):
        lse = np.zeros(1, np.float32)
        self.ok(self.lib.gptoss_score(self.h, ptr(ids, I32P), k, int(target), ptr(lp, F32P), ptr(am, I32P),
                                      ptr(tl, F32P), ptr(lse, F32P)), "score")


class MemSampler(threading.Thread):
    """Peak device memory in use, read from the driver (covers allocations the engine does not report)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.done = 0, threading.Event()

    @staticmethod
    def used_mib():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])

    def run(self):
        while not self.done.is_set():
            self.peak = max(self.peak, self.used_mib())
            self.done.wait(0.1)


def task_score(eng, data, out):
    meta = json.loads((data / "golden.json").read_text())
    for j in range(meta["n_seqs"]):
        toks = np.load(data / f"tokens_{j}.npy").astype(np.int32)
        ids = np.ascontiguousarray(np.load(data / f"ids_{j}.npy").astype(np.int32))
        n, k = len(toks), ids.shape[1]
        for path in ("dec", "pf"):
            lp = np.zeros((n - 1, k), np.float32)
            am = np.zeros(n - 1, np.int32)
            tl = np.zeros(n - 1, np.float32)
            eng.reset()
            if path == "dec":
                for p in range(n - 1):
                    eng.step(toks[p])
                    eng.score(ids[p], k, toks[p + 1], lp[p], am[p:p + 1], tl[p:p + 1])
            else:
                eng.prefill(toks, ids, k, lp, am, tl)
            np.save(out / f"{path}_lp_{j}.npy", lp)
            np.save(out / f"{path}_top1_{j}.npy", am)
            np.save(out / f"{path}_target_{j}.npy", tl)
    return {}


def task_bench(eng, data, out):
    toks_all = np.load(data / "bench_tokens.npy").astype(np.int32)
    results = {}
    for sc in json.loads((data / "bench.json").read_text()):
        P, D = sc["prompt"], sc["decode"]
        toks = toks_all[: P + D + 1]
        eng.reset()
        t0 = time.perf_counter()
        eng.prefill(np.ascontiguousarray(toks[:P]))
        t1 = time.perf_counter()
        tl = np.zeros(D, np.float32)
        am = np.zeros(1, np.int32)
        dummy = np.zeros(1, np.float32)
        for i in range(D):
            eng.step(toks[P + i])
            eng.score(None, 0, toks[P + i + 1], dummy, am, tl[i:i + 1])
        t2 = time.perf_counter()
        results[sc["name"]] = {"prompt": P, "decode": D, "prefill_tok_s": P / (t1 - t0),
                               "decode_tok_s": D / (t2 - t1), "target_lp": tl.tolist()}
    return {"scenarios": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", choices=("score", "bench"), required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-ctx", type=int, required=True)
    ap.add_argument("--budget-mib", type=int, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"task": args.task, "ok": False}
    sampler = MemSampler()
    try:
        result["baseline_mib"] = MemSampler.used_mib()
        sampler.start()
        eng = Engine(load(args.lib), args.model, args.max_ctx, args.budget_mib << 20)
        result["device_bytes"] = eng.lib.gptoss_device_bytes(eng.h)
        t0 = time.perf_counter()
        result.update((task_score if args.task == "score" else task_bench)(eng, Path(args.data), out))
        result["seconds"] = time.perf_counter() - t0
        eng.lib.gptoss_destroy(eng.h)
        result["ok"] = True
    except Exception as e:                                   # reported, judged outside
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-2000:]
    finally:
        sampler.done.set()
        sampler.join(timeout=2)
        result["peak_mib"] = sampler.peak
    (out / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
