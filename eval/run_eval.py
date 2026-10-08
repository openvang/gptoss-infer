#!/usr/bin/env python3
"""Evaluate a candidate commit against a base commit on this GPU box and write a verdict.

Each commit is exported with `git archive`, built in the eval image, and measured in fresh containers that run
with no network and see only read-only weights, read-only data, the harness from this (base) checkout and their
own output directory. The candidate's code never runs outside a container; all judging happens here.

    python eval/run_eval.py --repo /data/gptoss-eval/repo --base <sha> --cand <sha> --models /data/models \
        --goldens /data/gptoss-eval/goldens --work /data/gptoss-eval/runs [--pairs 5]

Prints the verdict JSON and, last, the path of verdict.json. Exit 0 with a verdict (including REJECT), 2 on
infrastructure failure (the base itself failed: nothing can be judged).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "reference"))
sys.path.insert(0, str(HERE))
import policy  # noqa: E402
from gptoss_ref import compare, golden  # noqa: E402

IMAGE = "gptoss-eval:1"
MODEL = "gpt-oss-20b"
SCORE_CTX = 8192
BENCH = [{"name": "s128", "prompt": 128, "decode": 128}, {"name": "s4k", "prompt": 4096, "decode": 128}]
AXIS_SOURCE = {"decode@128": ("s128", "decode_tok_s"), "decode@4k": ("s4k", "decode_tok_s"),
               "prefill@4k": ("s4k", "prefill_tok_s")}


class InfraError(RuntimeError):
    pass


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def export(repo, sha, dst):
    dst.mkdir(parents=True)
    archive = subprocess.Popen(["git", "-C", str(repo), "archive", sha], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dst)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise InfraError(f"git archive {sha} failed")


def docker(mounts, cmd, log, timeout):
    args = ["docker", "run", "--rm", "--network", "none", "--gpus", "all", "--entrypoint", ""]
    for host, ctr, mode in mounts:
        args += ["-v", f"{host}:{ctr}:{mode}"]
    with open(log, "w") as f:
        try:
            return subprocess.run(args + [IMAGE, "sh", "-c", cmd], stdout=f, stderr=subprocess.STDOUT,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[timeout after {timeout} s]\n")
            return 124


def build(role_dir):
    lib = role_dir / "lib"
    lib.mkdir()
    cmd = ("cp -a /src /work/src && cmake -S /work/src -B /work/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a "
           "&& cmake --build /work/build --target gptoss && cp /work/build/libgptoss.so /out/")
    rc = docker([(role_dir / "src", "/src", "ro"), (lib, "/out", "rw")], cmd, role_dir / "build.log", 1800)
    return rc == 0 and (lib / "libgptoss.so").exists()


def run_task(task, role_dir, data, models, out, max_ctx):
    out.mkdir(parents=True)
    cmd = (f"python3 /harness/driver.py --lib /engine/libgptoss.so --model /models/{MODEL} --task {task} --data /data "
           f"--out /out --max-ctx {max_ctx} --budget-mib {policy.VRAM_BUDGET_MIB}")
    # Not /lib: on Ubuntu 24.04 /lib is /usr/lib, and shadowing it stops the NVIDIA runtime injecting the driver.
    mounts = [(role_dir / "lib", "/engine", "ro"), (HERE / "driver", "/harness", "ro"), (data, "/data", "ro"),
              (models, "/models", "ro"), (out, "/out", "rw")]
    rc = docker(mounts, cmd, out.parent / f"{out.name}.log", 3600)
    path = out / "result.json"
    return json.loads(path.read_text()) if path.exists() else {"ok": False, "error": f"no result (exit {rc})"}


def prepare_data(data, records):
    data.mkdir()
    for j, r in enumerate(records):
        np.save(data / f"tokens_{j}.npy", r["tokens"].numpy())
        np.save(data / f"ids_{j}.npy", r["top_ids"].numpy())
    (data / "golden.json").write_text(json.dumps({"n_seqs": len(records), "k": int(records[0]["top_ids"].shape[1])}))
    longest = max(records, key=lambda r: len(r["tokens"]))
    need = max(s["prompt"] + s["decode"] + 1 for s in BENCH)
    if len(longest["tokens"]) < need:
        raise InfraError("golden corpus too short for the bench scenarios")
    np.save(data / "bench_tokens.npy", longest["tokens"].numpy())
    (data / "bench.json").write_text(json.dumps(BENCH))


def golden_summary(records, out, path):
    cands = [{"lp_at_ref": torch.from_numpy(np.load(out / f"{path}_lp_{j}.npy")),
              "top1": torch.from_numpy(np.load(out / f"{path}_top1_{j}.npy")),
              "target_lp": torch.from_numpy(np.load(out / f"{path}_target_{j}.npy"))} for j in range(len(records))]
    summary, _ = compare.compare(records, cands)
    return summary


def gpu_info():
    q = "name,driver_version,power.limit,clocks.max.sm"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip()
    return dict(zip(q.split(","), [x.strip() for x in out.split(",")]))


def evaluate(args, run, verdict):
    repo, models = Path(args.repo), Path(args.models)
    changed = git(repo, "diff", "--name-only", verdict["base"], verdict["cand"]).splitlines()
    verdict["changed_files"] = changed
    touched = [f for f in changed if f.startswith(policy.PROTECTED)]
    if touched:
        verdict.update(label="skipped", tier=None, reasons=[f"touches maintainer-owned paths: {', '.join(touched)}"])
        return
    for role in ("base", "cand"):
        export(repo, verdict[role], run / role / "src")
    # The golden file is a release artifact; only its manifest (with the sha256) is committed.
    manifest = json.loads((ROOT / "reference/goldens/golden_v1.json").read_text())
    gfile = Path(args.goldens) / manifest["file"]
    if not gfile.exists() or golden.file_sha256(gfile) != manifest["sha256"]:
        raise InfraError(f"{gfile} is missing or does not match the committed manifest")
    records, meta = golden.load(gfile)
    prepare_data(run / "data", records)

    if not build(run / "base"):
        raise InfraError("base failed to build; see base/build.log")
    if not build(run / "cand"):
        verdict.update(policy.decide(False, [], 0, {}, 0.0))
        return

    summaries, peaks = {}, {"base": 0, "cand": 0}
    for role in ("base", "cand"):
        res = run_task("score", run / role, run / "data", models, run / role / "score", SCORE_CTX)
        if not res["ok"]:
            if role == "base":
                raise InfraError(f"base score run failed: {res.get('error')}")
            verdict.update(label="REJECT", tier=None, reasons=[f"golden run failed: {res.get('error')}"])
            return
        peaks[role] = max(peaks[role], res["peak_mib"] - res["baseline_mib"])
        summaries[role] = {"decode": golden_summary(records, run / role / "score", "dec"),
                           "prefill": golden_summary(records, run / role / "score", "pf")}
    verdict["golden"] = summaries
    correct = policy.correctness_reasons(summaries["cand"], summaries["base"])

    values = {axis: ([], []) for axis in policy.AXES}
    timed_lp = {"base": None, "cand": None}
    max_ctx = max(s["prompt"] + s["decode"] + 1 for s in BENCH)
    for i in range(args.pairs):
        for role in (("base", "cand") if i % 2 == 0 else ("cand", "base")):   # alternate order: drift cancels
            res = run_task("bench", run / role, run / "data", models, run / role / f"bench{i}", max_ctx)
            if not res["ok"]:
                if role == "base":
                    raise InfraError(f"base bench run failed: {res.get('error')}")
                verdict.update(label="REJECT", tier=None, reasons=[f"bench run failed: {res.get('error')}"])
                return
            peaks[role] = max(peaks[role], res["peak_mib"] - res["baseline_mib"])
            for axis, (scenario, key) in AXIS_SOURCE.items():
                values[axis][0 if role == "base" else 1].append(res["scenarios"][scenario][key])
            if timed_lp[role] is None:
                timed_lp[role] = np.concatenate([np.array(res["scenarios"][s["name"]]["target_lp"]) for s in BENCH])
    integrity = float(np.mean(np.abs(timed_lp["cand"] - timed_lp["base"])))
    verdict.update(vram_peak_mib=peaks, integrity_mean_abs_lp=integrity,
                   raw=({axis: {"base": b, "cand": c} for axis, (b, c) in values.items()}))
    verdict.update(policy.decide(True, correct, peaks["cand"], values, integrity))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--goldens", required=True, help="directory holding the golden .safetensors files")
    ap.add_argument("--work", required=True)
    ap.add_argument("--pairs", type=int, default=5)
    args = ap.parse_args()
    base, cand = git(args.repo, "rev-parse", args.base), git(args.repo, "rev-parse", args.cand)
    run = Path(args.work) / f"{time.strftime('%Y%m%d-%H%M%S')}-{base[:8]}-{cand[:8]}"
    run.mkdir(parents=True)
    verdict = {"policy_version": policy.POLICY_VERSION, "base": base, "cand": cand, "run": run.name,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu_info(),
               "vram_budget_mib": policy.VRAM_BUDGET_MIB}
    code = 0
    try:
        evaluate(args, run, verdict)
    except InfraError as e:
        verdict.update(label="error", tier=None, reasons=[str(e)])
        code = 2
    verdict["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: verdict.get(k) for k in ("label", "tier", "reasons", "axes", "vram_peak_mib")}, indent=2))
    print(run / "verdict.json")
    sys.exit(code)


if __name__ == "__main__":
    main()
