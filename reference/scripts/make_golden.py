"""Generate a golden file from a corpus with the fp32 reference.

    python reference/scripts/make_golden.py --model-dir /path/to/gpt-oss-20b \
        --corpus reference/corpus/golden_v1.jsonl --out reference/goldens/golden_v1

Writes <out>.safetensors (the distributions) and <out>.json (the manifest that pins it). The manifest records the
checkpoint revision, the corpus and rendered-token hashes, the reference source hash and the machine, so anyone
can regenerate the file and check they get the same bytes on the same software stack.
"""
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptoss_ref import GptOssConfig, Reference, SafetensorsWeights, golden  # noqa: E402
from gptoss_ref import harmony_render as hr  # noqa: E402

LOCK = Path(__file__).resolve().parents[1] / "weights.lock.json"


def check_checkpoint(model_dir):
    """Fail unless fetch_weights.py verified this directory against the lock (sizes are re-checked here)."""
    lock = json.loads(LOCK.read_text())
    marker = Path(model_dir) / "VERIFIED.json"
    if not marker.exists() or json.loads(marker.read_text()).get("revision") != lock["revision"]:
        sys.exit(f"{model_dir} was not verified against {LOCK.name}; run reference/scripts/fetch_weights.py first")
    for name, e in lock["files"].items():
        if (Path(model_dir) / name).stat().st_size != e["size"]:
            sys.exit(f"{name}: size differs from the lock")
    return lock


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="corpus ids to include (default: all)")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    lock = check_checkpoint(args.model_dir)
    corpus_bytes = Path(args.corpus).read_bytes()
    items = [json.loads(line) for line in corpus_bytes.decode().splitlines() if line.strip()]
    if args.only:
        items = [it for it in items if it["id"] in set(args.only)]
    seqs = [torch.tensor(hr.render(it)) for it in items]
    cfg = GptOssConfig.from_model_dir(args.model_dir)
    for it, s in zip(items, seqs):
        if int(s.max()) >= cfg.vocab_size:
            sys.exit(f"{it['id']}: token id outside the vocabulary")
    print(f"{len(items)} items, {sum(map(len, seqs))} tokens; scoring with {torch.get_num_threads()} threads",
          flush=True)

    ref = Reference(cfg, SafetensorsWeights(args.model_dir))
    t0 = time.time()
    records = golden.score(ref, seqs, k=args.k,
                           progress=lambda l: print(f"  layer {l:2d} done at {time.time() - t0:7.1f} s", flush=True))
    elapsed = time.time() - t0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = golden.provenance(
        model={"repo": lock["repo"], "revision": lock["revision"]},
        corpus={"file": Path(args.corpus).name, "sha256": hashlib.sha256(corpus_bytes).hexdigest(),
                "ids": [it["id"] for it in items]},
        harmony={"package": "openai-harmony", "version": importlib.metadata.version("openai-harmony"),
                 "encoding": "HARMONY_GPT_OSS", "auto_drop_analysis": False},
        k=args.k, elapsed_s=round(elapsed, 1))
    sha = golden.save(out.with_suffix(".safetensors"), records, meta)
    manifest = {"format": golden.FORMAT, "file": out.with_suffix(".safetensors").name, "sha256": sha,
                "items": [{"id": it["id"], "tokens": len(s),
                           "tokens_sha256": hashlib.sha256(s.to(torch.int32).numpy().tobytes()).hexdigest()}
                          for it, s in zip(items, seqs)],
                **meta}
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out.with_suffix('.safetensors')} (sha256 {sha[:16]}…) in {elapsed:.0f} s")
    # Quality sanity check on the positions the model generates. NLL over prompt positions is not meaningful:
    # gpt-oss was never trained to predict system, user or tool text.
    for it, r in zip(items, records):
        gen = torch.tensor(hr.generated_mask(r["tokens"]), dtype=torch.bool)
        top1 = (r["top_ids"][:, 0] == r["tokens"][1:])
        print(f"  {it['id']:16s} {len(r['tokens']):6d} tokens, {int(gen.sum()):4d} generated: "
              f"NLL {-r['target_lp'][gen].mean().item():.3f} (all positions {-r['target_lp'].mean().item():.3f}), "
              f"corpus token top-1 {top1[gen].double().mean().item():.1%}")


if __name__ == "__main__":
    main()
