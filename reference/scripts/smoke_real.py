"""Sanity-check the reference on the real checkpoint: does it predict each corpus item's own next tokens sensibly?

    python reference/scripts/smoke_real.py --model-dir /path/to/gpt-oss-20b --item qa-capital

Prints, for every position in the item's last message, the reference's rank and probability of the token that
actually follows, plus the top-3 alternatives. A healthy model ranks most of a fluent answer's tokens first.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptoss_ref import GptOssConfig, Reference, SafetensorsWeights, golden  # noqa: E402
from gptoss_ref import harmony_render as hr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "corpus/golden_v1.jsonl"))
    ap.add_argument("--item", default="qa-capital")
    ap.add_argument("--show", type=int, default=40, help="positions to print, counted back from the end")
    args = ap.parse_args()

    item = next(json.loads(l) for l in open(args.corpus) if json.loads(l)["id"] == args.item)
    ids = torch.tensor(hr.render(item))
    enc = hr.encoding()
    cfg = GptOssConfig.from_model_dir(args.model_dir)
    ref = Reference(cfg, SafetensorsWeights(args.model_dir))

    t0 = time.time()
    rec = golden.score(ref, [ids], k=8,
                       progress=lambda l: print(f"  layer {l:2d} done at {time.time() - t0:6.1f} s", flush=True))[0]
    dt = time.time() - t0
    print(f"{len(ids)} tokens in {dt:.1f} s ({len(ids) / dt:.1f} tok/s on {torch.get_num_threads()} threads)")

    top1 = (rec["top_ids"][:, 0] == ids[1:]).double()
    print(f"top-1 matches the corpus's next token at {top1.mean().item():.1%} of positions")
    for p in range(max(0, len(ids) - 1 - args.show), len(ids) - 1):
        nxt = int(ids[p + 1])
        row = rec["top_ids"][p].tolist()
        rank = row.index(nxt) + 1 if nxt in row else ">8"
        alts = ", ".join(f"{enc.decode([t])!r}:{rec['top_lp'][p, j].exp().item():.2f}" for j, t in enumerate(row[:3]))
        print(f"  {p:5d} next={enc.decode([nxt])!r:18s} p={rec['target_lp'][p].exp().item():.3f} rank={rank}  top3: {alts}")


if __name__ == "__main__":
    main()
