"""End-to-end check of a golden file: re-score some of its items with the reference and gate them through
compare_golden's code path. Expect top-1 = 1.0 and a KL near zero; anything else means the golden, the corpus
rendering or the reference is not reproducible.

    python reference/scripts/self_check.py --model-dir /path/to/gpt-oss-20b \
        --golden reference/goldens/golden_v1 --items qa-capital tool-weather --threads 4

Using a different thread count than the golden run changes the floating-point summation order, so the KL this
reports is the reference's own numerical noise floor, the lower bound any engine threshold must sit above.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptoss_ref import GptOssConfig, Reference, SafetensorsWeights, compare, golden  # noqa: E402
from gptoss_ref import harmony_render as hr  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--golden", required=True, help="golden path without extension")
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "corpus/golden_v1.jsonl"))
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    manifest = json.loads(Path(args.golden + ".json").read_text())
    gfile = Path(args.golden + ".safetensors")
    if golden.file_sha256(gfile) != manifest["sha256"]:
        sys.exit("golden file does not match its manifest's sha256")
    records, meta = golden.load(gfile)
    ids = meta["corpus"]["ids"]
    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.corpus) if l.strip()}

    sel = [ids.index(i) for i in args.items]
    seqs = [torch.tensor(hr.render(items[i])) for i in args.items]
    for j, s in zip(sel, seqs):
        if not torch.equal(s.to(torch.int32), records[j]["tokens"]):
            sys.exit(f"{ids[j]}: rendering today differs from the golden's tokens (harmony or corpus drift)")

    ref = Reference(GptOssConfig.from_model_dir(args.model_dir), SafetensorsWeights(args.model_dir))
    hidden = ref.final_hidden(seqs)
    lm_head = ref.w.get("lm_head.weight")
    logits = [h @ lm_head.t() for h in hidden]
    sub = [records[j] for j in sel]
    cand = compare.candidate_from_logits(sub, logits)

    # Round-trip through the candidate file format, as an engine would.
    with tempfile.TemporaryDirectory() as d:
        compare.save_candidate(Path(d) / "c.safetensors", cand, golden_sha256=manifest["sha256"], engine="reference")
        cand, md = compare.load_candidate(Path(d) / "c.safetensors")
    summary, per_seq = compare.compare(sub, cand)
    print(json.dumps({"threads": args.threads, "golden_threads": meta.get("threads"), "summary": summary,
                      "per_seq": [dict(r, id=ids[j]) for r, j in zip(per_seq, sel)]}, indent=2))


if __name__ == "__main__":
    main()
