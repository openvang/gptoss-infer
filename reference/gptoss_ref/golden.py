"""Golden next-token distributions: produce them with the reference, store them, load them.

For each sequence of n tokens, position p (0 <= p < n-1) holds the reference distribution of token p+1 given
tokens 0..p, stored as:
    top_ids   int32 [n-1, k]   the k most likely tokens, most likely first
    top_lp    fp32  [n-1, k]   their log-probabilities
    lse       fp32  [n-1]      logsumexp of the raw logits (logits = logprob + lse)
    target_lp fp32  [n-1]      log-probability of the token that actually follows
Files are safetensors (no pickle); provenance lives in the safetensors metadata and in a JSON manifest that pins
the file by sha256.
"""
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FORMAT = "gptoss-golden/1"
FIELDS = ("tokens", "top_ids", "top_lp", "lse", "target_lp")


def source_sha256():
    """Hash of the reference implementation's sources, recorded with every golden it produces."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


@torch.inference_mode()
def score_hidden(hidden, tokens, lm_head, k=64, chunk=256):
    """Golden record for one sequence from its final hidden states [n, H] and the fp32 LM head [V, H]."""
    n = len(tokens)
    nxt = tokens.long().to(hidden.device)
    top_ids, top_lp, lse, target = [], [], [], []
    for c0 in range(0, n - 1, chunk):
        c1 = min(n - 1, c0 + chunk)
        logits = hidden[c0:c1] @ lm_head.t()
        z = torch.logsumexp(logits, dim=-1)
        lp = logits - z.unsqueeze(1)
        v, i = torch.topk(lp, k, dim=-1, sorted=True)
        top_ids.append(i.to(torch.int32).cpu())
        top_lp.append(v.cpu())
        lse.append(z.cpu())
        target.append(lp.gather(1, nxt[c0 + 1:c1 + 1].unsqueeze(1)).squeeze(1).cpu())
    return {"tokens": tokens.to(torch.int32).cpu(), "top_ids": torch.cat(top_ids), "top_lp": torch.cat(top_lp),
            "lse": torch.cat(lse), "target_lp": torch.cat(target)}


def score(ref, seqs, k=64, progress=None):
    """Run the reference over `seqs` (1-D LongTensors, each >= 2 tokens) and return one record per sequence."""
    hidden = ref.final_hidden(seqs, progress=progress)
    lm_head = ref.w.get("lm_head.weight").to(ref.device)
    return [score_hidden(h, s, lm_head, k=k) for h, s in zip(hidden, seqs)]


def save(path, records, meta):
    """Write records + metadata; returns the file's sha256."""
    path = Path(path)
    tensors = {f"s{j}.{f}": r[f].contiguous() for j, r in enumerate(records) for f in FIELDS}
    md = {"format": FORMAT, "n_seqs": str(len(records)), "k": str(records[0]["top_ids"].shape[1]),
          "meta": json.dumps(meta, sort_keys=True)}
    save_file(tensors, str(path), metadata=md)
    return file_sha256(path)


def load(path):
    """(records, meta) from a golden file."""
    with safe_open(str(path), framework="pt") as f:
        md = f.metadata()
    if md.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    t = load_file(str(path))
    records = [{f: t[f"s{j}.{f}"] for f in FIELDS} for j in range(int(md["n_seqs"]))]
    return records, json.loads(md["meta"])


def provenance(**extra):
    return {"reference_sha256": source_sha256(), "torch": torch.__version__, "threads": torch.get_num_threads(),
            "python": platform.python_version(), "machine": platform.machine(),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
