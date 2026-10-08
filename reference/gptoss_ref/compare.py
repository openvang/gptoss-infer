"""Compare an engine's next-token distributions against a golden file.

The engine contract (what a candidate dump must contain, per sequence and per scored position p):
    lp_at_ref  fp32  [n-1, k]  the engine's log-probability of each of the golden's top-k token ids, same order
    top1       int32 [n-1]     the engine's own most likely next token
    target_lp  fp32  [n-1]     (optional) the engine's log-probability of the token that actually follows
The engine must run the golden's exact token sequences (teacher forcing); nothing here re-tokenizes.

KL(ref || engine) is computed over the golden's top-k ids plus one "everything else" bucket. That coarsening can
only lower the KL (data-processing inequality), so a pass is honest only while the golden's top-k mass is close
to 1; the report includes the minimum covered mass so that can be checked.
"""
import json
import math

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

CAND_FIELDS = ("lp_at_ref", "top1", "target_lp")
CAND_FORMAT = "gptoss-candidate/1"


def save_candidate(path, records, golden_sha256, engine=""):
    """Write a candidate dump bound to the golden file it was computed against."""
    tensors = {f"s{j}.{f}": r[f].contiguous() for j, r in enumerate(records) for f in CAND_FIELDS
               if r.get(f) is not None}
    save_file(tensors, str(path), metadata={"format": CAND_FORMAT, "n_seqs": str(len(records)),
                                            "golden_sha256": golden_sha256, "engine": engine})


def load_candidate(path):
    """(records, metadata) from a candidate dump."""
    with safe_open(str(path), framework="pt") as f:
        md = f.metadata()
    if md.get("format") != CAND_FORMAT:
        raise ValueError(f"{path}: not a {CAND_FORMAT} file")
    t = load_file(str(path))
    records = [{f: t.get(f"s{j}.{f}") for f in CAND_FIELDS} for j in range(int(md["n_seqs"]))]
    return records, md


def candidate_from_logits(golden_records, logits_list):
    """Build a candidate dump from full logits [n, V] per sequence (Python engines, tests)."""
    out = []
    for g, logits in zip(golden_records, logits_list):
        lp = torch.log_softmax(logits[:-1].to(torch.float32), dim=-1)
        out.append({"lp_at_ref": lp.gather(1, g["top_ids"].long()),
                    "top1": lp.argmax(dim=-1).to(torch.int32),
                    "target_lp": lp.gather(1, g["tokens"][1:].long().unsqueeze(1)).squeeze(1)})
    return out


def _kl_rows(ref_lp, cand_lp):
    p = ref_lp.double().exp()
    q = cand_lp.double().exp()
    head = (p * (ref_lp.double() - cand_lp.double())).sum(dim=1)
    p_tail = (1 - p.sum(dim=1)).clamp_min(0)
    q_tail = (1 - q.sum(dim=1)).clamp_min(1e-12)
    tail = torch.where(p_tail > 0, p_tail * (p_tail.clamp_min(1e-300).log() - q_tail.log()), torch.zeros_like(p_tail))
    return (head + tail).clamp_min(0), p.sum(dim=1)


def _stats(kl, hits, prefix=""):
    if len(kl) == 0:
        return {f"{prefix}positions": 0}
    return {f"{prefix}positions": len(kl), f"{prefix}top1": hits.double().mean().item(),
            f"{prefix}kl_mean": kl.mean().item(), f"{prefix}kl_p99": torch.quantile(kl, 0.99).item(),
            f"{prefix}kl_max": kl.max().item()}


def compare(golden_records, cand_records):
    """Summary metrics plus a per-sequence breakdown.

    Metrics are reported twice: over every position (numerical fidelity, what the gate uses) and, with a `gen_`
    prefix, over positions whose next token the model itself generates at inference (see
    harmony_render.generated_mask). The second view is what users experience; gpt-oss's predictions inside
    prompt messages are untrained, so quality-style numbers belong there.
    """
    from .harmony_render import generated_mask

    per_seq, all_kl, all_hits, gen_kl, gen_hits, dnll, min_cover = [], [], [], [], [], [], 1.0
    for j, (g, c) in enumerate(zip(golden_records, cand_records)):
        if c["lp_at_ref"].shape != g["top_lp"].shape:
            raise ValueError(f"sequence {j}: lp_at_ref {tuple(c['lp_at_ref'].shape)} != golden {tuple(g['top_lp'].shape)}")
        kl, cover = _kl_rows(g["top_lp"], c["lp_at_ref"])
        hits = (c["top1"].long() == g["top_ids"][:, 0].long())
        gen = torch.tensor(generated_mask(g["tokens"]), dtype=torch.bool)
        row = {"seq": j, **_stats(kl, hits), **_stats(kl[gen], hits[gen], "gen_"),
               "min_topk_mass": cover.min().item()}
        if c.get("target_lp") is not None:
            d = (g["target_lp"].double() - c["target_lp"].double())   # >0 means the engine is less sure of the truth
            row["nll_delta_mean"] = d.mean().item()
            dnll.append(d)
        per_seq.append(row)
        all_kl.append(kl)
        all_hits.append(hits)
        gen_kl.append(kl[gen])
        gen_hits.append(hits[gen])
        min_cover = min(min_cover, row["min_topk_mass"])
    summary = {**_stats(torch.cat(all_kl), torch.cat(all_hits)),
               **_stats(torch.cat(gen_kl), torch.cat(gen_hits), "gen_"), "min_topk_mass": min_cover}
    if dnll:
        summary["nll_delta_mean"] = torch.cat(dnll).mean().item()
    return summary, per_seq


def verdict(summary, top1_min, kl_mean_max, kl_p99_max):
    """Gate decision. The thresholds must come from calibration (an exact engine vs this golden), not guesses."""
    reasons = []
    if summary["top1"] < top1_min:
        reasons.append(f"top1 {summary['top1']:.4f} < {top1_min}")
    if summary["kl_mean"] > kl_mean_max:
        reasons.append(f"kl_mean {summary['kl_mean']:.5f} > {kl_mean_max}")
    if summary["kl_p99"] > kl_p99_max:
        reasons.append(f"kl_p99 {summary['kl_p99']:.5f} > {kl_p99_max}")
    if not math.isfinite(summary["kl_mean"]):
        reasons.append("non-finite KL")
    return (not reasons), reasons


def report(summary, per_seq):
    return json.dumps({"summary": summary, "per_seq": per_seq}, indent=2)
