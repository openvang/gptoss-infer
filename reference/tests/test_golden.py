import pytest
import torch

from gptoss_ref import compare, golden, model
from gptoss_ref.weights import DictWeights


def _setup(tiny_cfg, tiny_weights, tmp_path):
    ref = model.Reference(tiny_cfg, DictWeights(tiny_weights))
    g = torch.Generator().manual_seed(5)
    seqs = [torch.randint(0, tiny_cfg.vocab_size, (n,), generator=g) for n in (25, 9)]
    records = golden.score(ref, seqs, k=16)
    sha = golden.save(tmp_path / "g.safetensors", records, golden.provenance(note="test"))
    loaded, meta = golden.load(tmp_path / "g.safetensors")
    return ref, seqs, records, loaded, meta, sha


def test_roundtrip_and_self_compare(tiny_cfg, tiny_weights, tmp_path):
    ref, seqs, records, loaded, meta, sha = _setup(tiny_cfg, tiny_weights, tmp_path)
    assert len(sha) == 64 and meta["note"] == "test" and meta["reference_sha256"] == golden.source_sha256()
    for a, b in zip(records, loaded):
        for f in golden.FIELDS:
            assert torch.equal(a[f], b[f])
    cand = compare.candidate_from_logits(loaded, [ref.logits(s) for s in seqs])
    summary, per_seq = compare.compare(loaded, cand)
    assert summary["top1"] == 1.0 and summary["kl_max"] < 1e-6 and abs(summary["nll_delta_mean"]) < 1e-6
    ok, reasons = compare.verdict(summary, top1_min=0.99, kl_mean_max=1e-3, kl_p99_max=1e-2)
    assert ok, reasons


def test_records_are_consistent(tiny_cfg, tiny_weights, tmp_path):
    _, seqs, records, *_ = _setup(tiny_cfg, tiny_weights, tmp_path)
    for s, r in zip(seqs, records):
        n = len(s)
        assert r["top_ids"].shape == (n - 1, 16) and r["lse"].shape == (n - 1,)
        assert torch.all(r["top_lp"][:, :-1] >= r["top_lp"][:, 1:])          # sorted, most likely first
        assert torch.all(r["target_lp"] <= 0)


def test_perturbed_candidate_is_caught(tiny_cfg, tiny_weights, tmp_path):
    ref, seqs, _, loaded, _, _ = _setup(tiny_cfg, tiny_weights, tmp_path)
    noisy = [ref.logits(s) + torch.randn(len(s), tiny_cfg.vocab_size, generator=torch.Generator().manual_seed(1))
             for s in seqs]
    summary, _ = compare.compare(loaded, compare.candidate_from_logits(loaded, noisy))
    assert summary["kl_mean"] > 0.05
    ok, reasons = compare.verdict(summary, top1_min=0.99, kl_mean_max=1e-3, kl_p99_max=1e-2)
    assert not ok and reasons


def test_candidate_file_roundtrip_is_bound_to_its_golden(tiny_cfg, tiny_weights, tmp_path):
    ref, seqs, _, loaded, _, sha = _setup(tiny_cfg, tiny_weights, tmp_path)
    cand = compare.candidate_from_logits(loaded, [ref.logits(s) for s in seqs])
    compare.save_candidate(tmp_path / "c.safetensors", cand, golden_sha256=sha, engine="test")
    back, md = compare.load_candidate(tmp_path / "c.safetensors")
    assert md["golden_sha256"] == sha == golden.file_sha256(tmp_path / "g.safetensors")
    for a, b in zip(cand, back):
        for f in compare.CAND_FIELDS:
            assert torch.equal(a[f], b[f])


def test_shape_mismatch_is_rejected(tiny_cfg, tiny_weights, tmp_path):
    ref, seqs, _, loaded, _, _ = _setup(tiny_cfg, tiny_weights, tmp_path)
    cand = compare.candidate_from_logits(loaded, [ref.logits(s) for s in seqs])
    cand[0]["lp_at_ref"] = cand[0]["lp_at_ref"][:-1]
    with pytest.raises(ValueError):
        compare.compare(loaded, cand)
