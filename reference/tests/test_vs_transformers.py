"""Cross-check the whole reference forward against Hugging Face transformers' independent gpt-oss implementation."""
import pytest
import torch

from conftest import TINY, dense_experts
from gptoss_ref import model
from gptoss_ref.weights import DictWeights

transformers = pytest.importorskip("transformers")


def hf_model(tensors, cfg):
    rope = dict(TINY["rope_scaling"], rope_theta=TINY["rope_theta"])
    hf_cfg = transformers.GptOssConfig(
        vocab_size=TINY["vocab_size"], hidden_size=TINY["hidden_size"], intermediate_size=TINY["intermediate_size"],
        num_hidden_layers=TINY["num_hidden_layers"], num_attention_heads=TINY["num_attention_heads"],
        num_key_value_heads=TINY["num_key_value_heads"], head_dim=TINY["head_dim"],
        num_local_experts=TINY["num_local_experts"], num_experts_per_tok=TINY["num_experts_per_tok"],
        sliding_window=TINY["sliding_window"], layer_types=TINY["layer_types"], rms_norm_eps=TINY["rms_norm_eps"],
        max_position_embeddings=TINY["max_position_embeddings"], rope_parameters=rope,
        swiglu_limit=TINY["swiglu_limit"], tie_word_embeddings=False, attention_bias=True)
    hf_cfg._attn_implementation = "eager"
    m = transformers.GptOssForCausalLM(hf_cfg).eval()
    state = {k: v for k, v in tensors.items() if not k.endswith(("_blocks", "_scales"))}
    state |= dense_experts(tensors, cfg)
    missing, unexpected = m.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if "rotary" not in k], missing
    return m


@pytest.mark.parametrize("n", [1, 7, 40])
def test_logits_match_transformers(tiny_cfg, tiny_weights, n):
    tokens = torch.randint(0, TINY["vocab_size"], (n,), generator=torch.Generator().manual_seed(n))
    ours = model.Reference(tiny_cfg, DictWeights(tiny_weights)).logits(tokens)
    with torch.no_grad():
        theirs = hf_model(tiny_weights, tiny_cfg)(tokens.unsqueeze(0)).logits[0]
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)


def test_batch_is_independent_per_sequence(tiny_cfg, tiny_weights):
    """Layer-major batching must give each sequence exactly what it gets alone (positions restart at 0)."""
    ref = model.Reference(tiny_cfg, DictWeights(tiny_weights))
    g = torch.Generator().manual_seed(9)
    seqs = [torch.randint(0, TINY["vocab_size"], (n,), generator=g) for n in (12, 30, 3)]
    together = ref.final_hidden(seqs)
    for s, h in zip(seqs, together):
        torch.testing.assert_close(h, ref.final_hidden([s])[0], rtol=1e-5, atol=1e-5)
