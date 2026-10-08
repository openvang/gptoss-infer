"""gpt-oss model configuration, read from the checkpoint's config.json (Hugging Face format)."""
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GptOssConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_local_experts: int
    num_experts_per_tok: int
    vocab_size: int
    sliding_window: int
    layer_types: tuple
    rms_norm_eps: float = 1e-5
    rope_theta: float = 150000.0
    rope_factor: float = 32.0
    rope_original_max_pos: int = 4096
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_truncate: bool = False
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    eos_token_ids: tuple = field(default_factory=tuple)

    @property
    def q_per_kv(self):
        return self.num_attention_heads // self.num_key_value_heads

    def window(self, layer):
        """Keys visible to a query in `layer`, counting the query itself; 0 means full causal attention."""
        return self.sliding_window if self.layer_types[layer] == "sliding_attention" else 0

    @classmethod
    def from_dict(cls, d, eos_token_ids=()):
        rope = d.get("rope_scaling") or d.get("rope_parameters") or {}
        if rope.get("rope_type", rope.get("type")) != "yarn":
            raise ValueError(f"expected YaRN rope scaling, got {rope}")
        n = d["num_hidden_layers"]
        layer_types = tuple(d.get("layer_types") or
                            ("sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n)))
        cfg = cls(
            num_hidden_layers=n,
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            num_local_experts=d.get("num_local_experts", d.get("num_experts")),
            num_experts_per_tok=d.get("num_experts_per_tok", d.get("experts_per_token")),
            vocab_size=d["vocab_size"],
            sliding_window=d["sliding_window"],
            layer_types=layer_types,
            rms_norm_eps=d.get("rms_norm_eps", 1e-5),
            rope_theta=float(d.get("rope_theta", rope.get("rope_theta", 150000.0))),
            rope_factor=float(rope["factor"]),
            rope_original_max_pos=int(rope["original_max_position_embeddings"]),
            rope_beta_fast=float(rope.get("beta_fast", 32.0)),
            rope_beta_slow=float(rope.get("beta_slow", 1.0)),
            # HF's default is truncate=True; gpt-oss explicitly ships False. Reading a missing key as True would
            # silently change 9 of the 32 frequency bands, so the checkpoint value is required, not defaulted.
            rope_truncate=bool(rope["truncate"]),
            swiglu_alpha=float(d.get("swiglu_alpha", 1.702)),
            swiglu_limit=float(d.get("swiglu_limit", 7.0)),
            eos_token_ids=tuple(eos_token_ids),
        )
        if len(cfg.layer_types) != n:
            raise ValueError("layer_types length does not match num_hidden_layers")
        if cfg.num_attention_heads % cfg.num_key_value_heads:
            raise ValueError("query heads must be a multiple of key/value heads")
        if cfg.hidden_size % 32 or cfg.intermediate_size % 32:
            raise ValueError("MXFP4 blocks need hidden and intermediate sizes divisible by 32")
        return cfg

    @classmethod
    def from_model_dir(cls, model_dir):
        model_dir = Path(model_dir)
        d = json.loads((model_dir / "config.json").read_text())
        eos = ()
        gen = model_dir / "generation_config.json"
        if gen.exists():
            e = json.loads(gen.read_text()).get("eos_token_id", ())
            eos = tuple(e) if isinstance(e, list) else (e,)
        return cls.from_dict(d, eos_token_ids=eos)
