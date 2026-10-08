"""fp32 reference forward pass for gpt-oss.

This is the correctness anchor for every engine and PR, so it optimizes for being obviously right, not fast:
  * all weights are upcast to fp32 (bf16 -> fp32 and MXFP4 -> fp32 are both exact) and all math is fp32;
  * the math follows OpenAI's gpt_oss/torch/model.py line for line, cross-checked against Hugging Face
    transformers in tests/test_vs_transformers.py;
  * it runs layer-major over a batch of sequences, so each layer's weights are read once per batch and the 13.8 GB
    checkpoint never has to be resident (peak RAM is a few GB on CPU).
"""
import math

import torch

from . import rope


def _p(layer, rest):
    return f"model.layers.{layer}.{rest}"


def rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def swiglu(t, alpha, limit):
    """gpt-oss's clamped SwiGLU over interleaved rows: even = glu, odd = linear."""
    glu, lin = t[..., 0::2], t[..., 1::2]
    glu = glu.clamp(max=limit)               # clamped from above only
    lin = lin.clamp(min=-limit, max=limit)
    return glu * torch.sigmoid(alpha * glu) * (lin + 1)


def attention(cfg, x, aw, window, cos, sin, max_scores=1 << 26, kv_dtype=None):
    """Causal GQA attention with learned sinks for one sequence. x is the normed input [n, H].

    window > 0: a query at position p sees keys p-window+1 .. p (window keys, itself included); 0 = full causal.
    The sink is a per-head logit appended as one extra softmax column and then dropped: it takes probability
    mass but contributes no value. It is NOT multiplied by the softmax scale.
    kv_dtype: if set, rotated K and V are rounded to it before use, mimicking an engine's KV cache format.
    """
    n = x.shape[0]
    hq, hkv, d, g = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.q_per_kv
    q = (x @ aw["q_w"].t() + aw["q_b"]).view(n, hq, d)
    k = (x @ aw["k_w"].t() + aw["k_b"]).view(n, hkv, d)
    v = (x @ aw["v_w"].t() + aw["v_b"]).view(n, hkv, d)
    q, k = rope.apply(q, cos, sin), rope.apply(k, cos, sin)
    if kv_dtype is not None:
        k, v = k.to(kv_dtype).to(torch.float32), v.to(kv_dtype).to(torch.float32)
    q = q.view(n, hkv, g, d)                 # query head h uses key/value head h // g
    scale = 1.0 / math.sqrt(d)
    sinks = aw["sinks"].view(hkv, g, 1, 1)
    out = torch.empty(n, hkv, g, d, device=x.device)
    # Bound the [hkv, g, chunk, keys] score tensor so long sequences stay within memory. A chunk of c queries sees
    # up to n keys under full attention, and c + window - 1 keys under a window.
    budget = max_scores / hq
    if window == 0:
        chunk = int(budget // n)
    else:
        chunk = int((-(window - 1) + math.sqrt((window - 1) ** 2 + 4 * budget)) / 2)
    chunk = max(1, min(n, chunk))
    for q0 in range(0, n, chunk):
        q1 = min(n, q0 + chunk)
        k0 = 0 if window == 0 else max(0, q0 - window + 1)
        s = torch.einsum("qhgd,khd->hgqk", q[q0:q1], k[k0:q1]) * scale
        qpos = torch.arange(q0, q1, device=x.device).unsqueeze(1)
        kpos = torch.arange(k0, q1, device=x.device).unsqueeze(0)
        allowed = kpos <= qpos
        if window:
            allowed &= kpos > qpos - window
        s.masked_fill_(~allowed, float("-inf"))
        s = torch.cat((s, sinks.expand(hkv, g, q1 - q0, 1)), dim=-1)
        w = torch.softmax(s, dim=-1)[..., :-1]
        out[q0:q1] = torch.einsum("hgqk,khd->qhgd", w, v[k0:q1])
    return out.reshape(n, hq * d) @ aw["o_w"].t() + aw["o_b"]


class Reference:
    def __init__(self, cfg, weights, threads=None, device="cpu", kv_dtype=None):
        self.cfg, self.w, self.device, self.kv_dtype = cfg, weights, torch.device(device), kv_dtype
        if threads:
            torch.set_num_threads(threads)

    def _attn_weights(self, layer):
        g = lambda rest: self.w.get(_p(layer, rest)).to(self.device)
        return {"q_w": g("self_attn.q_proj.weight"), "q_b": g("self_attn.q_proj.bias"),
                "k_w": g("self_attn.k_proj.weight"), "k_b": g("self_attn.k_proj.bias"),
                "v_w": g("self_attn.v_proj.weight"), "v_b": g("self_attn.v_proj.bias"),
                "o_w": g("self_attn.o_proj.weight"), "o_b": g("self_attn.o_proj.bias"),
                "sinks": g("self_attn.sinks"),
                "ln1": g("input_layernorm.weight"), "ln2": g("post_attention_layernorm.weight")}

    def moe(self, layer, h, return_routing=False):
        """MoE block on normed tokens h [T, H]: top-k of the router logits, then softmax over those k."""
        cfg = self.cfg
        router_w = self.w.get(_p(layer, "mlp.router.weight")).to(self.device)
        logits = h @ router_w.t() + self.w.get(_p(layer, "mlp.router.bias")).to(self.device)
        top_val, top_idx = torch.topk(logits, cfg.num_experts_per_tok, dim=-1, sorted=True)
        gate = torch.softmax(top_val, dim=-1)
        out = torch.zeros_like(h)
        for e in torch.unique(top_idx).tolist():
            rows, slot = (top_idx == e).nonzero(as_tuple=True)
            w1, b1, w2, b2 = self.w.expert(layer, e, device=self.device)
            y = swiglu(h[rows] @ w1.t() + b1, cfg.swiglu_alpha, cfg.swiglu_limit) @ w2.t() + b2
            out.index_add_(0, rows, y * gate[rows, slot].unsqueeze(1))
        return (out, top_idx) if return_routing else out

    @torch.inference_mode()
    def final_hidden(self, seqs, progress=None):
        """Final-normed hidden states [n_j, H] for each token sequence (1-D LongTensors), processed layer-major."""
        cfg = self.cfg
        xs = [self.w.get_rows("model.embed_tokens.weight", s).to(self.device) for s in seqs]
        trig = [tuple(t.to(self.device) for t in rope.cos_sin(cfg, torch.arange(len(s)))) for s in seqs]
        for layer in range(cfg.num_hidden_layers):
            aw = self._attn_weights(layer)
            for j, x in enumerate(xs):
                xs[j] = x + attention(cfg, rms_norm(x, aw["ln1"], cfg.rms_norm_eps), aw, cfg.window(layer), *trig[j],
                                      kv_dtype=self.kv_dtype)
            # One MoE pass over every token of every sequence, so each expert is decoded once per layer.
            h = torch.cat([rms_norm(x, aw["ln2"], cfg.rms_norm_eps) for x in xs])
            out = self.moe(layer, h)
            xs = [x + o for x, o in zip(xs, out.split([len(x) for x in xs]))]
            if progress:
                progress(layer)
        norm = self.w.get("model.norm.weight").to(self.device)
        return [rms_norm(x, norm, cfg.rms_norm_eps) for x in xs]

    @torch.inference_mode()
    def logits(self, tokens):
        """Full fp32 logits [n, V] for one short sequence (tests; the vocab makes this large for long inputs)."""
        h = self.final_hidden([tokens])[0]
        return h @ self.w.get("lm_head.weight").to(self.device).t()
