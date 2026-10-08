# reference/: the correctness anchor

Every engine change in this repository is judged against distributions produced here. This directory is
maintainer-owned: a pull request that touches it is never scored.

## What is here

| Path | What it does |
|---|---|
| `gptoss_ref/model.py` | fp32 forward pass of gpt-oss. It runs layer-major over a batch of sequences, so the full model never has to be resident; peak RAM is ~14 GB on CPU, mostly page cache. |
| `gptoss_ref/{mxfp4,rope,config,weights}.py` | MXFP4 decoding, YaRN (untruncated, float32 numerics as in OpenAI's and HF's code), config, lazy safetensors access |
| `gptoss_ref/harmony_render.py` | corpus item → token ids, using OpenAI's `openai-harmony` with a pinned date and analysis kept |
| `gptoss_ref/golden.py` | golden format: per position, top-k ids and log-probs, logsumexp, and the target token's log-prob |
| `gptoss_ref/compare.py` | candidate-vs-golden metrics (top-1 agreement; KL over top-k plus a tail bucket; NLL delta) and the engine contract |
| `corpus/golden_v1.jsonl` | 12 pinned conversations, 9,290 tokens: chat at low, medium and high effort; a full tool-call round trip; JSON output; multi-turn; Spanish, Chinese and Japanese; code; and a 5.4k-token document (license plus this repo's docs) that runs past YaRN's 4,096-token original context |
| `weights.lock.json` | the pinned checkpoint, `openai/gpt-oss-20b@6cee5e8`: every file's size, plus sha256 (LFS files) or git blob id (small files) |
| `goldens/*.json` | manifests that pin each golden file by sha256, with full provenance. The `.safetensors` files are release artifacts, not committed. |
| `scripts/` | `fetch_weights.py`, `make_golden.py`, `compare_golden.py`, `smoke_real.py`, `tokenizer_check.py` |
| `tests/` | 25 CPU tests, including a full-forward cross-check against Hugging Face transformers' independent gpt-oss code |

## How we know the reference is right

1. **Cross-check against transformers.** `tests/test_vs_transformers.py` builds a tiny random gpt-oss with MXFP4 experts, alternating window/full layers, sinks, biases and YaRN. Our logits match transformers' to 1e-4 for sequences longer than the window.
2. **Attention spec.** `tests/test_attention.py` checks the chunked attention against a transcription of OpenAI's `sdpa`. It also checks that a query sees exactly `window` keys, and that sinks absorb probability without adding value.
3. **YaRN.** `tests/test_rope.py` checks:
   - the frequencies are bit-identical to transformers';
   - the correction range is untruncated (8.0928, 17.3980);
   - truncating it would change exactly rotary pairs 9–17.

   That last case is the trap llama.cpp falls into.
4. **MXFP4.** `tests/test_mxfp4.py` decodes against a per-element spec implementation. It covers every code, both nibble orders and all 255 valid scales.
5. **Real checkpoint.** `scripts/smoke_real.py` gives confident, gpt-oss-like predictions on the real checkpoint, for example `final`, `<|message|>` and `<|return|>` at probability 1.00.

## Use it

```bash
python -m venv .venv && .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
.venv/bin/pip install -r reference/requirements.txt
.venv/bin/python reference/scripts/fetch_weights.py --out ../models/gpt-oss-20b    # 13.8 GB, verified
(cd reference && ../.venv/bin/python -m pytest -q tests)
.venv/bin/python reference/scripts/make_golden.py --model-dir ../models/gpt-oss-20b \
    --corpus reference/corpus/golden_v1.jsonl --out reference/goldens/golden_v1          # ~25 min on 8 CPU cores
```

## The engine contract

An engine is gated by running the golden's exact token sequences (teacher forcing) and writing a candidate file
with `gptoss_ref.compare.save_candidate`. For each sequence and position it holds:
- `lp_at_ref`: the engine's log-probs of the golden's top-k ids;
- `top1`: the engine's argmax;
- `target_lp`: optional, the engine's log-prob of the actual next token.

Then run:

```bash
.venv/bin/python reference/scripts/compare_golden.py --golden reference/goldens/golden_v1.safetensors \
    --candidate engine.safetensors --top1-min … --kl-mean-max … --kl-p99-max …
```

The candidate records the sha256 of the golden it was computed against; a mismatch is rejected.

## Reading the numbers: generated vs prompt positions

gpt-oss was post-trained with loss only on the tokens it generates. Inside system, developer, user and tool
messages, its next-token predictions are untrained and often poor. Inside the long user message in `long-doc`,
for example, it predicts "..." where the Apache License text plainly continues, and NLL there is 12–16 nats.

That is model behaviour, not a bug. Measured on the same license text with the same reference:

| Where the text sits | Mean NLL | Top-1 |
|---|---:|---:|
| Raw text, no chat format | 0.41 | 92.5 % |
| The assistant's own final message | 0.15 | 97.7 % |

So `compare.py` reports every metric twice:
- **Over all positions:** numerical fidelity. An exact engine reproduces even the odd distributions, so this is
  what the gate uses.
- **With a `gen_` prefix:** over the positions the model generates at inference (`harmony_render.generated_mask`).
  That covers everything after the role token in an assistant message, and the `<|start|>assistant` of an
  assistant message that follows another one. Quality-style numbers belong here.

`make_golden.py` prints per-item NLL on generated positions as a sanity check.

## Not done yet

- **Thresholds are uncalibrated.** `compare_golden.py` reports without a verdict until they are set. To set them:
  1. Measure an exact bf16 engine and the planned 8-bit-attention build against `golden_v1`, several times.
  2. Use the gap plus margin.
  3. Write the thresholds and their evidence here.

  Router near-ties make occasional per-position spikes normal, so gate on mean and p99, not max.

  The floor is known. `scripts/self_check.py` re-scored `qa-capital` and `tool-weather` at 4 threads against
  `golden_v1` (made at 8) and found:
  - top-1 1.000 at every position;
  - KL mean ~3e-7 over all positions (7.8e-7 over generated ones), p99 5.5e-6, max 1.5e-5.

  That is fp32 summation-order noise, the lowest threshold any engine could meet.
- **Top-k coverage is thin in places.** At its thinnest position, `golden_v1`'s top-64 holds only 42 % of the
  probability mass, at flat prompt-side distributions. The tail bucket keeps KL a lower bound there. A future
  golden should use k = 256, or report coverage-weighted KL.
- **Long-context goldens** (32k–128k) need a bigger machine or a GPU box. Generate them with the same script and
  a corpus whose items reach those lengths.
- **Greedy-continuation goldens**, for token-exact gates on the engine's own outputs, also belong on the GPU box.

## Findings worth carrying into the runtime

- **Tokenizer parity.** `tokenizer.json` tokenizes identically to harmony and tiktoken `o200k_harmony` on 82
  texts (edge cases plus the corpus), as long as special-token matching is off.
- **Untrusted text must stay plain text.** By default `tokenizer.json` turns user-typed `<|start|>`,
  `<|channel|>` and `<|message|>` into the real control ids. The server must encode untrusted text with
  special-token matching disabled (`encode_special_tokens = true`) and insert control tokens by id only.
  Otherwise a user can forge harmony messages.
- **Keep the analysis messages.** `openai-harmony`'s training renderer drops every analysis message by default,
  including the current turn's reasoning. The corpus renders with `auto_drop_analysis=False`, because inference
  keeps analysis, tool calls and the final answer in one context.
- **The last-turn role token.** After a user turn, the reference gives `assistant` low probability at
  `<|start|>`. That is expected: real prompts always end with `<|start|>assistant`, so the model never learned to
  predict it. It is no problem for the KL gate, which checks numerical fidelity at every position. It does
  inflate any quality-style NLL measured over prompt tokens.
