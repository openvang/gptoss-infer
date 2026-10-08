# gptoss-infer

A Blackwell-native inference runtime for **OpenAI gpt-oss-20b on one RTX 5090**, built to be made faster by
open, benchmark-scored pull requests: the mechanism SparkInfer runs on Bittensor SN74, applied to a different
model and with a stricter trust model.

**Status: step 1 of the build plan.** The correctness anchor, the pinned weights and the compute-pool manifest
exist. The C++/CUDA runtime is next. Nothing here serves requests yet.

## Layout

| Path | State | What |
|---|---|---|
| [`reference/`](reference/) | **done** | fp32 reference forward, golden next-token distributions, and the gate every engine change is judged by. 25 CPU tests, cross-checked against transformers. |
| [`docker/manifest.yaml`](docker/manifest.yaml) | draft | The Gittensor compute-pool manifest: 9 pinned artifacts and 2 tool-call canaries; it validates against the pool's schema. The image digest and measured numbers are placeholders. |
| `runtime/`, `kernels/`, `server/` | next | SparkInfer's infrastructure (paged KV, scheduler, prefix cache, OpenAI-compatible server), plus a new gpt-oss model class |
| `eval/` | later | The PR-scoring bot, built after the runtime exists |

## The target

**Model.** `openai/gpt-oss-20b` at commit `6cee5e8`, unchanged:
- MXFP4 experts, 32 of them with 4 active per token;
- head_dim-64 attention with learned sinks, with 128-token sliding windows on alternate layers;
- YaRN up to 131k tokens;
- the harmony chat format, with tool calls.

**Hardware.** RTX 5090 (`sm_120`, built as `120a`/`120f`); DGX Spark (`sm_121`) later.

**Baselines to beat**, measured on the eval card before launch:
- llama.cpp: ≈412 tok/s single-stream decode and ≈20k tok/s prefill (published, 2026).
- vLLM and SGLang: serving throughput.
- An engine that keeps the shipped BF16 attention and LM head tops out at 483 tok/s even at full memory
  bandwidth. Beating llama.cpp at single-stream decode therefore needs 8-bit attention and head weights, gated
  on KL against the reference.

## Principles

- **Correct before fast.** An engine change passes the golden gate (top-1 agreement and KL against fp32
  distributions of the official checkpoint) before its speed counts.
- **Harmony is part of correctness.** Tool calls, channels and stop tokens (`<|return|>`, `<|call|>`; never
  `<|end|>`) are tested like kernels.
- **User text is never control tokens.** The server encodes untrusted text with special-token matching off and
  inserts harmony control tokens by id.
- **Maintainers own the measuring instruments.** `reference/`, `eval/`, `docker/manifest.yaml` and `.github/`
  can't change in a scored PR.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how scoring will work, and [NOTICE](NOTICE) for third-party material.
MIT licensed.
