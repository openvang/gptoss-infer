# gptoss-infer

A Blackwell-native inference runtime for **OpenAI gpt-oss-20b on one RTX 5090**, built to be made faster by
open, benchmark-scored pull requests: the mechanism SparkInfer runs on Bittensor SN74, applied to a different
model and with a stricter trust model.

**Status: baseline engine and contribution mechanism.**
- **Engine (M1):** a correct, single-sequence decode engine. It matches the fp32 reference to the precision of
  its FP16 KV cache.
- **Speed:** contributors make it fast. Every pull request is measured on an RTX 5090 and merged when it is
  faster and keeps accuracy (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- **Memory:** the engine must stay within 24 GiB, so speech-to-text and text-to-speech models can later share
  the card.

## Layout

| Path | State | What |
|---|---|---|
| [`reference/`](reference/) | **done** | fp32 reference forward, golden next-token distributions, and the gate every engine change is judged by. 25 CPU tests, cross-checked against transformers. |
| [`docker/manifest.yaml`](docker/manifest.yaml) | draft | The Gittensor compute-pool manifest: 9 pinned artifacts and 2 tool-call canaries; it validates against the pool's schema. The image digest and measured numbers are placeholders. |
| [`runtime/`](runtime/) | M1 | C++/CUDA decode engine and C API, with kernel tests, a golden scoring tool and a benchmark |
| [`eval/`](eval/) | v1 | the PR evaluation mechanism: isolated GPU runs, golden gates, the VRAM budget, paired timing, the bot that merges or closes |
| [`bench/`](bench/) | | llama.cpp baseline script and results; engine results by milestone |

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
