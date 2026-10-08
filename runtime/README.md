# runtime/: the gptoss-infer engine

A native C++/CUDA engine for gpt-oss-20b on Blackwell (`sm_120a`). This is milestone M1: single-sequence
decode, with weights in their checkpoint formats.
- **Experts:** MXFP4 blocks.
- **Attention, router, embeddings and LM head:** BF16.
- **Activations:** FP32.
- **KV cache:** FP16.

```bash
cmake -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120a && cmake --build build -j
GPTOSS_LIB=build/libgptoss.so PYTHONPATH=reference python -m pytest -q runtime/tests     # kernel tests
PYTHONPATH=reference python runtime/tools/score_golden.py --lib build/libgptoss.so \
    --model-dir /path/to/gpt-oss-20b --golden reference/goldens/golden_v1 --out cand.safetensors
./build/gptoss-bench /path/to/gpt-oss-20b                                               # decode tok/s
```

## Layout

| File | What |
|---|---|
| `include/gptoss/gptoss.h` | C API: create, step, score and bench, plus kernel entry points for tests |
| `src/engine.{h,cu}` | config validation, weight upload (Q/K/V fused), YaRN table, the decode step, CUDA graph |
| `src/kernels.{cuh,cu}` | the kernels listed below |
| `src/safetensors.*`, `src/json.*` | mmap'd checkpoint reader with no dependencies |
| `tools/bench.cpp` | decode throughput at fixed depths, comparable to `llama-bench -n 128 -d …` |
| `tools/score_golden.py` | teacher-forces the golden corpus and gates the result with `reference/` |
| `tests/test_kernels.py` | each kernel against a float64 torch version of the reference semantics |

**Kernels:**
- BF16 GEMV with bias and residual;
- RMSNorm;
- YaRN RoPE with the KV append;
- split-K decode attention with learned sinks and exact 128-token windows;
- router (top-4, then softmax);
- MXFP4 expert gate/up with clamped SwiGLU, expert down, and combine;
- logsumexp, argmax and gather for scoring.

## M1 results (RTX 5090 capped at 525 W, 2026-10-08)

**Correctness against `golden_v1`** (`score_golden.py`):

| Positions | Top-1 | Mean KL | p99 KL |
|---|---:|---:|---:|
| Generated | 0.997 | 3.7e-4 | 0.0030 |
| All | 0.972 | 8.6e-3 | 0.157 |

That equals the fp32 reference run with an FP16 KV cache (`reference/scripts/reference_candidate.py --kv-dtype fp16`). The engine adds no error beyond its KV format.

**Decode, tg128 in tok/s:**

| Depth | gptoss-infer M1 | llama.cpp v0.6.0 | llama.cpp with `GGML_CUDA_GRAPH_OPT=1` |
|---:|---:|---:|---:|
| 0 | 283.8 | 366.0 | 403.5 |
| 4k | 270.5 | 358.7 | 375.5 |
| 16k | 241.1 | 334.3 | 346.9 |
| 32k | 217.0 | 309.5 | 310.3 |

**Why M1 is slower.** It reads 3.71 GB per token, against llama.cpp's 2.58 GB (Q8_0 attention and head). Both run at about 1.05 TB/s.

**Where the time goes** (per token at depth 0, from the profiler):

| Kernels | Time | Share | Bandwidth |
|---|---:|---:|---:|
| BF16 GEMVs | 1,651 µs | 48 % | 1.47 TB/s |
| MoE gate/up | 763 µs | 22 % | 1.11 TB/s |
| MoE down | 451 µs | 13 % | 0.94 TB/s |
| Router | 224 µs | 6.5 % | latency-bound |

**Next.**
- **Exact path:** fix the router's latency, raise MoE bandwidth and fuse the small kernels. The ceiling is roughly 400 tok/s.
- **Beating llama.cpp at single-stream decode** needs 8-bit attention and LM-head weights, gated on KL.
