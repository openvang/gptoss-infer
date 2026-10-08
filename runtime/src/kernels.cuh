// Batch-1 decode kernels for gpt-oss. All activations are fp32; weights stay in their checkpoint formats
// (bf16, MXFP4 blocks + E8M0 scales). Every launcher takes the stream to run on.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace gptoss {

// KV-cache element type. fp16, not bf16: same bytes, 3 more mantissa bits; on golden_v1 a bf16 cache costs
// ~2.5x the KL of fp16, and gpt-oss's K/V stay far inside fp16 range (max |K| 187, |V| 62).
using kv_t = __half;

// Attention decode splits the keys of one KV head into at most this many blocks.
constexpr int kAttnMaxSplits = 128;
constexpr int kAttnMinKeysPerSplit = 64;

void launch_embed(const __nv_bfloat16* table, const int* d_token, float* x, int H, cudaStream_t s);
void launch_rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t s);
// y[N] = W[N,K] (bf16) * x[K] (+ bias[N]) (+ resid[N]). bias/resid may be null; resid may alias y.
void launch_gemv_bf16(const __nv_bfloat16* W, const float* x, const float* bias, const float* resid, float* y,
                      int N, int K, cudaStream_t s);
// Rotate q (64 heads) and k (8 heads) in qkv in place at position *d_pos, then write k and v (fp16) into the
// layer's cache. window > 0: ring buffer of `window` slots; window == 0: slot = position.
void launch_rope_append(float* qkv, const float* cos_t, const float* sin_t, const int* d_pos,
                        kv_t* k_cache, kv_t* v_cache, int window, cudaStream_t s);
// Attention for the query at *d_pos over its visible keys, with learned sinks. Writes out[64*64].
// part_* are workspace buffers sized by attn_workspace_floats().
// max_ctx bounds the per-split key count, which sizes the kernel's shared memory.
void launch_attention(const float* q, const kv_t* k_cache, const kv_t* v_cache, const float* sinks,
                      const int* d_pos, int window, int max_ctx, float* part, float* out, cudaStream_t s);
size_t attn_workspace_floats();
// Router: logits = W[E,H] x + b; top-k (sorted, ties to the lower index); softmax over the k selected logits.
// logits[E] and done[1] are scratch owned by the caller; *done must be 0 before the first launch and stays 0 after
// each one. Launches sharing the scratch must not run concurrently.
void launch_router(const __nv_bfloat16* W, const float* b, const float* x, int E, int H, int k, float* logits,
                   unsigned* done, int* ids, float* weights, cudaStream_t s);
// For each of k selected experts: a[e][j] = swiglu(W1[e] x + b1[e]) with W1 in MXFP4 [E][2I][H/32][16].
void launch_moe_gateup(const uint8_t* blocks, const uint8_t* scales, const float* bias, const float* x,
                       const int* ids, float* a, int k, int H, int I, float alpha, float limit, cudaStream_t s);
// y[e] = W2[e] a[e] + b2[e] with W2 in MXFP4 [E][H][I/32][16].
void launch_moe_down(const uint8_t* blocks, const uint8_t* scales, const float* bias, const float* a,
                     const int* ids, float* y, int k, int H, int I, cudaStream_t s);
// x[i] += sum_e weights[e] * y[e][i], experts summed in slot order.
void launch_moe_combine(float* x, const float* y, const float* weights, int k, int H, cudaStream_t s);
void launch_increment(int* d_pos, cudaStream_t s);
// Scoring support: lse and argmax of logits[V], then lp[i] = logits[ids[i]] - lse and lp of `target`.
void launch_score(const float* logits, int V, const int* ids, int k, int target, float* work, float* lp_out,
                  int* argmax_out, float* target_lp_out, float* lse_out, cudaStream_t s);
size_t score_workspace_floats();

}  // namespace gptoss
