/* C API of the gptoss-infer runtime. Every function returns 0 on success and -1 on failure (see
 * gptoss_last_error), except where noted. Not thread-safe: use one engine per thread. */
#ifndef GPTOSS_H
#define GPTOSS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gptoss_engine gptoss_engine;

/* Loads the checkpoint in model_dir onto the current CUDA device; returns NULL on failure.
 * vram_budget_bytes > 0 caps the engine's own device allocations (weights, KV cache, workspaces): creation fails
 * if they would not fit. The evaluator separately enforces the whole process's measured peak. */
gptoss_engine* gptoss_create(const char* model_dir, int max_ctx, long long vram_budget_bytes);
void gptoss_destroy(gptoss_engine* e);
const char* gptoss_last_error(void);

int gptoss_vocab(const gptoss_engine* e);           /* returns the vocabulary size */
int gptoss_position(const gptoss_engine* e);        /* returns the number of cached tokens */
long long gptoss_device_bytes(const gptoss_engine* e);

int gptoss_reset(gptoss_engine* e);
int gptoss_use_graph(gptoss_engine* e, int on);
/* Appends one token; afterwards the logits predict the token at the next position. */
int gptoss_step(gptoss_engine* e, int token);
int gptoss_logits(gptoss_engine* e, float* out);    /* copies vocab floats */
/* Appends tokens[0..n) and returns when they are processed. If ids != NULL, also scores every position p < n-1
 * against tokens[p+1]: lp[p*k + i] = log p(ids[p*k + i]), argmax[p], target_lp[p]. Afterwards the logits predict
 * the token after tokens[n-1]. This is the entry point for batched prefill: any implementation must produce the
 * same distributions as n single steps (the evaluator gates both paths against the golden). */
int gptoss_prefill(gptoss_engine* e, const int32_t* tokens, int n, const int32_t* ids, int k, float* lp,
                   int32_t* argmax, float* target_lp);
/* Log-probabilities of ids[0..k), the argmax token, log p(target) (target < 0: not computed) and logsumexp. */
int gptoss_score(gptoss_engine* e, const int32_t* ids, int k, int target, float* lp, int32_t* argmax,
                 float* target_lp, float* lse);

/* Benchmarking only: pretend `pos` tokens are cached, then run n steps that all feed `token`. */
int gptoss_set_position(gptoss_engine* e, int pos);
int gptoss_bench_steps(gptoss_engine* e, int token, int n);

/* Kernel entry points for tests. Pointers are device pointers; each call synchronizes. */
int gptoss_kernel_gemv_bf16(const void* W, const float* x, const float* bias, const float* resid, float* y, int N,
                            int K);
int gptoss_kernel_rmsnorm(const float* x, const float* w, float* y, int n, float eps);
int gptoss_kernel_rope_append(float* qkv, const float* cos_t, const float* sin_t, int pos, void* k_cache,
                              void* v_cache, int window);
int gptoss_kernel_attention(const float* q, const void* k_cache, const void* v_cache, const float* sinks, int pos,
                            int window, int max_ctx, float* out);
int gptoss_kernel_router(const void* W, const float* b, const float* x, int E, int H, int k, int32_t* ids,
                         float* weights);
/* x_resid += sum_e w[e] * (W2[ids[e]] swiglu(W1[ids[e]] x_norm + b1) + b2) */
int gptoss_kernel_moe(const uint8_t* gu_blocks, const uint8_t* gu_scales, const float* gu_bias,
                      const uint8_t* dn_blocks, const uint8_t* dn_scales, const float* dn_bias, const float* x_norm,
                      const int32_t* ids, const float* weights, float* x_resid, int k, int H, int I, float alpha,
                      float limit);

#ifdef __cplusplus
}
#endif

#endif /* GPTOSS_H */
