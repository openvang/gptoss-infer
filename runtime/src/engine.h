// Single-sequence decode engine for gpt-oss (batch 1). Weights stay in checkpoint formats on the GPU.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace gptoss {

struct Config {
    int layers = 0, hidden = 0, intermediate = 0, vocab = 0, experts = 0, top_k = 0, window = 0;
    int heads = 0, kv_heads = 0, head_dim = 0, max_positions = 0;
    float eps = 0, swiglu_alpha = 1.702f, swiglu_limit = 0;
    double rope_theta = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0;
    int rope_original = 0;
    bool rope_truncate = true;
    std::vector<bool> sliding;     // per layer
    std::vector<int> eos;
    static Config load(const std::string& model_dir);
};

struct Layer {
    __nv_bfloat16 *wqkv = nullptr, *wo = nullptr, *wr = nullptr;
    __half *kc = nullptr, *vc = nullptr;      // KV cache (kv_t)
    float *bqkv = nullptr, *bo = nullptr, *br = nullptr, *sinks = nullptr, *ln1 = nullptr, *ln2 = nullptr;
    uint8_t *gu_blocks = nullptr, *gu_scales = nullptr, *dn_blocks = nullptr, *dn_scales = nullptr;
    float *gu_bias = nullptr, *dn_bias = nullptr;
    int window = 0;
};

class Engine {
public:
    Engine(const std::string& model_dir, int max_ctx, size_t vram_budget = 0);   // 0: no budget
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const Config& config() const { return cfg_; }
    int max_ctx() const { return max_ctx_; }
    int position() const { return pos_; }
    size_t device_bytes() const { return bytes_; }

    void reset();                         // empty context
    void set_position(int pos);           // benchmarking only: treat `pos` tokens as cached (contents unspecified)
    void use_graph(bool on);              // replay one captured CUDA graph per step
    void step(int token);                 // appends `token`; logits for the next position are then available
    // Appends tokens[0..n); optionally scores positions 0..n-2 (see gptoss_prefill). Synchronizes.
    void prefill(const int* tokens, int n, const int* ids, int k, float* lp, int* argmax, float* target_lp);
    void bench_steps(int token, int n);   // benchmarking: n steps feeding the same token, no host round trips
    void copy_logits(float* host);        // synchronizes
    // lp[i] = log p(ids[i]); argmax of the logits; log p(target) (target < 0: skipped); logsumexp. Synchronizes.
    void score(const int* ids, int k, int target, float* lp, int* argmax, float* target_lp, float* lse);

private:
    void forward(cudaStream_t s);
    void capture();
    void* raw_alloc(size_t bytes);
    template <class T>
    T* alloc(size_t n) { return static_cast<T*>(raw_alloc(n * sizeof(T))); }

    Config cfg_;
    int max_ctx_ = 0, pos_ = 0;
    size_t bytes_ = 0, budget_ = 0;
    std::vector<void*> allocs_;
    std::vector<Layer> layers_;
    __nv_bfloat16 *embed_ = nullptr, *lm_head_ = nullptr;
    float *norm_ = nullptr, *cos_ = nullptr, *sin_ = nullptr;
    float *x_ = nullptr, *h_ = nullptr, *qkv_ = nullptr, *attn_ = nullptr, *part_ = nullptr, *a_ = nullptr;
    float *y_ = nullptr, *gate_w_ = nullptr, *logits_ = nullptr, *score_work_ = nullptr, *score_f_ = nullptr;
    float* router_logits_ = nullptr;
    unsigned* router_done_ = nullptr;
    int *ids_ = nullptr, *token_ = nullptr, *pos_d_ = nullptr, *score_ids_ = nullptr, *score_i_ = nullptr;
    int score_cap_ = 1024;
    cudaStream_t stream_ = nullptr;
    cudaGraphExec_t graph_ = nullptr;
    bool graph_on_ = false;
};

}  // namespace gptoss
