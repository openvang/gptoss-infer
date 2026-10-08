#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "engine.h"
#include "gptoss/gptoss.h"
#include "kernels.cuh"

using gptoss::Engine;

struct gptoss_engine {
    Engine impl;
    gptoss_engine(const char* dir, int max_ctx) : impl(dir, max_ctx) {}
};

namespace {

thread_local std::string g_error;

template <class F>
int guard(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
    } catch (...) {
        g_error = "unknown error";
    }
    return -1;
}

void sync_check(const char* what) {
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Scratch device memory for the kernel test entry points.
template <class T>
struct DevBuf {
    T* p = nullptr;
    explicit DevBuf(size_t n) {
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    }
    ~DevBuf() { cudaFree(p); }
};

}  // namespace

extern "C" {

gptoss_engine* gptoss_create(const char* model_dir, int max_ctx) {
    gptoss_engine* e = nullptr;
    guard([&] { e = new gptoss_engine(model_dir, max_ctx); });
    return e;
}

void gptoss_destroy(gptoss_engine* e) { delete e; }
const char* gptoss_last_error(void) { return g_error.c_str(); }
int gptoss_vocab(const gptoss_engine* e) { return e->impl.config().vocab; }
int gptoss_position(const gptoss_engine* e) { return e->impl.position(); }
long long gptoss_device_bytes(const gptoss_engine* e) { return (long long)e->impl.device_bytes(); }

int gptoss_reset(gptoss_engine* e) { return guard([&] { e->impl.reset(); }); }
int gptoss_use_graph(gptoss_engine* e, int on) { return guard([&] { e->impl.use_graph(on != 0); }); }
int gptoss_step(gptoss_engine* e, int token) { return guard([&] { e->impl.step(token); }); }
int gptoss_logits(gptoss_engine* e, float* out) { return guard([&] { e->impl.copy_logits(out); }); }

int gptoss_score(gptoss_engine* e, const int32_t* ids, int k, int target, float* lp, int32_t* argmax,
                 float* target_lp, float* lse) {
    return guard([&] { e->impl.score(ids, k, target, lp, argmax, target_lp, lse); });
}

int gptoss_set_position(gptoss_engine* e, int pos) { return guard([&] { e->impl.set_position(pos); }); }
int gptoss_bench_steps(gptoss_engine* e, int token, int n) { return guard([&] { e->impl.bench_steps(token, n); }); }

int gptoss_kernel_gemv_bf16(const void* W, const float* x, const float* bias, const float* resid, float* y, int N,
                            int K) {
    return guard([&] {
        gptoss::launch_gemv_bf16(static_cast<const __nv_bfloat16*>(W), x, bias, resid, y, N, K, nullptr);
        sync_check("gemv_bf16");
    });
}

int gptoss_kernel_rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    return guard([&] {
        gptoss::launch_rmsnorm(x, w, y, n, eps, nullptr);
        sync_check("rmsnorm");
    });
}

int gptoss_kernel_rope_append(float* qkv, const float* cos_t, const float* sin_t, int pos, void* k_cache,
                              void* v_cache, int window) {
    return guard([&] {
        DevBuf<int> d_pos(1);
        if (cudaMemcpy(d_pos.p, &pos, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("copy pos");
        gptoss::launch_rope_append(qkv, cos_t, sin_t, d_pos.p, static_cast<gptoss::kv_t*>(k_cache),
                                   static_cast<gptoss::kv_t*>(v_cache), window, nullptr);
        sync_check("rope_append");
    });
}

int gptoss_kernel_attention(const float* q, const void* k_cache, const void* v_cache, const float* sinks, int pos,
                            int window, int max_ctx, float* out) {
    return guard([&] {
        DevBuf<int> d_pos(1);
        DevBuf<float> part(gptoss::attn_workspace_floats());
        if (cudaMemcpy(d_pos.p, &pos, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("copy pos");
        gptoss::launch_attention(q, static_cast<const gptoss::kv_t*>(k_cache),
                                 static_cast<const gptoss::kv_t*>(v_cache), sinks, d_pos.p, window, max_ctx, part.p,
                                 out, nullptr);
        sync_check("attention");
    });
}

int gptoss_kernel_router(const void* W, const float* b, const float* x, int E, int H, int k, int32_t* ids,
                         float* weights) {
    return guard([&] {
        gptoss::launch_router(static_cast<const __nv_bfloat16*>(W), b, x, E, H, k, ids, weights, nullptr);
        sync_check("router");
    });
}

int gptoss_kernel_moe(const uint8_t* gu_blocks, const uint8_t* gu_scales, const float* gu_bias,
                      const uint8_t* dn_blocks, const uint8_t* dn_scales, const float* dn_bias, const float* x_norm,
                      const int32_t* ids, const float* weights, float* x_resid, int k, int H, int I, float alpha,
                      float limit) {
    return guard([&] {
        DevBuf<float> a(size_t(k) * I), y(size_t(k) * H);
        gptoss::launch_moe_gateup(gu_blocks, gu_scales, gu_bias, x_norm, ids, a.p, k, H, I, alpha, limit, nullptr);
        gptoss::launch_moe_down(dn_blocks, dn_scales, dn_bias, a.p, ids, y.p, k, H, I, nullptr);
        gptoss::launch_moe_combine(x_resid, y.p, weights, k, H, nullptr);
        sync_check("moe");
    });
}

}  // extern "C"
