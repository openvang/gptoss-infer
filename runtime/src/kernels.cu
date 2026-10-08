#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels.cuh"

namespace gptoss {
namespace {

constexpr unsigned kFull = 0xffffffffu;
constexpr int kHeads = 64, kKvHeads = 8, kHeadDim = 64, kGroup = kHeads / kKvHeads, kPairs = kHeadDim / 2;

// E2M1 code -> value (codes 8..15 are the negatives of 0..7).
__constant__ float kFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float h_lo(uint32_t w) { return __half2float(__ushort_as_half((unsigned short)(w & 0xffffu))); }
__device__ __forceinline__ float h_hi(uint32_t w) { return __half2float(__ushort_as_half((unsigned short)(w >> 16))); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
    return v;
}

// E8M0 scale byte -> 2^(s-127), exact (s == 0 is the fp32 subnormal 2^-127; s == 255 is rejected at load).
__device__ __forceinline__ float e8m0(uint32_t s) {
    return s == 0 ? __uint_as_float(0x00400000u) : __uint_as_float(s << 23);
}

// One MXFP4 block (16 bytes = 32 values, low nibble first) dotted with 32 activations stored at stride 1 in
// a padded shared-memory row, times the block scale.
__device__ __forceinline__ float mxfp4_dot32(const uint8_t* blk, uint32_t scale, const float* x, const float* lut) {
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(blk));
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
    float s = 0.f;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (w[q] >> (8 * b)) & 0xffu;
            const int i = (q * 4 + b) * 2;
            s += lut[byte & 15u] * x[i] + lut[byte >> 4] * x[i + 1];
        }
    }
    return s * e8m0(scale);
}

__global__ void embed_kernel(const __nv_bfloat16* table, const int* d_token, float* x, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) x[i] = __bfloat162float(table[(size_t)(*d_token) * H + i]);
}

__global__ void rmsnorm_kernel(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float red[32];
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    const float r = 1.0f / sqrtf(red[0] / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * r * w[i];
}

constexpr int kGemvRowsPerWarp = 4;

__global__ void __launch_bounds__(256) gemv_bf16_kernel(const __nv_bfloat16* __restrict__ W, const float* __restrict__ x,
                                                        const float* __restrict__ bias, const float* resid, float* y,
                                                        int N, int K) {
    extern __shared__ float xs[];
    for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * (blockDim.x >> 5) + warp) * kGemvRowsPerWarp;
    if (row0 >= N) return;
    const int K8 = K >> 3;
    const uint4* Wv = reinterpret_cast<const uint4*>(W);
    float acc[kGemvRowsPerWarp] = {};
    for (int c = lane; c < K8; c += 32) {
        const float4 xa = *reinterpret_cast<const float4*>(xs + c * 8);
        const float4 xb = *reinterpret_cast<const float4*>(xs + c * 8 + 4);
#pragma unroll
        for (int r = 0; r < kGemvRowsPerWarp; ++r) {
            if (row0 + r < N) {
                const uint4 u = __ldg(Wv + (size_t)(row0 + r) * K8 + c);
                acc[r] += bf_lo(u.x) * xa.x + bf_hi(u.x) * xa.y + bf_lo(u.y) * xa.z + bf_hi(u.y) * xa.w +
                          bf_lo(u.z) * xb.x + bf_hi(u.z) * xb.y + bf_lo(u.w) * xb.z + bf_hi(u.w) * xb.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) {
        float v = warp_sum(acc[r]);
        const int row = row0 + r;
        if (lane == 0 && row < N) {
            if (bias) v += bias[row];
            if (resid) v += resid[row];
            y[row] = v;
        }
    }
}

__global__ void rope_append_kernel(float* qkv, const float* cos_t, const float* sin_t, const int* d_pos,
                                   kv_t* kc, kv_t* vc, int window) {
    const int pos = *d_pos;
    const int head = blockIdx.x;      // 0..63 query heads, 64..71 key heads
    const int i = threadIdx.x;        // rotary pair
    const float c = cos_t[(size_t)pos * kPairs + i], sn = sin_t[(size_t)pos * kPairs + i];
    float* base = head < kHeads ? qkv + head * kHeadDim : qkv + kHeads * kHeadDim + (head - kHeads) * kHeadDim;
    const float x1 = base[i], x2 = base[i + kPairs];
    const float o1 = x1 * c - x2 * sn, o2 = x2 * c + x1 * sn;   // NeoX half-split rotation
    base[i] = o1;
    base[i + kPairs] = o2;
    if (head >= kHeads) {
        const int kh = head - kHeads;
        const size_t slot = window > 0 ? size_t(pos % window) : size_t(pos);
        const size_t off = (slot * kKvHeads + kh) * kHeadDim;
        const float* v = qkv + (kHeads + kKvHeads) * kHeadDim + kh * kHeadDim;
        kc[off + i] = __float2half_rn(o1);
        kc[off + i + kPairs] = __float2half_rn(o2);
        vc[off + i] = __float2half_rn(v[i]);
        vc[off + i + kPairs] = __float2half_rn(v[i + kPairs]);
    }
}

// Visible keys for the query at `pos`: positions first .. pos. Window layers see `window` keys including the
// query's own; full layers see all. Keys are split into `splits` contiguous ranges of `split_len`.
__device__ __forceinline__ void attn_range(int pos, int window, int& n_keys, int& first, int& splits, int& split_len) {
    n_keys = window > 0 ? min(pos + 1, window) : pos + 1;
    first = pos + 1 - n_keys;
    splits = min(kAttnMaxSplits, (n_keys + kAttnMinKeysPerSplit - 1) / kAttnMinKeysPerSplit);
    split_len = (n_keys + splits - 1) / splits;
}

constexpr size_t kPartStats = size_t(kKvHeads) * kAttnMaxSplits * kGroup;   // floats per stat array

// One block per (KV head, split): scores for the 8 query heads of the group, softmax stats, and the
// unnormalized value sum. Workspace layout: m[kPartStats], l[kPartStats], acc[kPartStats * 64].
__global__ void __launch_bounds__(128) attn_partial_kernel(const float* __restrict__ q,
                                                           const kv_t* __restrict__ kc,
                                                           const kv_t* __restrict__ vc, const int* d_pos,
                                                           int window, int max_split_len, float* part) {
    extern __shared__ float sc[];                        // [kGroup][max_split_len]
    __shared__ float qs[kGroup][kHeadDim];
    __shared__ float red[4][kGroup][kHeadDim];
    int n_keys, first, splits, split_len;
    attn_range(*d_pos, window, n_keys, first, splits, split_len);
    const int kvh = blockIdx.x, split = blockIdx.y;
    if (split >= splits) return;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int k0 = split * split_len, len = min(n_keys, k0 + split_len) - k0;
    float* pm = part;
    float* pl = part + kPartStats;
    float* pacc = part + 2 * kPartStats;
    const size_t pidx = (size_t(kvh) * kAttnMaxSplits + split) * kGroup;
    if (len <= 0) {                                      // empty trailing split: neutral partials
        if (tid < kGroup) { pm[pidx + tid] = -INFINITY; pl[pidx + tid] = 0.f; }
        for (int i = tid; i < kGroup * kHeadDim; i += blockDim.x) pacc[pidx * kHeadDim + i] = 0.f;
        return;
    }
    for (int i = tid; i < kGroup * kHeadDim; i += blockDim.x) qs[i / kHeadDim][i % kHeadDim] = q[kvh * kGroup * kHeadDim + i];
    __syncthreads();

    // Scores: 8 lanes per key (8 dims each), 4 keys per warp per step; loop bounds are warp-uniform.
    const int sub = lane & 7, kslot = lane >> 3;
    for (int base = warp * 4; base < len; base += 16) {
        const int kk = base + kslot;
        const bool valid = kk < len;
        float kv[8] = {};
        if (valid) {
            const int kpos = first + k0 + kk;
            const size_t slot = window > 0 ? size_t(kpos % window) : size_t(kpos);
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(kc + (slot * kKvHeads + kvh) * kHeadDim) + sub);
            kv[0] = h_lo(u.x); kv[1] = h_hi(u.x); kv[2] = h_lo(u.y); kv[3] = h_hi(u.y);
            kv[4] = h_lo(u.z); kv[5] = h_hi(u.z); kv[6] = h_lo(u.w); kv[7] = h_hi(u.w);
        }
        float s[kGroup];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            float t = 0.f;
#pragma unroll
            for (int d = 0; d < 8; ++d) t += qs[h][sub * 8 + d] * kv[d];
            s[h] = t;
        }
#pragma unroll
        for (int h = 0; h < kGroup; ++h)
            for (int o = 4; o > 0; o >>= 1) s[h] += __shfl_xor_sync(kFull, s[h], o);
        if (valid && sub == 0)
#pragma unroll
            for (int h = 0; h < kGroup; ++h) sc[h * max_split_len + kk] = s[h] * 0.125f;   // 1/sqrt(64)
    }
    __syncthreads();

    for (int h = warp; h < kGroup; h += 4) {
        float m = -INFINITY;
        for (int i = lane; i < len; i += 32) m = fmaxf(m, sc[h * max_split_len + i]);
        m = warp_max(m);
        float l = 0.f;
        for (int i = lane; i < len; i += 32) {
            const float p = expf(sc[h * max_split_len + i] - m);
            sc[h * max_split_len + i] = p;
            l += p;
        }
        l = warp_sum(l);
        if (lane == 0) { pm[pidx + h] = m; pl[pidx + h] = l; }
    }
    __syncthreads();

    // Values: warp w takes keys w, w+4, ...; lane holds dims 2*lane and 2*lane+1 for all 8 heads.
    float acc[kGroup][2] = {};
    for (int kk = warp; kk < len; kk += 4) {
        const int kpos = first + k0 + kk;
        const size_t slot = window > 0 ? size_t(kpos % window) : size_t(kpos);
        const uint32_t w2 = __ldg(reinterpret_cast<const uint32_t*>(vc + (slot * kKvHeads + kvh) * kHeadDim) + lane);
        const float v0 = h_lo(w2), v1 = h_hi(w2);
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float p = sc[h * max_split_len + kk];
            acc[h][0] += p * v0;
            acc[h][1] += p * v1;
        }
    }
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        red[warp][h][2 * lane] = acc[h][0];
        red[warp][h][2 * lane + 1] = acc[h][1];
    }
    __syncthreads();
    for (int i = tid; i < kGroup * kHeadDim; i += blockDim.x) {
        const int h = i / kHeadDim, d = i % kHeadDim;
        pacc[(pidx + h) * kHeadDim + d] = red[0][h][d] + red[1][h][d] + red[2][h][d] + red[3][h][d];
    }
}

// One block per query head: merge the splits and the learned sink. The sink joins the softmax denominator
// once per head and contributes no value.
__global__ void attn_combine_kernel(const float* part, const float* sinks, const int* d_pos, int window, float* out) {
    int n_keys, first, splits, split_len;
    attn_range(*d_pos, window, n_keys, first, splits, split_len);
    const int h = blockIdx.x, kvh = h / kGroup, hh = h % kGroup, d = threadIdx.x;
    const float* pm = part;
    const float* pl = part + kPartStats;
    const float* pacc = part + 2 * kPartStats;
    const float sink = sinks[h];
    float M = sink;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, pm[(size_t(kvh) * kAttnMaxSplits + s) * kGroup + hh]);
    float L = expf(sink - M), acc = 0.f;
    for (int s = 0; s < splits; ++s) {
        const size_t pi = (size_t(kvh) * kAttnMaxSplits + s) * kGroup + hh;
        const float w = expf(pm[pi] - M);
        L += pl[pi] * w;
        acc += pacc[pi * kHeadDim + d] * w;
    }
    out[h * kHeadDim + d] = acc / L;
}

constexpr int kRouterThreads = 256;

// Block e computes logit e; the last block to finish selects the top k with one warp (the __threadfence() pattern
// of the CUDA Programming Guide, "Memory Fence Functions"). *done is 0 between launches: the last block resets it.
__global__ void __launch_bounds__(kRouterThreads) router_kernel(const __nv_bfloat16* __restrict__ W,
                                                                const float* __restrict__ b,
                                                                const float* __restrict__ x, int E, int H, int k,
                                                                float* logits, unsigned* done, int* ids,
                                                                float* weights) {
    __shared__ float red[kRouterThreads / 32];
    __shared__ bool last;
    const int e = blockIdx.x, H8 = H >> 3, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint4* Wv = reinterpret_cast<const uint4*>(W) + size_t(e) * H8;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float acc = 0.f;
#pragma unroll 2
    for (int c = threadIdx.x; c < H8; c += kRouterThreads) {
        const uint4 u = __ldg(Wv + c);
        const float4 xa = __ldg(x4 + 2 * c), xb = __ldg(x4 + 2 * c + 1);
        acc += bf_lo(u.x) * xa.x + bf_hi(u.x) * xa.y + bf_lo(u.y) * xa.z + bf_hi(u.y) * xa.w +
               bf_lo(u.z) * xb.x + bf_hi(u.z) * xb.y + bf_lo(u.w) * xb.z + bf_hi(u.w) * xb.w;
    }
    acc = warp_sum(acc);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.f;
        for (int w = 0; w < kRouterThreads / 32; ++w) t += red[w];
        logits[e] = t + b[e];
        __threadfence();
        last = atomicAdd(done, 1u) == unsigned(E - 1);
    }
    __syncthreads();
    if (!last || warp) return;

    // Lane i holds expert i. Ids >= 32 mark lanes that can no longer be selected; NaN never wins, so the order is
    // total and every lane agrees on each pick. Ties go to the lower index.
    float v = lane < E ? reinterpret_cast<volatile float*>(logits)[lane] : -INFINITY;
    if (v != v) v = -INFINITY;
    int id = lane < E ? lane : 32 + lane;
    float sel = -INFINITY;
    int sel_id = 0;
    for (int j = 0; j < k; ++j) {
        float best = v;
        int besti = id;
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(kFull, best, o);
            const int oi = __shfl_xor_sync(kFull, besti, o);
            if (ob > best || (ob == best && oi < besti)) { best = ob; besti = oi; }
        }
        if (lane == j) { sel = best; sel_id = besti; }
        if (id == besti) { v = -INFINITY; id = 32 + lane; }
    }
    // Softmax over the k selected logits; lane 0 holds the largest.
    const float top = __shfl_sync(kFull, sel, 0);
    const float p = lane < k ? expf(sel - top) : 0.f;
    const float sum = warp_sum(p);
    if (lane < k) { ids[lane] = sel_id; weights[lane] = p / sum; }
    if (lane == 0) *done = 0;
}

// Activations for the MXFP4 kernels live in shared memory as rows of 32 values padded to 33, so the 32 lanes
// (each on a different block) read 32 different banks.
__device__ __forceinline__ void load_padded(float* dst, const float* src, int n) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) dst[(i >> 5) * 33 + (i & 31)] = src[i];
}

__global__ void __launch_bounds__(256) moe_gateup_kernel(const uint8_t* __restrict__ blocks,
                                                         const uint8_t* __restrict__ scales,
                                                         const float* __restrict__ bias, const float* __restrict__ x,
                                                         const int* __restrict__ ids, float* a, int H, int I,
                                                         float alpha, float limit) {
    extern __shared__ float xs[];                        // (H/32) rows of 33
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = kFp4[threadIdx.x];
    load_padded(xs, x, H);
    __syncthreads();
    const int slot = blockIdx.y, e = ids[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int j = blockIdx.x * 8 + warp;
    if (j >= I) return;
    const int G = H >> 5;
    const size_t rg = size_t(e) * 2 * I + 2 * size_t(j);   // glu row; the linear row is rg + 1
    float acc_g = 0.f, acc_l = 0.f;
    for (int g = lane; g < G; g += 32) {
        acc_g += mxfp4_dot32(blocks + (rg * G + g) * 16, scales[rg * G + g], xs + g * 33, lut);
        acc_l += mxfp4_dot32(blocks + ((rg + 1) * G + g) * 16, scales[(rg + 1) * G + g], xs + g * 33, lut);
    }
    acc_g = warp_sum(acc_g);
    acc_l = warp_sum(acc_l);
    if (lane == 0) {
        float gv = acc_g + bias[rg], lv = acc_l + bias[rg + 1];
        gv = fminf(gv, limit);                           // glu clamped from above only
        lv = fminf(fmaxf(lv, -limit), limit);
        a[size_t(slot) * I + j] = gv / (1.f + expf(-alpha * gv)) * (lv + 1.f);
    }
}

__global__ void __launch_bounds__(256) moe_down_kernel(const uint8_t* __restrict__ blocks,
                                                       const uint8_t* __restrict__ scales,
                                                       const float* __restrict__ bias, const float* __restrict__ a,
                                                       const int* __restrict__ ids, float* y, int H, int I) {
    extern __shared__ float as[];
    __shared__ float lut[16];
    const int slot = blockIdx.y, e = ids[slot];
    if (threadIdx.x < 16) lut[threadIdx.x] = kFp4[threadIdx.x];
    load_padded(as, a + size_t(slot) * I, I);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + warp;
    if (row >= H) return;
    const int G = I >> 5;
    const size_t r = size_t(e) * H + row;
    float acc = 0.f;
    for (int g = lane; g < G; g += 32) acc += mxfp4_dot32(blocks + (r * G + g) * 16, scales[r * G + g], as + g * 33, lut);
    acc = warp_sum(acc);
    if (lane == 0) y[size_t(slot) * H + row] = acc + bias[r];
}

__global__ void moe_combine_kernel(float* x, const float* y, const float* weights, int k, int H) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    float add = 0.f;
    for (int e = 0; e < k; ++e) add += weights[e] * y[size_t(e) * H + i];
    x[i] += add;
}

__global__ void increment_kernel(int* p) { *p += 1; }

constexpr int kScoreBlocks = 256;

__global__ void score_partial_kernel(const float* logits, int V, float* work) {
    __shared__ float sm[32], ss[32], sv[32];
    __shared__ int si[32];
    const int per = (V + kScoreBlocks - 1) / kScoreBlocks;
    const int b0 = blockIdx.x * per, b1 = min(V, b0 + per);
    float m = -INFINITY, best = -INFINITY;
    int besti = INT_MAX;
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) {
        const float v = logits[i];
        m = fmaxf(m, v);
        if (v > best) { best = v; besti = i; }
    }
    float s = 0.f;
    // Block-wide max and argmax first, then the sum of exp(logit - max) over the chunk.
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(kFull, best, o);
        const int oi = __shfl_xor_sync(kFull, besti, o);
        if (ob > best || (ob == best && oi < besti)) { best = ob; besti = oi; }
    }
    m = warp_max(m);
    if (lane == 0) { sm[warp] = m; sv[warp] = best; si[warp] = besti; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float M = -INFINITY, B = -INFINITY;
        int BI = INT_MAX;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) {
            M = fmaxf(M, sm[w]);
            if (sv[w] > B || (sv[w] == B && si[w] < BI)) { B = sv[w]; BI = si[w]; }
        }
        sm[0] = M; sv[0] = B; si[0] = BI;
    }
    __syncthreads();
    const float M = sm[0];
    for (int i = b0 + threadIdx.x; i < b1; i += blockDim.x) s += expf(logits[i] - M);
    s = warp_sum(s);
    if (lane == 0) ss[warp] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float S = 0.f;
        for (int w = 0; w < int(blockDim.x >> 5); ++w) S += ss[w];
        work[blockIdx.x * 4 + 0] = M;
        work[blockIdx.x * 4 + 1] = S;
        work[blockIdx.x * 4 + 2] = sv[0];
        work[blockIdx.x * 4 + 3] = __int_as_float(si[0]);
    }
}

__global__ void score_final_kernel(const float* logits, const float* work, const int* ids, int k, int target,
                                   float* lp_out, int* argmax_out, float* target_lp_out, float* lse_out) {
    __shared__ float lse_s;
    if (threadIdx.x == 0) {
        float M = -INFINITY, B = -INFINITY;
        int BI = INT_MAX;
        for (int b = 0; b < kScoreBlocks; ++b) {
            M = fmaxf(M, work[b * 4]);
            const float bv = work[b * 4 + 2];
            const int bi = __float_as_int(work[b * 4 + 3]);
            if (bv > B || (bv == B && bi < BI)) { B = bv; BI = bi; }
        }
        float S = 0.f;
        for (int b = 0; b < kScoreBlocks; ++b) S += work[b * 4 + 1] * expf(work[b * 4] - M);
        lse_s = M + logf(S);
        *argmax_out = BI;
        *lse_out = lse_s;
        if (target >= 0) *target_lp_out = logits[target] - lse_s;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < k; i += blockDim.x) lp_out[i] = logits[ids[i]] - lse_s;
}

}  // namespace

static void check_launch(const char* what) {
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void launch_embed(const __nv_bfloat16* table, const int* d_token, float* x, int H, cudaStream_t s) {
    embed_kernel<<<(H + 255) / 256, 256, 0, s>>>(table, d_token, x, H);
    check_launch("embed");
}

void launch_rmsnorm(const float* x, const float* w, float* y, int n, float eps, cudaStream_t s) {
    rmsnorm_kernel<<<1, 1024, 0, s>>>(x, w, y, n, eps);
    check_launch("rmsnorm");
}

void launch_gemv_bf16(const __nv_bfloat16* W, const float* x, const float* bias, const float* resid, float* y,
                      int N, int K, cudaStream_t s) {
    if (K % 8) throw std::runtime_error("gemv_bf16: K must be a multiple of 8");
    const int rows_per_block = 8 * kGemvRowsPerWarp;
    gemv_bf16_kernel<<<(N + rows_per_block - 1) / rows_per_block, 256, K * sizeof(float), s>>>(W, x, bias, resid, y, N, K);
    check_launch("gemv_bf16");
}

void launch_rope_append(float* qkv, const float* cos_t, const float* sin_t, const int* d_pos,
                        kv_t* k_cache, kv_t* v_cache, int window, cudaStream_t s) {
    rope_append_kernel<<<kHeads + kKvHeads, kPairs, 0, s>>>(qkv, cos_t, sin_t, d_pos, k_cache, v_cache, window);
    check_launch("rope_append");
}

size_t attn_workspace_floats() { return 2 * kPartStats + kPartStats * kHeadDim; }

void launch_attention(const float* q, const kv_t* k_cache, const kv_t* v_cache, const float* sinks,
                      const int* d_pos, int window, int max_ctx, float* part, float* out, cudaStream_t s) {
    // Longest split any position up to max_ctx can produce: 64 keys below 128 splits, ceil(n/128) above.
    const int keys_max = window > 0 ? window : max_ctx;
    const int max_split_len = std::max(kAttnMinKeysPerSplit, (keys_max + kAttnMaxSplits - 1) / kAttnMaxSplits);
    const size_t smem = size_t(kGroup) * max_split_len * sizeof(float);
    attn_partial_kernel<<<dim3(kKvHeads, kAttnMaxSplits), 128, smem, s>>>(q, k_cache, v_cache, d_pos, window,
                                                                         max_split_len, part);
    check_launch("attn_partial");
    attn_combine_kernel<<<kHeads, kHeadDim, 0, s>>>(part, sinks, d_pos, window, out);
    check_launch("attn_combine");
}

void launch_router(const __nv_bfloat16* W, const float* b, const float* x, int E, int H, int k, float* logits,
                   unsigned* done, int* ids, float* weights, cudaStream_t s) {
    if (E < 1 || E > 32 || k < 1 || k > 8 || k > E || H % 8) throw std::runtime_error("router: unsupported shape");
    router_kernel<<<E, kRouterThreads, 0, s>>>(W, b, x, E, H, k, logits, done, ids, weights);
    check_launch("router");
}

void launch_moe_gateup(const uint8_t* blocks, const uint8_t* scales, const float* bias, const float* x,
                       const int* ids, float* a, int k, int H, int I, float alpha, float limit, cudaStream_t s) {
    if (H % 32) throw std::runtime_error("moe_gateup: H must be a multiple of 32");
    moe_gateup_kernel<<<dim3((I + 7) / 8, k), 256, (H / 32) * 33 * sizeof(float), s>>>(blocks, scales, bias, x, ids,
                                                                                         a, H, I, alpha, limit);
    check_launch("moe_gateup");
}

void launch_moe_down(const uint8_t* blocks, const uint8_t* scales, const float* bias, const float* a,
                     const int* ids, float* y, int k, int H, int I, cudaStream_t s) {
    if (I % 32) throw std::runtime_error("moe_down: I must be a multiple of 32");
    moe_down_kernel<<<dim3((H + 7) / 8, k), 256, (I / 32) * 33 * sizeof(float), s>>>(blocks, scales, bias, a, ids, y,
                                                                                       H, I);
    check_launch("moe_down");
}

void launch_moe_combine(float* x, const float* y, const float* weights, int k, int H, cudaStream_t s) {
    moe_combine_kernel<<<(H + 255) / 256, 256, 0, s>>>(x, y, weights, k, H);
    check_launch("moe_combine");
}

void launch_increment(int* d_pos, cudaStream_t s) {
    increment_kernel<<<1, 1, 0, s>>>(d_pos);
    check_launch("increment");
}

size_t score_workspace_floats() { return size_t(kScoreBlocks) * 4; }

void launch_score(const float* logits, int V, const int* ids, int k, int target, float* work, float* lp_out,
                  int* argmax_out, float* target_lp_out, float* lse_out, cudaStream_t s) {
    score_partial_kernel<<<kScoreBlocks, 256, 0, s>>>(logits, V, work);
    check_launch("score_partial");
    score_final_kernel<<<1, 64, 0, s>>>(logits, work, ids, k, target, lp_out, argmax_out, target_lp_out, lse_out);
    check_launch("score_final");
}

}  // namespace gptoss
