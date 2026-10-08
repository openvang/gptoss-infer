#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.h"
#include "kernels.cuh"
#include "safetensors.h"

namespace gptoss {

#define CUDA_CHECK(x)                                                                                         \
    do {                                                                                                      \
        cudaError_t e_ = (x);                                                                                 \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string(#x) + ": " + cudaGetErrorString(e_));     \
    } while (0)

static std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("config: " + what);
}

Config Config::load(const std::string& dir) {
    const Json j = Json::parse(read_file(dir + "/config.json"));
    Config c;
    c.layers = int(j.at("num_hidden_layers").as_int());
    c.hidden = int(j.at("hidden_size").as_int());
    c.intermediate = int(j.at("intermediate_size").as_int());
    c.vocab = int(j.at("vocab_size").as_int());
    c.experts = int(j.at("num_local_experts").as_int());
    c.top_k = int(j.at("num_experts_per_tok").as_int());
    c.window = int(j.at("sliding_window").as_int());
    c.heads = int(j.at("num_attention_heads").as_int());
    c.kv_heads = int(j.at("num_key_value_heads").as_int());
    c.head_dim = int(j.at("head_dim").as_int());
    c.max_positions = int(j.at("max_position_embeddings").as_int());
    c.eps = float(j.at("rms_norm_eps").as_num());
    c.swiglu_limit = float(j.at("swiglu_limit").as_num());
    if (j.has("swiglu_alpha")) c.swiglu_alpha = float(j.at("swiglu_alpha").as_num());
    const Json& r = j.has("rope_scaling") ? j.at("rope_scaling") : j.at("rope_parameters");
    require(r.at("rope_type").as_str() == "yarn", "rope_type must be yarn");
    c.rope_theta = j.has("rope_theta") ? j.at("rope_theta").as_num() : r.at("rope_theta").as_num();
    c.rope_factor = r.at("factor").as_num();
    c.rope_original = int(r.at("original_max_position_embeddings").as_int());
    c.beta_fast = r.at("beta_fast").as_num();
    c.beta_slow = r.at("beta_slow").as_num();
    c.rope_truncate = r.at("truncate").as_bool();     // required: HF's default (true) is wrong for gpt-oss
    for (const auto& t : j.at("layer_types").arr) c.sliding.push_back(t.as_str() == "sliding_attention");
    const Json g = Json::parse(read_file(dir + "/generation_config.json"));
    const Json& eos = g.at("eos_token_id");
    if (eos.type == Json::Type::Array)
        for (const auto& e : eos.arr) c.eos.push_back(int(e.as_int()));
    else
        c.eos.push_back(int(eos.as_int()));

    // The decode kernels are specialised for gpt-oss's attention geometry.
    require(c.heads == 64 && c.kv_heads == 8 && c.head_dim == 64, "attention must be 64 heads / 8 kv heads / hd 64");
    require(int(c.sliding.size()) == c.layers, "layer_types length");
    require(c.hidden % 32 == 0 && c.intermediate % 32 == 0, "hidden and intermediate must be multiples of 32");
    require(c.experts <= 32 && c.top_k <= 8, "at most 32 experts and top-8");
    return c;
}

void* Engine::raw_alloc(size_t bytes) {
    if (budget_ && bytes_ + bytes > budget_)
        throw std::runtime_error("VRAM budget exceeded: need more than " + std::to_string((bytes_ + bytes) >> 20) +
                                 " MiB, budget " + std::to_string(budget_ >> 20) + " MiB");
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    allocs_.push_back(p);
    bytes_ += bytes;
    return p;
}

Engine::Engine(const std::string& dir, int max_ctx, size_t vram_budget) : cfg_(Config::load(dir)), budget_(vram_budget) {
    if (max_ctx <= 0 || max_ctx > cfg_.max_positions)
        throw std::runtime_error("max_ctx must be in 1.." + std::to_string(cfg_.max_positions));
    max_ctx_ = max_ctx;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    try {
        Checkpoint ck(dir);
        const int64_t H = cfg_.hidden, I = cfg_.intermediate, E = cfg_.experts, V = cfg_.vocab;
        const int64_t QD = int64_t(cfg_.heads) * cfg_.head_dim, KVD = int64_t(cfg_.kv_heads) * cfg_.head_dim;

        auto put = [&](void* dst, const void* src, size_t n) { CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice)); };
        auto bf16 = [&](const std::string& n, std::vector<int64_t> shape) {
            const TensorView& t = ck.get(n, "BF16", shape);
            auto* d = alloc<__nv_bfloat16>(size_t(t.numel()));
            put(d, t.data, t.nbytes);
            return d;
        };
        auto to_f32 = [](const TensorView& t, std::vector<float>& out) {   // exact: bf16 is the top half of fp32
            const uint16_t* s = reinterpret_cast<const uint16_t*>(t.data);
            const size_t n = size_t(t.numel()), o = out.size();
            out.resize(o + n);
            for (size_t i = 0; i < n; ++i) {
                const uint32_t w = uint32_t(s[i]) << 16;
                std::memcpy(&out[o + i], &w, 4);
            }
        };
        auto f32 = [&](const std::string& n, std::vector<int64_t> shape) {
            std::vector<float> h;
            to_f32(ck.get(n, "BF16", shape), h);
            auto* d = alloc<float>(h.size());
            put(d, h.data(), h.size() * sizeof(float));
            return d;
        };
        auto u8 = [&](const std::string& n, std::vector<int64_t> shape, bool scales) {
            const TensorView& t = ck.get(n, "U8", shape);
            if (scales && std::memchr(t.data, 0xFF, t.nbytes))
                throw std::runtime_error(n + " contains E8M0 0xFF (NaN)");
            auto* d = alloc<uint8_t>(t.nbytes);
            put(d, t.data, t.nbytes);
            return d;
        };

        embed_ = bf16("model.embed_tokens.weight", {V, H});
        lm_head_ = bf16("lm_head.weight", {V, H});
        norm_ = f32("model.norm.weight", {H});
        layers_.resize(cfg_.layers);
        for (int l = 0; l < cfg_.layers; ++l) {
            Layer& L = layers_[l];
            const std::string p = "model.layers." + std::to_string(l) + ".";
            L.window = cfg_.sliding[l] ? cfg_.window : 0;
            // Q, K and V are concatenated into one [QD + 2*KVD, H] matrix so one GEMV computes all three.
            L.wqkv = alloc<__nv_bfloat16>(size_t((QD + 2 * KVD) * H));
            const TensorView& q = ck.get(p + "self_attn.q_proj.weight", "BF16", {QD, H});
            const TensorView& k = ck.get(p + "self_attn.k_proj.weight", "BF16", {KVD, H});
            const TensorView& v = ck.get(p + "self_attn.v_proj.weight", "BF16", {KVD, H});
            put(L.wqkv, q.data, q.nbytes);
            put(L.wqkv + QD * H, k.data, k.nbytes);
            put(L.wqkv + (QD + KVD) * H, v.data, v.nbytes);
            std::vector<float> b;
            to_f32(ck.get(p + "self_attn.q_proj.bias", "BF16", {QD}), b);
            to_f32(ck.get(p + "self_attn.k_proj.bias", "BF16", {KVD}), b);
            to_f32(ck.get(p + "self_attn.v_proj.bias", "BF16", {KVD}), b);
            L.bqkv = alloc<float>(b.size());
            put(L.bqkv, b.data(), b.size() * sizeof(float));
            L.wo = bf16(p + "self_attn.o_proj.weight", {H, QD});
            L.bo = f32(p + "self_attn.o_proj.bias", {H});
            L.sinks = f32(p + "self_attn.sinks", {cfg_.heads});
            L.ln1 = f32(p + "input_layernorm.weight", {H});
            L.ln2 = f32(p + "post_attention_layernorm.weight", {H});
            L.wr = bf16(p + "mlp.router.weight", {E, H});
            L.br = f32(p + "mlp.router.bias", {E});
            L.gu_blocks = u8(p + "mlp.experts.gate_up_proj_blocks", {E, 2 * I, H / 32, 16}, false);
            L.gu_scales = u8(p + "mlp.experts.gate_up_proj_scales", {E, 2 * I, H / 32}, true);
            L.gu_bias = f32(p + "mlp.experts.gate_up_proj_bias", {E, 2 * I});
            L.dn_blocks = u8(p + "mlp.experts.down_proj_blocks", {E, H, I / 32, 16}, false);
            L.dn_scales = u8(p + "mlp.experts.down_proj_scales", {E, H, I / 32}, true);
            L.dn_bias = f32(p + "mlp.experts.down_proj_bias", {E, H});
            const size_t slots = L.window > 0 ? size_t(L.window) : size_t(max_ctx_);
            L.kc = alloc<kv_t>(slots * KVD);
            L.vc = alloc<kv_t>(slots * KVD);
        }

        // YaRN cos/sin table in float32, mirroring reference/gptoss_ref/rope.py operation for operation.
        const int d = cfg_.head_dim, half = d / 2;
        auto dim_for = [&](double rot) {
            return d * std::log(cfg_.rope_original / (rot * 2 * M_PI)) / (2 * std::log(cfg_.rope_theta));
        };
        double low = dim_for(cfg_.beta_fast), high = dim_for(cfg_.beta_slow);
        if (cfg_.rope_truncate) { low = std::floor(low); high = std::ceil(high); }
        low = std::max(low, 0.0);
        high = std::min(high, double(d - 1));
        if (high == low) high += 0.001;
        std::vector<float> inv(half);
        const float lowf = float(low), span = float(high - low), factor = float(cfg_.rope_factor);
        for (int i = 0; i < half; ++i) {
            const float fr = std::pow(float(cfg_.rope_theta), float(2 * i) / float(d));
            const float extrapolation = 1.0f / fr, interpolation = 1.0f / (factor * fr);
            const float ramp = std::min(std::max((float(i) - lowf) / span, 0.0f), 1.0f);
            const float keep = 1.0f - ramp;
            inv[i] = interpolation * (1.0f - keep) + extrapolation * keep;
        }
        const float conc = cfg_.rope_factor > 1 ? float(0.1 * std::log(cfg_.rope_factor) + 1.0) : 1.0f;
        std::vector<float> ct(size_t(max_ctx_) * half), st(size_t(max_ctx_) * half);
        for (int pos = 0; pos < max_ctx_; ++pos)
            for (int i = 0; i < half; ++i) {
                const float ang = float(pos) * inv[i];
                ct[size_t(pos) * half + i] = std::cos(ang) * conc;
                st[size_t(pos) * half + i] = std::sin(ang) * conc;
            }
        cos_ = alloc<float>(ct.size());
        sin_ = alloc<float>(st.size());
        put(cos_, ct.data(), ct.size() * sizeof(float));
        put(sin_, st.data(), st.size() * sizeof(float));

        x_ = alloc<float>(size_t(H));
        h_ = alloc<float>(size_t(H));
        qkv_ = alloc<float>(size_t(QD + 2 * KVD));
        attn_ = alloc<float>(size_t(QD));
        part_ = alloc<float>(attn_workspace_floats());
        a_ = alloc<float>(size_t(cfg_.top_k) * I);
        y_ = alloc<float>(size_t(cfg_.top_k) * H);
        gate_w_ = alloc<float>(size_t(cfg_.top_k));
        ids_ = alloc<int>(size_t(cfg_.top_k));
        router_logits_ = alloc<float>(size_t(E));
        router_done_ = alloc<unsigned>(1);
        CUDA_CHECK(cudaMemsetAsync(router_done_, 0, sizeof(unsigned), stream_));
        logits_ = alloc<float>(size_t(V));
        score_work_ = alloc<float>(score_workspace_floats());
        score_f_ = alloc<float>(size_t(score_cap_) + 2);
        score_ids_ = alloc<int>(size_t(score_cap_));
        score_i_ = alloc<int>(1);
        token_ = alloc<int>(1);
        pos_d_ = alloc<int>(1);
        reset();
    } catch (...) {
        for (void* p : allocs_) cudaFree(p);
        cudaStreamDestroy(stream_);
        throw;
    }
}

Engine::~Engine() {
    if (graph_) cudaGraphExecDestroy(graph_);
    for (void* p : allocs_) cudaFree(p);
    if (stream_) cudaStreamDestroy(stream_);
}

void Engine::reset() { set_position(0); }

void Engine::set_position(int pos) {
    if (pos < 0 || pos > max_ctx_) throw std::runtime_error("position out of range");
    pos_ = pos;
    CUDA_CHECK(cudaMemcpyAsync(pos_d_, &pos, sizeof(int), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::forward(cudaStream_t s) {
    const int H = cfg_.hidden, I = cfg_.intermediate;
    const int QD = cfg_.heads * cfg_.head_dim, KVD = cfg_.kv_heads * cfg_.head_dim;
    launch_embed(embed_, token_, x_, H, s);
    for (const Layer& L : layers_) {
        launch_rmsnorm(x_, L.ln1, h_, H, cfg_.eps, s);
        launch_gemv_bf16(L.wqkv, h_, L.bqkv, nullptr, qkv_, QD + 2 * KVD, H, s);
        launch_rope_append(qkv_, cos_, sin_, pos_d_, L.kc, L.vc, L.window, s);
        launch_attention(qkv_, L.kc, L.vc, L.sinks, pos_d_, L.window, max_ctx_, part_, attn_, s);
        launch_gemv_bf16(L.wo, attn_, L.bo, x_, x_, H, QD, s);                 // x += o_proj(attn) + b
        launch_rmsnorm(x_, L.ln2, h_, H, cfg_.eps, s);
        launch_router(L.wr, L.br, h_, cfg_.experts, H, cfg_.top_k, router_logits_, router_done_, ids_, gate_w_, s);
        launch_moe_gateup(L.gu_blocks, L.gu_scales, L.gu_bias, h_, ids_, a_, cfg_.top_k, H, I, cfg_.swiglu_alpha,
                          cfg_.swiglu_limit, s);
        launch_moe_down(L.dn_blocks, L.dn_scales, L.dn_bias, a_, ids_, y_, cfg_.top_k, H, I, s);
        launch_moe_combine(x_, y_, gate_w_, cfg_.top_k, H, s);
    }
    launch_rmsnorm(x_, norm_, h_, H, cfg_.eps, s);
    launch_gemv_bf16(lm_head_, h_, nullptr, nullptr, logits_, cfg_.vocab, H, s);
    launch_increment(pos_d_, s);
}

void Engine::capture() {
    if (graph_) { cudaGraphExecDestroy(graph_); graph_ = nullptr; }
    cudaGraph_t g = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    try {
        forward(stream_);
    } catch (...) {
        cudaStreamEndCapture(stream_, &g);
        if (g) cudaGraphDestroy(g);
        throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
    const cudaError_t e = cudaGraphInstantiate(&graph_, g, 0);
    cudaGraphDestroy(g);
    CUDA_CHECK(e);
}

void Engine::use_graph(bool on) { graph_on_ = on; }

void Engine::step(int token) {
    if (token < 0 || token >= cfg_.vocab) throw std::runtime_error("token id out of range");
    if (pos_ >= max_ctx_) throw std::runtime_error("context is full");
    CUDA_CHECK(cudaMemcpyAsync(token_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
    if (graph_on_) {
        if (!graph_) capture();
        CUDA_CHECK(cudaGraphLaunch(graph_, stream_));
    } else {
        forward(stream_);
    }
    ++pos_;
}

void Engine::prefill(const int* tokens, int n, const int* ids, int k, float* lp, int* argmax, float* target_lp) {
    if (n <= 0) throw std::runtime_error("prefill: n must be positive");
    if (pos_ + n > max_ctx_) throw std::runtime_error("context is full");
    // Reference implementation: n single-token steps. A batched prefill replaces this loop and must give the
    // same distributions (gated by the evaluator on both paths).
    for (int p = 0; p < n; ++p) {
        step(tokens[p]);
        if (ids && p + 1 < n)
            score(ids + size_t(p) * k, k, tokens[p + 1], lp + size_t(p) * k, argmax + p, target_lp + p, nullptr);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::bench_steps(int token, int n) {
    if (token < 0 || token >= cfg_.vocab) throw std::runtime_error("token id out of range");
    if (pos_ + n > max_ctx_) throw std::runtime_error("context is full");
    CUDA_CHECK(cudaMemcpyAsync(token_, &token, sizeof(int), cudaMemcpyHostToDevice, stream_));
    if (graph_on_ && !graph_) capture();
    for (int i = 0; i < n; ++i) {
        if (graph_on_) CUDA_CHECK(cudaGraphLaunch(graph_, stream_));
        else forward(stream_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    pos_ += n;
}

void Engine::copy_logits(float* host) {
    CUDA_CHECK(cudaMemcpyAsync(host, logits_, size_t(cfg_.vocab) * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::score(const int* ids, int k, int target, float* lp, int* argmax, float* target_lp, float* lse) {
    if (k < 0 || k > score_cap_) throw std::runtime_error("score: k out of range");
    if (target >= cfg_.vocab) throw std::runtime_error("score: target out of range");
    for (int i = 0; i < k; ++i)
        if (ids[i] < 0 || ids[i] >= cfg_.vocab) throw std::runtime_error("score: id out of range");
    if (k) CUDA_CHECK(cudaMemcpyAsync(score_ids_, ids, size_t(k) * sizeof(int), cudaMemcpyHostToDevice, stream_));
    launch_score(logits_, cfg_.vocab, score_ids_, k, target, score_work_, score_f_, score_i_, score_f_ + score_cap_,
                 score_f_ + score_cap_ + 1, stream_);
    if (k) CUDA_CHECK(cudaMemcpyAsync(lp, score_f_, size_t(k) * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    float tail[2];
    CUDA_CHECK(cudaMemcpyAsync(tail, score_f_ + score_cap_, sizeof(tail), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaMemcpyAsync(argmax, score_i_, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (target_lp) *target_lp = tail[0];
    if (lse) *lse = tail[1];
}

}  // namespace gptoss
