// Decode throughput at fixed context depths, comparable to `llama-bench -p 0 -n 128 -d <depths>`, and prefill
// throughput through gptoss_prefill.
//
//   gptoss-bench <model_dir> [--depths 0,4096,16384,32768] [--tokens 128] [--prefill 4096] [--reps 5] [--no-graph]
//
// For each depth the engine treats `depth` tokens as cached (decode cost depends on the number of keys, not
// their values), then times `tokens` decode steps. Prefill ingests `prefill` fixed pseudo-random token ids from
// an empty cache (0 skips it). One unmeasured run per test warms up (and captures the graph).
//
// `--depths 128,4096` gives the numbers for the PR template's table: tg128 at 128 and 4096 (decode@128,
// decode@4k) and pp4096 (prefill@4k).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "gptoss/gptoss.h"

static void die(const char* what) {
    std::fprintf(stderr, "%s: %s\n", what, gptoss_last_error());
    std::exit(1);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model_dir> [--depths a,b,c] [--tokens N] [--prefill N] [--reps N] [--no-graph]\n",
                     argv[0]);
        return 2;
    }
    std::vector<int> depths = {0, 4096, 16384, 32768};
    int tokens = 128, prefill = 4096, reps = 5, token = 1000;
    bool graph = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--depths" && i + 1 < argc) {
            depths.clear();
            std::stringstream ss(argv[++i]);
            for (std::string t; std::getline(ss, t, ',');) depths.push_back(std::atoi(t.c_str()));
        } else if (a == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
        else if (a == "--prefill" && i + 1 < argc) prefill = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--no-graph") graph = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (depths.empty() || tokens <= 0 || prefill < 0 || reps <= 0) { std::fprintf(stderr, "bad arguments\n"); return 2; }
    const int max_ctx = std::max(*std::max_element(depths.begin(), depths.end()) + tokens, prefill);
    gptoss_engine* e = gptoss_create(argv[1], max_ctx, 0);
    if (!e) die("create");
    if (gptoss_use_graph(e, graph ? 1 : 0)) die("use_graph");
    std::printf("{\"device_bytes\": %lld, \"graph\": %s, \"tokens\": %d, \"reps\": %d, \"results\": [\n",
                gptoss_device_bytes(e), graph ? "true" : "false", tokens, reps);
    std::vector<std::string> results;
    // Times run() reps + 1 times (the first is the warm-up) and records the median and spread of n / seconds.
    auto measure = [&](const char* test, int depth, int n, auto&& setup, auto&& run) {
        std::vector<double> tps;
        for (int r = 0; r <= reps; ++r) {
            setup();
            const auto t0 = std::chrono::steady_clock::now();
            run();
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (r > 0) tps.push_back(n / s);
        }
        std::sort(tps.begin(), tps.end());
        double mean = 0, var = 0;
        for (double v : tps) mean += v;
        mean /= tps.size();
        for (double v : tps) var += (v - mean) * (v - mean);
        const double sd = tps.size() > 1 ? std::sqrt(var / (tps.size() - 1)) : 0.0;
        char line[160];
        std::snprintf(line, sizeof line, "  {\"test\": \"%s%d\", \"depth\": %d, \"median_tok_s\": %.1f, \"stddev\": %.1f}",
                      test, n, depth, tps[tps.size() / 2], sd);
        results.push_back(line);
    };
    for (int depth : depths)
        measure("tg", depth, tokens, [&] { if (gptoss_set_position(e, depth)) die("set_position"); },
                [&] { if (gptoss_bench_steps(e, token, tokens)) die("bench_steps"); });
    if (prefill > 0) {
        std::vector<int32_t> ids(prefill);
        uint32_t x = 12345;
        for (auto& id : ids) { x = x * 1664525u + 1013904223u; id = int32_t((x >> 8) % uint32_t(gptoss_vocab(e))); }
        measure("pp", 0, prefill, [&] { if (gptoss_reset(e)) die("reset"); },
                [&] { if (gptoss_prefill(e, ids.data(), prefill, nullptr, 0, nullptr, nullptr, nullptr)) die("prefill"); });
    }
    for (size_t i = 0; i < results.size(); ++i)
        std::printf("%s%s\n", results[i].c_str(), i + 1 < results.size() ? "," : "");
    std::printf("]}\n");
    gptoss_destroy(e);
    return 0;
}
