// Decode throughput at fixed context depths, comparable to `llama-bench -p 0 -n 128 -d <depths>`.
//
//   gptoss-bench <model_dir> [--depths 0,4096,16384,32768] [--tokens 128] [--reps 5] [--no-graph]
//
// For each depth the engine treats `depth` tokens as cached (decode cost depends on the number of keys, not
// their values), then times `tokens` decode steps. One unmeasured run per depth warms up (and captures the graph).
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
        std::fprintf(stderr, "usage: %s <model_dir> [--depths a,b,c] [--tokens N] [--reps N] [--no-graph]\n", argv[0]);
        return 2;
    }
    std::vector<int> depths = {0, 4096, 16384, 32768};
    int tokens = 128, reps = 5, token = 1000;
    bool graph = true;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--depths" && i + 1 < argc) {
            depths.clear();
            std::stringstream ss(argv[++i]);
            for (std::string t; std::getline(ss, t, ',');) depths.push_back(std::atoi(t.c_str()));
        } else if (a == "--tokens" && i + 1 < argc) tokens = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--no-graph") graph = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const int max_ctx = *std::max_element(depths.begin(), depths.end()) + tokens;
    gptoss_engine* e = gptoss_create(argv[1], max_ctx);
    if (!e) die("create");
    if (gptoss_use_graph(e, graph ? 1 : 0)) die("use_graph");
    std::printf("{\"device_bytes\": %lld, \"graph\": %s, \"tokens\": %d, \"reps\": %d, \"results\": [\n",
                gptoss_device_bytes(e), graph ? "true" : "false", tokens, reps);
    for (size_t di = 0; di < depths.size(); ++di) {
        std::vector<double> tps;
        for (int r = 0; r <= reps; ++r) {
            if (gptoss_set_position(e, depths[di])) die("set_position");
            const auto t0 = std::chrono::steady_clock::now();
            if (gptoss_bench_steps(e, token, tokens)) die("bench_steps");
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (r > 0) tps.push_back(tokens / s);            // r == 0 is the warm-up
        }
        std::sort(tps.begin(), tps.end());
        double mean = 0, var = 0;
        for (double v : tps) mean += v;
        mean /= tps.size();
        for (double v : tps) var += (v - mean) * (v - mean);
        const double sd = tps.size() > 1 ? std::sqrt(var / (tps.size() - 1)) : 0.0;
        std::printf("  {\"test\": \"tg%d\", \"depth\": %d, \"median_tok_s\": %.1f, \"stddev\": %.1f}%s\n", tokens,
                    depths[di], tps[tps.size() / 2], sd, di + 1 < depths.size() ? "," : "");
    }
    std::printf("]}\n");
    gptoss_destroy(e);
    return 0;
}
