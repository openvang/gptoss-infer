#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include "json.h"

namespace gptoss {

static size_t dtype_size(const std::string& dt) {
    if (dt == "F32" || dt == "I32" || dt == "U32") return 4;
    if (dt == "BF16" || dt == "F16" || dt == "I16" || dt == "U16") return 2;
    if (dt == "U8" || dt == "I8" || dt == "BOOL" || dt == "F8_E4M3" || dt == "F8_E5M2") return 1;
    if (dt == "F64" || dt == "I64" || dt == "U64") return 8;
    throw std::runtime_error("safetensors: unsupported dtype " + dt);
}

int64_t TensorView::numel() const {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

Checkpoint::Checkpoint(const std::string& model_dir) {
    std::set<std::string> files;
    std::ifstream idx(model_dir + "/model.safetensors.index.json");
    if (idx) {
        std::stringstream ss;
        ss << idx.rdbuf();
        Json j = Json::parse(ss.str());
        for (const auto& kv : j.at("weight_map").obj) files.insert(kv.second.as_str());
    } else {
        files.insert("model.safetensors");
    }
    for (const auto& f : files) map_file(model_dir + "/" + f);
}

void Checkpoint::map_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("safetensors: cannot open " + path);
    struct stat st {};
    if (fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("safetensors: stat failed " + path); }
    size_t len = size_t(st.st_size);
    void* addr = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (addr == MAP_FAILED) throw std::runtime_error("safetensors: mmap failed " + path);
    maps_.push_back({addr, len});

    const uint8_t* base = static_cast<const uint8_t*>(addr);
    if (len < 8) throw std::runtime_error("safetensors: truncated " + path);
    uint64_t hlen = 0;
    std::memcpy(&hlen, base, 8);                       // little-endian u64 header length
    if (8 + hlen > len) throw std::runtime_error("safetensors: bad header length in " + path);
    Json h = Json::parse(std::string(reinterpret_cast<const char*>(base + 8), hlen));
    const uint8_t* data = base + 8 + hlen;
    size_t data_len = len - 8 - hlen;
    for (const auto& kv : h.obj) {
        if (kv.first == "__metadata__") continue;
        const Json& t = kv.second;
        TensorView v;
        v.dtype = t.at("dtype").as_str();
        for (const auto& d : t.at("shape").arr) v.shape.push_back(d.as_int());
        int64_t b = t.at("data_offsets")[0].as_int(), e = t.at("data_offsets")[1].as_int();
        if (b < 0 || e < b || size_t(e) > data_len) throw std::runtime_error("safetensors: bad offsets for " + kv.first);
        v.data = data + b;
        v.nbytes = size_t(e - b);
        if (v.nbytes != size_t(v.numel()) * dtype_size(v.dtype))
            throw std::runtime_error("safetensors: size/shape mismatch for " + kv.first);
        if (!tensors_.emplace(kv.first, v).second) throw std::runtime_error("safetensors: duplicate " + kv.first);
    }
}

Checkpoint::~Checkpoint() {
    for (auto& m : maps_) munmap(m.addr, m.len);
}

const TensorView& Checkpoint::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("checkpoint: missing tensor " + name);
    return it->second;
}

const TensorView& Checkpoint::get(const std::string& name, const std::string& dtype,
                                  const std::vector<int64_t>& shape) const {
    const TensorView& v = get(name);
    if (v.dtype != dtype || v.shape != shape) {
        std::string want, got;
        for (auto d : shape) want += std::to_string(d) + ",";
        for (auto d : v.shape) got += std::to_string(d) + ",";
        throw std::runtime_error("checkpoint: " + name + " is " + v.dtype + "[" + got + "], expected " + dtype + "[" +
                                 want + "]");
    }
    return v;
}

}  // namespace gptoss
