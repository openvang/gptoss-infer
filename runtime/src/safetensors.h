// Read-only, memory-mapped access to a sharded safetensors checkpoint.
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gptoss {

struct TensorView {
    std::string dtype;            // "BF16", "F32", "U8", ...
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t numel() const;
};

class Checkpoint {
public:
    explicit Checkpoint(const std::string& model_dir);   // reads model.safetensors.index.json (or *.safetensors)
    ~Checkpoint();
    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    // Throws unless the tensor exists with exactly this dtype and shape.
    const TensorView& get(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape) const;
    const TensorView& get(const std::string& name) const;
    size_t size() const { return tensors_.size(); }

private:
    struct Mapping { void* addr = nullptr; size_t len = 0; };
    void map_file(const std::string& path);
    std::vector<Mapping> maps_;
    std::map<std::string, TensorView> tensors_;
};

}  // namespace gptoss
