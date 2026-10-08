// Minimal JSON reader for config.json and safetensors headers. Parses into a value tree; no writer.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gptoss {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    static Json parse(const std::string& text);   // throws std::runtime_error with an offset on bad input

    bool has(const std::string& k) const { return type == Type::Object && obj.count(k); }
    const Json& at(const std::string& k) const;   // throws if missing
    const Json& operator[](size_t i) const { return arr.at(i); }
    int64_t as_int() const;
    double as_num() const;
    const std::string& as_str() const;
    bool as_bool() const;
};

}  // namespace gptoss
