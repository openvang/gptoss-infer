#include "json.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace gptoss {
namespace {

struct Parser {
    const std::string& s;
    size_t i = 0;

    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("json: ") + what + " at offset " + std::to_string(i));
    }
    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    bool lit(const char* w) {
        size_t n = std::char_traits<char>::length(w);
        if (s.compare(i, n, w) == 0) { i += n; return true; }
        return false;
    }
    static void utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += char(cp);
        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18)); out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F));
        }
    }
    uint32_t hex4() {
        if (i + 4 > s.size()) fail("short \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }
    std::string string() {
        if (s[i] != '"') fail("expected string");
        ++i;
        std::string out;
        while (true) {
            if (i >= s.size()) fail("unterminated string");
            char c = s[i++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) fail("bad escape");
            char e = s[i++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00) {          // surrogate pair
                        if (!lit("\\u")) fail("lone high surrogate");
                        uint32_t lo = hex4();
                        if (lo < 0xDC00 || lo >= 0xE000) fail("bad low surrogate");
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
    Json value() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        Json v;
        char c = s[i];
        if (c == '{') {
            v.type = Json::Type::Object;
            ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            while (true) {
                ws();
                std::string k = string();
                ws();
                if (i >= s.size() || s[i] != ':') fail("expected ':'");
                ++i;
                v.obj[k] = value();
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return v; }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type = Json::Type::Array;
            ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            while (true) {
                v.arr.push_back(value());
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return v; }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.type = Json::Type::String; v.str = string(); return v; }
        if (lit("true")) { v.type = Json::Type::Bool; v.b = true; return v; }
        if (lit("false")) { v.type = Json::Type::Bool; v.b = false; return v; }
        if (lit("null")) return v;
        const char* start = s.c_str() + i;
        char* end = nullptr;
        double d = std::strtod(start, &end);
        if (end == start) fail("bad value");
        i += size_t(end - start);
        v.type = Json::Type::Number;
        v.num = d;
        return v;
    }
};

}  // namespace

Json Json::parse(const std::string& text) {
    Parser p{text};
    Json v = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return v;
}

const Json& Json::at(const std::string& k) const {
    if (type != Type::Object) throw std::runtime_error("json: not an object (looking up '" + k + "')");
    auto it = obj.find(k);
    if (it == obj.end()) throw std::runtime_error("json: missing key '" + k + "'");
    return it->second;
}

int64_t Json::as_int() const {
    if (type != Type::Number || num != std::floor(num)) throw std::runtime_error("json: not an integer");
    return int64_t(num);
}
double Json::as_num() const {
    if (type != Type::Number) throw std::runtime_error("json: not a number");
    return num;
}
const std::string& Json::as_str() const {
    if (type != Type::String) throw std::runtime_error("json: not a string");
    return str;
}
bool Json::as_bool() const {
    if (type != Type::Bool) throw std::runtime_error("json: not a bool");
    return b;
}

}  // namespace gptoss
