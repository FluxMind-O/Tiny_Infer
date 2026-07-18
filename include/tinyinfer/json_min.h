#pragma once
// TinyInfer 轻量 JSON 解析器（仅覆盖本引擎所需子集：对象/数组/字符串/数字/布尔/null）
// 免去对 nlohmann/json 的外部依赖，便于直接编译。
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <cctype>

namespace tinynfer {
namespace json {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<ValuePtr> arr;
    std::unordered_map<std::string, ValuePtr> obj;

    bool is_object() const { return type == Object; }
    bool is_array() const { return type == Array; }
    bool is_string() const { return type == String; }
    bool is_number() const { return type == Number; }
    const std::string& as_string() const { return str; }
    double as_number() const { return num; }
    int as_int() const { return (int)num; }
    bool as_bool() const { return b; }

    const ValuePtr& at(const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end())
            throw std::runtime_error("JSON: missing key '" + k + "'");
        return it->second;
    }
    bool has(const std::string& k) const { return obj.find(k) != obj.end(); }
};

class Parser {
public:
    explicit Parser(const std::string& s) : s_(s), i_(0) {}

    ValuePtr parse() {
        skip_ws();
        ValuePtr v = parse_value();
        skip_ws();
        if (i_ != s_.size())
            throw std::runtime_error("JSON: trailing characters at " + std::to_string(i_));
        return v;
    }

private:
    const std::string& s_;
    size_t i_;

    void skip_ws() {
        while (i_ < s_.size() && std::isspace((unsigned char)s_[i_])) ++i_;
    }

    char peek() { return i_ < s_.size() ? s_[i_] : '\0'; }

    ValuePtr parse_value() {
        skip_ws();
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string_value();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') { i_ += 4; return std::make_shared<Value>(); }
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw std::runtime_error("JSON: unexpected char '" + std::string(1, c) +
                                 "' at " + std::to_string(i_));
    }

    ValuePtr parse_object() {
        auto v = std::make_shared<Value>();
        v->type = Value::Object;
        ++i_;  // {
        skip_ws();
        if (peek() == '}') { ++i_; return v; }
        while (true) {
            skip_ws();
            if (peek() != '"') throw std::runtime_error("JSON: expected key string");
            std::string key = parse_raw_string();
            skip_ws();
            if (peek() != ':') throw std::runtime_error("JSON: expected ':'");
            ++i_;
            ValuePtr val = parse_value();
            v->obj[key] = val;
            skip_ws();
            char c = peek();
            if (c == ',') { ++i_; continue; }
            if (c == '}') { ++i_; break; }
            throw std::runtime_error("JSON: expected ',' or '}'");
        }
        return v;
    }

    ValuePtr parse_array() {
        auto v = std::make_shared<Value>();
        v->type = Value::Array;
        ++i_;  // [
        skip_ws();
        if (peek() == ']') { ++i_; return v; }
        while (true) {
            ValuePtr val = parse_value();
            v->arr.push_back(val);
            skip_ws();
            char c = peek();
            if (c == ',') { ++i_; continue; }
            if (c == ']') { ++i_; break; }
            throw std::runtime_error("JSON: expected ',' or ']'");
        }
        return v;
    }

    std::string parse_raw_string() {
        if (peek() != '"') throw std::runtime_error("JSON: expected '\"'");
        ++i_;
        std::string out;
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') break;
            if (c == '\\') {
                char e = s_[i_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        return out;
    }

    ValuePtr parse_string_value() {
        auto v = std::make_shared<Value>();
        v->type = Value::String;
        v->str = parse_raw_string();
        return v;
    }

    ValuePtr parse_bool() {
        auto v = std::make_shared<Value>();
        v->type = Value::Bool;
        if (s_.compare(i_, 4, "true") == 0) { v->b = true; i_ += 4; }
        else if (s_.compare(i_, 5, "false") == 0) { v->b = false; i_ += 5; }
        else throw std::runtime_error("JSON: invalid bool");
        return v;
    }

    ValuePtr parse_number() {
        size_t start = i_;
        if (peek() == '-') ++i_;
        while (i_ < s_.size() &&
               (std::isdigit((unsigned char)s_[i_]) || s_[i_] == '.' ||
                s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-'))
            ++i_;
        auto v = std::make_shared<Value>();
        v->type = Value::Number;
        v->num = std::stod(s_.substr(start, i_ - start));
        return v;
    }
};

// 从文件读取并解析
ValuePtr parse_file(const std::string& path);

}  // namespace json
}  // namespace tinynfer
