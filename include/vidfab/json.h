// Minimal JSON reader, sized for safetensors headers.
//
// Deliberately not a general-purpose library: it parses the subset that appears
// in safetensors metadata (objects, arrays, strings, numbers, bool, null) and
// nothing more. Keeping it here avoids a third-party dependency for what is
// ultimately a few hundred lines of header parsing.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vidfab::json {

class Value;
using Object = std::map<std::string, Value>;
using Array = std::vector<Value>;

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
 public:
  Value() : type_(Type::Null) {}
  explicit Value(bool b) : type_(Type::Bool), bool_(b) {}
  explicit Value(double d) : type_(Type::Number), num_(d) {}
  explicit Value(std::string s) : type_(Type::String), str_(std::move(s)) {}
  explicit Value(Array a) : type_(Type::Array), arr_(std::make_shared<Array>(std::move(a))) {}
  explicit Value(Object o) : type_(Type::Object), obj_(std::make_shared<Object>(std::move(o))) {}

  Type type() const { return type_; }
  bool is_null() const { return type_ == Type::Null; }
  bool is_object() const { return type_ == Type::Object; }
  bool is_array() const { return type_ == Type::Array; }
  bool is_string() const { return type_ == Type::String; }
  bool is_number() const { return type_ == Type::Number; }

  const std::string& as_string() const {
    if (type_ != Type::String) throw std::runtime_error("json: not a string");
    return str_;
  }
  double as_number() const {
    if (type_ != Type::Number) throw std::runtime_error("json: not a number");
    return num_;
  }
  int64_t as_int() const { return static_cast<int64_t>(as_number()); }
  bool as_bool() const {
    if (type_ != Type::Bool) throw std::runtime_error("json: not a bool");
    return bool_;
  }
  const Array& as_array() const {
    if (type_ != Type::Array) throw std::runtime_error("json: not an array");
    return *arr_;
  }
  const Object& as_object() const {
    if (type_ != Type::Object) throw std::runtime_error("json: not an object");
    return *obj_;
  }

  // Object lookup returning nullptr when absent, so callers can probe optional
  // fields without exceptions.
  const Value* find(std::string_view key) const {
    if (type_ != Type::Object) return nullptr;
    auto it = obj_->find(std::string(key));
    return it == obj_->end() ? nullptr : &it->second;
  }

 private:
  Type type_;
  bool bool_ = false;
  double num_ = 0.0;
  std::string str_;
  std::shared_ptr<Array> arr_;
  std::shared_ptr<Object> obj_;
};

// Parses `text` in full. Throws std::runtime_error with a byte offset on
// malformed input; a truncated safetensors header should fail loudly rather
// than yield a half-built tensor table.
Value parse(std::string_view text);

}  // namespace vidfab::json
