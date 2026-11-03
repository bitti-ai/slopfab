#include "script.h"
#include "motions.h"
#include "script_host.h"
#include "script_bootstrap.h"
#include "commands.h"
#include "pipe_process.h"
#include "slopfab/sha256.h"
#include <quickjs.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>
#include <cstdlib>
#include <cstring>

namespace slopfab::cli {
namespace fs = std::filesystem;

namespace {
struct Owned {
  JSContext* ctx;
  JSValue value;

  Owned(JSContext* c, JSValue v) : ctx(c), value(v) {
  }

  ~Owned() {
    JS_FreeValue(ctx, value);
  }

  Owned(const Owned&) = delete;
};

std::string js_string(JSContext* ctx, JSValueConst value) {
  size_t length;
  const char* text = JS_ToCStringLen(ctx, &length, value);
  if (!text)
    throw std::runtime_error("cannot convert JavaScript value to string");
  std::string out(text, length);
  JS_FreeCString(ctx, text);
  return out;
}

std::string js_error(JSContext* ctx, JSValueConst error) {
  std::string text = js_string(ctx, error);
  Owned stack(ctx, JS_GetPropertyStr(ctx, error, "stack"));
  if (JS_IsString(stack.value))
    text += "\n" + js_string(ctx, stack.value);
  return text;
}

void require_js(JSContext* ctx, JSValueConst value) {
  if (JS_IsException(value)) {
    Owned error(ctx, JS_GetException(ctx));
    throw std::runtime_error(js_error(ctx, error.value));
  }
}

fs::path path_arg(const std::string& text) {
  if (text.find('\0') != std::string::npos)
    throw std::runtime_error("path cannot contain NUL");
  return fs::u8path(text);
}

std::string read_file(const fs::path& path, uint64_t limit = 64 * 1024 * 1024) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot read " + path.u8string());
  if (!fs::is_regular_file(path))
    throw std::runtime_error("not a regular file: " + path.u8string());
  const auto size = fs::file_size(path);
  if (size > limit)
    throw std::runtime_error("file exceeds maxBytes: " + path.u8string());
  std::string data(size_t(size), '\0');
  input.read(data.data(), std::streamsize(data.size()));
  if (size_t(input.gcount()) != data.size())
    throw std::runtime_error("cannot read complete file");
  return data;
}

void write_file(const fs::path& path, const void* data, size_t size, bool append = false) {
  std::ofstream output(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
  output.write(static_cast<const char*>(data), std::streamsize(size));
  output.close();
  if (!output)
    throw std::runtime_error("cannot write " + path.u8string());
}

std::string file_url(const fs::path& path) {
  const auto raw = fs::absolute(path).lexically_normal().generic_u8string();
  std::string out = raw.rfind("//", 0) == 0 ? "file:" : raw.front() == '/' ? "file://" : "file:///";
  const char* hex = "0123456789ABCDEF";
  for (unsigned char c : raw) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
        std::strchr("/-._~:", c))
      out += char(c);
    else {
      out += '%';
      out += hex[c >> 4];
      out += hex[c & 15];
    }
  }
  return out;
}

struct Runtime {
  JSRuntime* rt = nullptr;
  JSContext* ctx = nullptr;
  uint64_t timeout;
  std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
  ScriptProcesses processes;
  Motions motions;
  Value result;
  bool has_result = false;
  std::map<void*, std::string> rejections;

  Runtime(size_t memory_mb, uint64_t timeout_ms)
      : timeout(timeout_ms), processes([this] {
          check();
        }),
        motions([this] {
          check();
        }) {
    rt = JS_NewRuntime();
    if (!rt)
      throw std::runtime_error("cannot allocate QuickJS runtime");
    JS_SetMemoryLimit(rt, memory_mb * 1024 * 1024);
    JS_SetMaxStackSize(rt, 512 * 1024);
    JS_SetCanBlock(rt, false);
    ctx = JS_NewContext(rt);
    if (!ctx) {
      JS_FreeRuntime(rt);
      rt = nullptr;
      throw std::runtime_error("cannot allocate QuickJS context");
    }
    JS_SetContextOpaque(ctx, this);
    JS_SetInterruptHandler(
        rt,
        [](JSContext*, void* opaque) -> int {
          try {
            static_cast<Runtime*>(opaque)->check();
            return 0;
          } catch (...) {
            return 1;
          }
        },
        this);
    JS_SetHostPromiseRejectionTracker(
        rt,
        [](JSContext* ctx, JSValueConst promise, JSValueConst reason, bool handled, void* opaque) {
          auto& r = *static_cast<Runtime*>(opaque);
          try {
            if (handled)
              r.rejections.erase(JS_VALUE_GET_PTR(promise));
            else
              r.rejections[JS_VALUE_GET_PTR(promise)] = js_error(ctx, reason);
          } catch (...) {
          } // Never unwind through the C engine.
        },
        this);
  }

  ~Runtime() {
    if (ctx)
      JS_FreeContext(ctx);
    if (rt)
      JS_FreeRuntime(rt);
  }

  void check() {
    if (cancelled())
      throw std::runtime_error("script cancelled");
    if (timeout && std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - started)
                           .count() >= int64_t(timeout))
      throw std::runtime_error("script timeout");
  }

  void sleep(uint64_t ms) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
    do {
      check();
      auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                           end - std::chrono::steady_clock::now())
                           .count();
      if (remaining <= 0)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(std::min<int64_t>(remaining, 10)));
    } while (true);
  }
};

Value host_call(Runtime& r, const std::string& op, const Array& args) {
  r.check();
  if (op.rfind("motions.", 0) == 0)
    return r.motions.call(op.substr(8), args);
  const auto s = [&](size_t i) -> const std::string& {
    return args.at(i).as_string();
  };
  const auto p = [&](size_t i) {
    return path_arg(s(i));
  };
  if (op == "result") {
    r.result = args.at(0);
    r.has_result = true;
    return Value();
  }
  if (op == "log") {
    std::fprintf(stderr, "%s\n", s(0).c_str());
    return Value();
  }
  if (op == "spawn") {
    std::vector<std::string> argv;
    for (const auto& arg : args.at(0).as_array())
      argv.push_back(arg.as_string());
    return r.processes.spawn(argv, args.at(1));
  }
  if (op == "poll" || op == "wait")
    return r.processes.poll(int(args.at(0).as_int()), op == "wait");
  if (op == "kill") {
    r.processes.kill(int(args.at(0).as_int()));
    return Value();
  }
  if (op == "http")
    return script_http(s(0), args.at(1), s(2), [&] {
      r.check();
    });
  if (op == "readText")
    return str(read_file(p(0), script_uint(args.at(1), "maxBytes", 64 * 1024 * 1024)));
  if (op == "writeText") {
    write_file(p(0), s(1).data(), s(1).size(), script_bool(args.at(2), "append"));
    return Value();
  }
  if (op == "exists")
    return Value(fs::exists(p(0)));
  if (op == "stat") {
    const auto status = fs::symlink_status(p(0));
    return Value(
        Object{{"isFile", Value(fs::is_regular_file(status))},
               {"isDirectory", Value(fs::is_directory(status))},
               {"isSymlink", Value(fs::is_symlink(status))},
               {"size", num(fs::is_regular_file(status) ? double(fs::file_size(p(0))) : 0)}});
  }
  if (op == "list") {
    std::vector<fs::path> entries;
    for (const auto& e : fs::directory_iterator(p(0)))
      entries.push_back(e.path());
    std::sort(entries.begin(), entries.end());
    Array out;
    for (const auto& e : entries)
      out.push_back(Value(Object{{"name", str(e.filename().u8string())},
                                 {"path", str(fs::absolute(e).u8string())},
                                 {"isDirectory", Value(fs::is_directory(fs::symlink_status(e)))},
                                 {"isSymlink", Value(fs::is_symlink(fs::symlink_status(e)))}}));
    return Value(out);
  }
  if (op == "mkdir") {
    fs::create_directories(p(0));
    return Value();
  }
  if (op == "copy") {
    fs::copy(p(0), p(1),
             (script_bool(args.at(2), "recursive") ? fs::copy_options::recursive
                                                   : fs::copy_options::none) |
                 (script_bool(args.at(2), "overwrite") ? fs::copy_options::overwrite_existing
                                                       : fs::copy_options::none));
    return Value();
  }
  if (op == "rename") {
    fs::rename(p(0), p(1));
    return Value();
  }
  if (op == "remove") {
    return num(
        double(script_bool(args.at(1), "recursive") ? fs::remove_all(p(0)) : fs::remove(p(0))));
  }
  if (op == "cwd")
    return str(fs::current_path().u8string());
  if (op == "chdir") {
    fs::current_path(p(0));
    return Value();
  }
  if (op == "tool")
    return str(media_tool(s(0).c_str(), agent().executable.c_str()));
  if (op == "sha256") {
    const auto digest = sha256_file(fs::absolute(p(0)).u8string());
    std::string hex;
    for (auto byte : digest) {
      hex += "0123456789abcdef"[byte >> 4];
      hex += "0123456789abcdef"[byte & 15];
    }
    return str(hex);
  }
  if (op == "env.get" || op == "env.set") {
    if (s(0).empty() || s(0).find('=') != std::string::npos || s(0).find('\0') != std::string::npos)
      throw std::runtime_error("invalid environment key");
    if (op == "env.get") {
#ifdef _WIN32
      const wchar_t* value = _wgetenv(fs::u8path(s(0)).c_str());
      return value ? str(fs::path(value).u8string()) : Value();
#else
      const char* value = std::getenv(s(0).c_str());
      return value ? str(value) : Value();
#endif
    }
    const bool unset = args.at(1).is_null();
    const auto value = unset ? std::string() : s(1);
    if (value.find('\0') != std::string::npos)
      throw std::runtime_error("environment value cannot contain NUL");
#ifdef _WIN32
    if (_wputenv_s(fs::u8path(s(0)).c_str(), fs::u8path(value).c_str()) != 0)
      throw std::runtime_error("cannot set environment");
#else
    if ((unset ? unsetenv(s(0).c_str()) : setenv(s(0).c_str(), value.c_str(), 1)) != 0)
      throw std::runtime_error("cannot set environment");
#endif
    return Value();
  }
  if (op == "sleep") {
    r.sleep(script_uint(Value(Object{{"ms", args.at(0)}}), "ms", 0));
    return Value();
  }
  if (op == "path.join" || op == "path.resolve") {
    fs::path path;
    for (size_t i = 0; i < args.size(); ++i)
      path /= p(i);
    return str((op == "path.resolve" ? fs::absolute(path) : path).lexically_normal().u8string());
  }
  if (op == "path.dirname")
    return str(p(0).parent_path().u8string());
  if (op == "path.basename")
    return str(p(0).filename().u8string());
  if (op == "path.extname")
    return str(p(0).extension().u8string());
  if (op == "path.fileUrl")
    return str(file_url(p(0)));
  throw std::runtime_error("unknown host operation: " + op);
}

JSValue native_host(JSContext* ctx, JSValueConst, int argc, JSValueConst* argv) {
  try {
    if (argc != 2)
      throw std::runtime_error("host requires operation and arguments");
    auto& runtime = *static_cast<Runtime*>(JS_GetContextOpaque(ctx));
    const auto args = parse_cli_json(js_string(ctx, argv[1]));
    const auto out = host_call(runtime, js_string(ctx, argv[0]), args.as_array());
    const auto text = json::stringify(out);
    return JS_ParseJSON(ctx, text.data(), text.size(), "<host-result>");
  } catch (const std::exception& e) {
    return JS_ThrowPlainError(ctx, "%s", e.what());
  }
}

JSValue native_bytes(JSContext* ctx, JSValueConst, int argc, JSValueConst* argv, int operation) {
  try {
    if (argc < 1)
      throw std::runtime_error("missing argument");
    auto& runtime = *static_cast<Runtime*>(JS_GetContextOpaque(ctx));
    runtime.check();
    if (operation == 0 || operation == 2) {
      uint64_t limit = 64 * 1024 * 1024;
      if (operation == 0 && argc > 1) {
        int64_t n;
        if (JS_ToInt64(ctx, &n, argv[1]) < 0 || n < 0)
          throw std::runtime_error("invalid maxBytes");
        limit = uint64_t(n);
      }
      const auto data = operation == 0 ? read_file(path_arg(js_string(ctx, argv[0])), limit)
                                       : js_string(ctx, argv[0]);
      return JS_NewUint8ArrayCopy(ctx, reinterpret_cast<const uint8_t*>(data.data()), data.size());
    }
    const int index = operation == 1 ? 1 : 0;
    if (argc <= index)
      throw std::runtime_error("missing binary data");
    size_t offset = 0, length = 0, element = 0;
    JSValue buffer = JS_UNDEFINED;
    uint8_t* data = nullptr;
    if (JS_IsArrayBuffer(argv[index]))
      data = JS_GetArrayBuffer(ctx, &length, argv[index]);
    else {
      buffer = JS_GetTypedArrayBuffer(ctx, argv[index], &offset, &length, &element);
      require_js(ctx, buffer);
      size_t bytes = 0;
      data = JS_GetArrayBuffer(ctx, &bytes, buffer);
      if (offset > bytes || length > bytes - offset) {
        JS_FreeValue(ctx, buffer);
        throw std::runtime_error("invalid binary view");
      }
      if (data)
        data += offset;
    }
    Owned owned(ctx, buffer);
    if (!data && length)
      throw std::runtime_error("detached binary data");
    if (operation == 3)
      return JS_NewStringLen(ctx, reinterpret_cast<const char*>(data), length);
    write_file(path_arg(js_string(ctx, argv[0])), data, length);
    return JS_UNDEFINED;
  } catch (const std::exception& e) {
    return JS_ThrowPlainError(ctx, "%s", e.what());
  }
}

char* normalize_module(JSContext* ctx, const char* base, const char* name, void*) {
  try {
    std::string resolved = name;
    if (resolved != "slopfab") {
      fs::path path = path_arg(resolved);
      if (!path.is_absolute())
        path = path_arg(base).parent_path() / path;
      resolved = fs::absolute(path).lexically_normal().u8string();
    }
    return js_strdup(ctx, resolved.c_str());
  } catch (const std::exception& e) {
    JS_ThrowReferenceError(ctx, "%s", e.what());
    return nullptr;
  }
}

void import_meta(JSContext* ctx, JSModuleDef* module, const std::string& name, bool main) {
  Owned meta(ctx, JS_GetImportMeta(ctx, module));
  require_js(ctx, meta.value);
  JS_SetPropertyStr(
      ctx, meta.value, "url",
      JS_NewString(ctx, name == "slopfab" ? "slopfab" : file_url(path_arg(name)).c_str()));
  JS_SetPropertyStr(ctx, meta.value, "main", JS_NewBool(ctx, main));
}

JSModuleDef* load_module(JSContext* ctx, const char* name, void*) {
  try {
    const auto source =
        std::string(name) == "slopfab"
            ? std::string(
                  "export default globalThis.slopfab; export const {fs,path,http,exec,spawn,cli,env,sleep,sha256,tool,args,motions}=globalThis.slopfab;")
            : read_file(path_arg(name));
    Owned compiled(ctx, JS_Eval(ctx, source.data(), source.size(), name,
                                JS_EVAL_TYPE_MODULE | JS_EVAL_FLAG_COMPILE_ONLY));
    require_js(ctx, compiled.value);
    auto* module = static_cast<JSModuleDef*>(JS_VALUE_GET_PTR(compiled.value));
    import_meta(ctx, module, name, false);
    return module;
  } catch (const std::exception& e) {
    JS_ThrowReferenceError(ctx, "%s", e.what());
    return nullptr;
  }
}

// Drive microtasks and timers; unresolved promises without work are errors.
JSValue await_value(Runtime& r, JSValueConst value, bool drain = false) {
  if (JS_IsPromise(value))
    JS_PromiseMarkAsHandled(r.ctx, value);
  Owned global(r.ctx, JS_GetGlobalObject(r.ctx));
  Owned tick(r.ctx, JS_GetPropertyStr(r.ctx, global.value, "__slopfabTick"));
  for (;;) {
    r.check();
    JSContext* job_ctx = nullptr;
    while (JS_IsJobPending(r.rt)) {
      r.check();
      if (JS_ExecutePendingJob(r.rt, &job_ctx) < 0)
        require_js(job_ctx, JS_EXCEPTION);
    }
    const auto state = JS_PromiseState(r.ctx, value);
    if (state == JS_PROMISE_REJECTED) {
      Owned reason(r.ctx, JS_PromiseResult(r.ctx, value));
      throw std::runtime_error(js_error(r.ctx, reason.value));
    }
    if (!drain && state != JS_PROMISE_PENDING)
      return state == JS_PROMISE_FULFILLED ? JS_PromiseResult(r.ctx, value)
                                           : JS_DupValue(r.ctx, value);
    Owned timers(r.ctx, JS_Call(r.ctx, tick.value, JS_UNDEFINED, 0, nullptr));
    require_js(r.ctx, timers.value);
    int64_t delay = -1;
    JS_ToInt64(r.ctx, &delay, timers.value);
    if (JS_IsJobPending(r.rt))
      continue;
    if (JS_PromiseState(r.ctx, value) != state)
      continue;
    if (delay < 0) {
      if (state == JS_PROMISE_PENDING)
        throw std::runtime_error("promise is unresolved with no pending jobs or timers");
      return JS_UNDEFINED;
    }
    r.sleep(uint64_t(std::min<int64_t>(delay, 10)));
  }
}
}

int run_script(const fs::path& input, const std::vector<std::string>& args,
               const std::string* source, size_t memory_mb, uint64_t timeout_ms) {
  if (memory_mb < 16 || memory_mb > 16384)
    throw CliError("INVALID_REQUEST", "memory-mb must be in [16,16384]");
  const auto path = fs::absolute(input).lexically_normal();
  const auto text = source ? *source : read_file(path);
  try {
    Runtime r(memory_mb, timeout_ms);
    JS_SetModuleLoaderFunc(r.rt, normalize_module, load_module, nullptr);
    Owned global(r.ctx, JS_GetGlobalObject(r.ctx));
    JS_SetPropertyStr(r.ctx, global.value, "__slopfabHost",
                      JS_NewCFunction(r.ctx, native_host, "host", 2));
    const char* names[] = {"__slopfabReadBytes", "__slopfabWriteBytes", "__slopfabEncode",
                           "__slopfabDecode"};
    for (int i = 0; i < 4; ++i)
      JS_SetPropertyStr(
          r.ctx, global.value, names[i],
          JS_NewCFunctionMagic(r.ctx, native_bytes, names[i], 2, JS_CFUNC_generic_magic, i));
    Array arguments;
    for (const auto& arg : args)
      arguments.push_back(str(arg));
    const auto info =
        json::stringify(Value(Object{{"args", Value(arguments)},
                                     {"scriptPath", str(path.u8string())},
                                     {"scriptDirectory", str(path.parent_path().u8string())},
                                     {"executable", str(agent().executable)},
                                     {"config", str(fs::absolute(config_path()).u8string())},
                                     {"jobs", str(fs::absolute(jobs_path()).u8string())},
                                     {"version", str(kVersion)},
                                     {"apiVersion", num(1)},
#ifdef _WIN32
                                     {"platform", str("win32")}
#else
                                     {"platform", str("linux")}
#endif
        }));
    JS_SetPropertyStr(r.ctx, global.value, "__slopfabInfo",
                      JS_ParseJSON(r.ctx, info.data(), info.size(), "<info>"));
    Owned bootstrap(r.ctx, JS_Eval(r.ctx, kScriptBootstrap, std::strlen(kScriptBootstrap),
                                   "<slopfab>", JS_EVAL_TYPE_GLOBAL));
    require_js(r.ctx, bootstrap.value);
    Owned module(r.ctx, JS_Eval(r.ctx, text.data(), text.size(), path.u8string().c_str(),
                                JS_EVAL_TYPE_MODULE | JS_EVAL_FLAG_COMPILE_ONLY));
    require_js(r.ctx, module.value);
    auto* def = static_cast<JSModuleDef*>(JS_VALUE_GET_PTR(module.value));
    import_meta(r.ctx, def, path.u8string(), true);
    Owned evaluated(r.ctx, JS_EvalFunction(r.ctx, JS_DupValue(r.ctx, module.value)));
    require_js(r.ctx, evaluated.value);
    Owned ready(r.ctx, await_value(r, evaluated.value));
    Owned space(r.ctx, JS_GetModuleNamespace(r.ctx, def));
    require_js(r.ctx, space.value);
    Owned main(r.ctx, JS_GetPropertyStr(r.ctx, space.value, "default"));
    require_js(r.ctx, main.value);
    if (JS_IsFunction(r.ctx, main.value)) {
      Owned sf(r.ctx, JS_GetPropertyStr(r.ctx, global.value, "slopfab"));
      Owned argv(r.ctx, JS_GetPropertyStr(r.ctx, sf.value, "args"));
      Owned called(r.ctx, JS_Call(r.ctx, main.value, JS_UNDEFINED, 1, &argv.value));
      require_js(r.ctx, called.value);
      Owned result(r.ctx, await_value(r, called.value));
      if (!r.has_result && !JS_IsUndefined(result.value)) {
        Owned encoded(r.ctx, JS_JSONStringify(r.ctx, result.value, JS_UNDEFINED, JS_UNDEFINED));
        require_js(r.ctx, encoded.value);
        r.result = parse_cli_json(js_string(r.ctx, encoded.value));
      }
    }
    Owned drained(r.ctx, await_value(r, JS_UNDEFINED, true));
    if (!r.rejections.empty())
      throw std::runtime_error("Unhandled promise rejection: " + r.rejections.begin()->second);
    emit({{"type", str("result")},
          {"ok", Value(true)},
          {"script", str(path.u8string())},
          {"data", r.result}},
         true);
    return 0;
  } catch (const CliError&) {
    throw;
  } catch (const std::exception& e) {
    throw CliError("SCRIPT_ERROR", e.what(), 6, path.u8string());
  }
}

int script_command(const std::vector<std::string>& args) {
  size_t memory = 256;
  uint64_t timeout = 0;
  std::string path, source;
  bool inline_source = false;
  std::vector<std::string> script_args;
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--") {
      script_args.assign(args.begin() + i + 1, args.end());
      break;
    }
    if (args[i] == "--memory-mb" || args[i] == "--timeout-ms") {
      const auto option = args[i];
      if (++i == args.size())
        throw CliError("INVALID_REQUEST", "missing " + option);
      const auto value = parse_cli_json(args[i]);
      const auto n = script_uint(Value(Object{{"value", value}}), "value", 0);
      if (option == "--memory-mb")
        memory = size_t(n);
      else
        timeout = n;
    } else if (args[i] == "--eval") {
      if (!path.empty() || ++i == args.size())
        throw CliError("INVALID_REQUEST", "run accepts one file or --eval source");
      path = "<eval>.js";
      source = args[i];
      inline_source = true;
    } else if (path.empty())
      path = args[i];
    else
      throw CliError("INVALID_REQUEST", "put script arguments after --");
  }
  if (path.empty())
    throw CliError("INVALID_REQUEST", "use run FILE.js [-- ARGS] or run --eval SOURCE");
  if (path == "-") {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    source = buffer.str();
    path = "<stdin>.js";
    inline_source = true;
  }
  return run_script(path, script_args, inline_source ? &source : nullptr, memory, timeout);
}
}
