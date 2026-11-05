#include "script_host.h"
#include <algorithm>
#include <atomic>
#include <charconv>
#include <fstream>
#include <future>
#include <thread>

namespace slopfab::cli {
namespace fs = std::filesystem;

namespace {
constexpr uint64_t max_size = 9007199254740991ULL; // Exact JSON integers.

struct SingleStream : std::runtime_error {
  SingleStream() : std::runtime_error("server does not support validated byte ranges") {
  }
};

struct InvalidRange : std::runtime_error {
  using std::runtime_error::runtime_error;
};

std::string lower(std::string text) {
  for (auto& c : text)
    if (c >= 'A' && c <= 'Z')
      c += 'a' - 'A';
  return text;
}

std::string header(const Object& headers, const char* name) {
  const auto found = headers.find(name);
  return found == headers.end() ? "" : found->second.as_string();
}

uint64_t integer(const std::string& text) {
  uint64_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value > max_size)
    throw InvalidRange("invalid HTTP byte count");
  return value;
}

struct Range {
  uint64_t first, last, size;
};

Range content_range(const Object& headers) {
  const auto text = header(headers, "content-range");
  const auto dash = text.find('-'), slash = text.find('/');
  if (text.rfind("bytes ", 0) || dash == std::string::npos || slash == std::string::npos ||
      dash >= slash)
    throw InvalidRange("invalid HTTP Content-Range");
  Range range{integer(text.substr(6, dash - 6)), integer(text.substr(dash + 1, slash - dash - 1)),
              integer(text.substr(slash + 1))};
  if (range.first > range.last || range.last >= range.size)
    throw InvalidRange("invalid HTTP Content-Range bounds");
  return range;
}

bool strong_etag(const std::string& text) {
  if (text.size() < 2 || text.front() != '"' || text.back() != '"')
    return false;
  return std::all_of(text.begin() + 1, text.end() - 1, [](unsigned char c) {
    return c >= 0x21 && c != '"' && c != 0x7f;
  });
}

void identity(const Object& headers) {
  const auto encoding = lower(header(headers, "content-encoding"));
  if (!encoding.empty() && encoding != "identity")
    throw InvalidRange("ranged download requires identity Content-Encoding");
}

void regular_or_missing(const fs::path& path) {
  const auto status = fs::symlink_status(path);
  if (fs::exists(status) && !fs::is_regular_file(status))
    throw std::runtime_error("download state must be a regular file: " + path.u8string());
}
}

Value script_download(const std::string& url, const Value& options, const std::string& destination,
                      const std::function<void()>& check) {
  const auto requested = script_uint(options, "connections", 4);
  const auto retries = script_uint(options, "retries", 2);
  if (requested < 1 || requested > 16 || retries > 10)
    throw std::runtime_error("download connections must be 1..16 and retries 0..10");
  if (script_string(options, "method", "GET") != "GET" || !script_string(options, "body").empty())
    throw std::runtime_error("resumable downloads require GET without a request body");
  const bool resume = script_bool(options, "resume", true);
  const bool overwrite = script_bool(options, "overwrite");
  const auto path = fs::absolute(fs::u8path(destination));
  const auto state = fs::u8path(path.u8string() + ".download");
  const auto lock_path = fs::u8path(path.u8string() + ".download.lock");
  regular_or_missing(lock_path);
  FileLock lock(lock_path);
  if (!lock.acquired())
    throw std::runtime_error("download destination is already in use");
  if (!overwrite && fs::exists(path))
    throw std::runtime_error("download destination exists");
  if (fs::exists(fs::symlink_status(state)) && !fs::is_directory(fs::symlink_status(state)))
    throw std::runtime_error("download state must be a directory");
  for (int i = 0; i < 16; ++i)
    regular_or_missing(state / (std::to_string(i) + ".part"));
  regular_or_missing(state / "metadata.json");
  regular_or_missing(state / "assembled.part");
  auto clear_state = [&] {
    for (int i = 0; i < 16; ++i)
      fs::remove(state / (std::to_string(i) + ".part"));
    fs::remove(state / "assembled.part");
    fs::remove(state / "metadata.json");
    // Never recursively remove a directory or unrelated files.
    if (fs::exists(state) && fs::is_empty(state))
      fs::remove(state);
  };
  const auto started = std::chrono::steady_clock::now();
  const auto timeout = script_uint(options, "timeoutMs", 300000);
  std::atomic<bool> stop{false};
  auto worker_check = [&] {
    if (stop.load())
      throw std::runtime_error("download cancelled");
    if (timeout && uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count()) >= timeout)
      throw std::runtime_error("HTTP timeout");
  };
  auto main_check = [&] {
    check();
    worker_check();
  };
  Object base_headers;
  if (const auto* headers = options.find("headers")) {
    for (const auto& entry : headers->as_object()) {
      const auto name = lower(entry.first);
      if (name == "range" || name == "if-range" || name == "accept-encoding")
        throw std::runtime_error(
            "resumable downloads manage Range, If-Range and Accept-Encoding headers");
      base_headers[name] = entry.second;
    }
  }
  base_headers["accept-encoding"] = str("identity");
  auto request_options = [&](const std::string& range, const std::string& etag) {
    Object value = options.as_object(), headers = base_headers;
    if (!range.empty())
      headers["range"] = str("bytes=" + range);
    if (!etag.empty())
      headers["if-range"] = str(etag);
    value["headers"] = Value(headers);
    return Value(value);
  };
  auto single = [&]() {
    clear_state();
    for (uint64_t attempt = 0;; ++attempt) {
      main_check();
      const auto temporary = fs::u8path(path.u8string() + "." + unique_id() + ".part");
      std::ofstream output;
      uint64_t bytes = 0, expected = 0;
      bool known_size = false;
      HttpStream stream{[&](int status, const Object& headers) {
                          if (status != 200)
                            throw std::runtime_error("download returned HTTP " +
                                                     std::to_string(status));
                          identity(headers);
                          const auto length = header(headers, "content-length");
                          known_size = !length.empty();
                          if (known_size)
                            expected = integer(length);
                          output.open(temporary, std::ios::binary);
                          if (!output)
                            throw std::runtime_error("cannot open download destination");
                        },
                        [&](const char* data, size_t count) {
                          if (known_size && count > expected - bytes)
                            throw InvalidRange("download exceeds Content-Length");
                          output.write(data, std::streamsize(count));
                          if (!output)
                            throw std::runtime_error("cannot write download");
                          bytes += count;
                        }};
      try {
        auto response = script_http_request(url, request_options("", ""), "", main_check, &stream);
        output.close();
        if (!output || (known_size && bytes != expected))
          throw std::runtime_error("incomplete download");
        main_check();
        publish_http_download(temporary, path, overwrite);
        auto result = response.as_object();
        result["path"] = str(path.u8string());
        result["connections"] = num(1);
        result["resumedBytes"] = num(0);
        return Value(result);
      } catch (...) {
        output.close();
        std::error_code error;
        fs::remove(temporary, error);
        main_check();
        if (attempt == retries)
          throw;
      }
    }
  };
  uint64_t size = 0;
  std::string etag;
  Object response_headers;
  uint64_t probe_bytes = 0;
  HttpStream probe{[&](int status, const Object& headers) {
                     if (status == 200 || status == 416)
                       throw SingleStream();
                     if (status != 206)
                       throw std::runtime_error("download probe returned HTTP " +
                                                std::to_string(status));
                     identity(headers);
                     const auto range = content_range(headers);
                     if (range.first != 0 || range.last != 0)
                       throw InvalidRange("incorrect probe Content-Range");
                     etag = header(headers, "etag");
                     if (!strong_etag(etag))
                       throw SingleStream();
                     size = range.size;
                     response_headers = headers;
                   },
                   [&](const char*, size_t bytes) {
                     probe_bytes += bytes;
                     if (probe_bytes > 1)
                       throw InvalidRange("oversized range probe");
                   }};
  for (uint64_t attempt = 0;; ++attempt) {
    main_check();
    probe_bytes = 0;
    try {
      script_http_request(url, request_options("0-0", ""), "", main_check, &probe);
      if (probe_bytes != 1)
        throw InvalidRange("incomplete range probe");
      break;
    } catch (const SingleStream&) {
      return single();
    } catch (const InvalidRange&) {
      throw;
    } catch (...) {
      main_check();
      if (attempt == retries)
        throw;
    }
  }
  const auto connections = std::min(requested, size);
  const Value metadata(Object{{"version", num(1)},
                              {"url", str(url)},
                              {"etag", str(etag)},
                              {"size", num(double(size))},
                              {"connections", num(double(connections))}});
  if (fs::exists(state / "metadata.json")) {
    const auto previous = read_json(state / "metadata.json");
    if (!resume || json::stringify(previous) != json::stringify(metadata))
      clear_state();
  } else if (fs::exists(state) && !fs::is_empty(state)) {
    throw std::runtime_error("download state is missing metadata; move it aside before retrying");
  }
  fs::create_directories(state);
  atomic_json(state / "metadata.json", metadata);
  fs::remove(state / "assembled.part");
  uint64_t resumed_bytes = 0;
  std::vector<std::future<void>> workers;
  std::exception_ptr failure;
  try {
    for (uint64_t i = 0; i < connections; ++i) {
      const auto begin = size / connections * i + std::min(size % connections, i);
      const auto end = size / connections * (i + 1) + std::min(size % connections, i + 1);
      const auto part = state / (std::to_string(i) + ".part");
      if (fs::exists(part) && fs::file_size(part) > end - begin)
        fs::remove(part);
      if (fs::exists(part))
        resumed_bytes += fs::file_size(part);
      workers.push_back(std::async(std::launch::async, [&, begin, end, part] {
        for (uint64_t attempt = 0;; ++attempt) {
          worker_check();
          uint64_t have = fs::exists(part) ? fs::file_size(part) : 0;
          if (have == end - begin)
            return;
          const auto first = begin + have;
          std::ofstream output;
          HttpStream stream{
              [&](int status, const Object& headers) {
                if (status == 200)
                  throw SingleStream();
                if (status != 206)
                  throw std::runtime_error("range download returned HTTP " +
                                           std::to_string(status));
                identity(headers);
                const auto range = content_range(headers);
                if (range.first != first || range.last != end - 1 || range.size != size ||
                    header(headers, "etag") != etag)
                  throw InvalidRange("range response or ETag does not match requested file");
                const auto length = header(headers, "content-length");
                if (!length.empty() && integer(length) != end - first)
                  throw InvalidRange("range Content-Length does not match requested bytes");
                output.open(part, std::ios::binary | std::ios::app);
                if (!output)
                  throw std::runtime_error("cannot open download part");
              },
              [&](const char* data, size_t bytes) {
                if (bytes > end - begin - have)
                  throw InvalidRange("range body exceeds requested bytes");
                output.write(data, std::streamsize(bytes));
                if (!output)
                  throw std::runtime_error("cannot write download part");
                have += bytes;
              }};
          try {
            script_http_request(
                url, request_options(std::to_string(first) + "-" + std::to_string(end - 1), etag),
                "", worker_check, &stream);
            output.close();
            if (!output || have != end - begin)
              throw std::runtime_error("incomplete download part");
            return;
          } catch (const SingleStream&) {
            throw;
          } catch (const InvalidRange&) {
            if (output.is_open()) {
              output.close();
              fs::resize_file(part, first - begin);
            }
            throw;
          } catch (...) {
            output.close();
            worker_check();
            if (attempt == retries)
              throw;
          }
        }
      }));
    }
    // Only the calling thread invokes the runtime's cancellation/deadline callback.
    bool pending = true;
    while (pending) {
      main_check();
      pending = false;
      for (auto& worker : workers) {
        if (!worker.valid())
          continue;
        if (worker.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
          worker.get();
        else
          pending = true;
      }
      if (pending)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  } catch (...) {
    failure = std::current_exception();
    stop.store(true);
  }
  for (auto& worker : workers)
    if (worker.valid()) {
      try {
        worker.get();
      } catch (...) {
        if (!failure)
          failure = std::current_exception();
      }
    }
  if (failure) {
    if (!resume)
      clear_state();
    try {
      std::rethrow_exception(failure);
    } catch (const SingleStream&) {
      stop.store(false);
      return single();
    }
  }
  const auto assembled = state / "assembled.part";
  try {
    std::ofstream output(assembled, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot assemble download");
    char buffer[65536];
    for (uint64_t i = 0; i < connections; ++i) {
      std::ifstream input(state / (std::to_string(i) + ".part"), std::ios::binary);
      if (!input)
        throw std::runtime_error("cannot read download part");
      while (input) {
        main_check();
        input.read(buffer, sizeof(buffer));
        output.write(buffer, input.gcount());
        if (!output)
          throw std::runtime_error("cannot assemble download");
      }
      if (!input.eof())
        throw std::runtime_error("cannot read download part");
    }
    output.close();
    if (!output || fs::file_size(assembled) != size)
      throw std::runtime_error("assembled download size mismatch");
    main_check();
    publish_http_download(assembled, path, overwrite);
  } catch (...) {
    std::error_code error;
    fs::remove(assembled, error);
    if (!resume)
      clear_state();
    throw;
  }
  clear_state();
  response_headers.erase("content-range");
  response_headers["content-length"] = str(std::to_string(size));
  return Value(Object{{"status", num(200)},
                      {"ok", Value(true)},
                      {"headers", Value(response_headers)},
                      {"body", str("")},
                      {"bytes", num(double(size))},
                      {"path", str(path.u8string())},
                      {"connections", num(double(connections))},
                      {"resumedBytes", num(double(resumed_bytes))}});
}
}
