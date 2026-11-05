#include "script_host.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <limits>
#include <climits>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#else
#include <curl/curl.h>
#include <unistd.h>
#endif

namespace slopfab::cli {
namespace fs = std::filesystem;

namespace {
struct Transfer {
  std::function<void()> check;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  uint64_t timeout = 300000, limit = 16 * 1024 * 1024, bytes = 0;
  std::string body;
  std::exception_ptr failure;
  std::ofstream file;
  Object headers;
  HttpStream* stream = nullptr;
  int status = 0;
  bool accepted = false;

  void accept(int code) {
    status = code;
    if (accepted)
      return;
    if (stream)
      stream->headers(status, headers);
    accepted = true;
  }

  void tick() {
    check();
    if (timeout && std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start)
                           .count() >= int64_t(timeout))
      throw std::runtime_error("HTTP timeout");
  }

  void write(const char* data, size_t size) {
    tick();
    // libcurl can deliver redirect bodies before the final response.
    if (status >= 300 && status < 400)
      return;
    bytes += size;
    if (stream) {
      accept(status);
      stream->write(data, size);
    } else if (file.is_open()) {
      file.write(data, std::streamsize(size));
      if (!file)
        throw std::runtime_error("cannot write HTTP download");
    } else {
      if (bytes > limit)
        throw std::runtime_error("HTTP response exceeds maxBytes; use http.download");
      body.append(data, size);
    }
  }

  void header(std::string line) {
    if (line.rfind("HTTP/", 0) == 0) {
      headers.clear();
      accepted = false;
      bytes = 0;
      std::istringstream response(line);
      std::string protocol;
      response >> protocol >> status;
      return;
    }
    if (line == "\r\n" || line == "\n") {
      if (status >= 200 && !(status >= 300 && status < 400))
        accept(status);
      return;
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos)
      return;
    auto name = line.substr(0, colon);
    for (auto& c : name)
      if (c >= 'A' && c <= 'Z')
        c += 'a' - 'A';
    auto value = line.substr(colon + 1);
    const auto begin = value.find_first_not_of(" \t");
    value = begin == std::string::npos ? "" : value.substr(begin);
    const auto end = value.find_last_not_of("\r\n \t");
    value = end == std::string::npos ? "" : value.substr(0, end + 1);
    headers[name] = str(value);
  }
};

void valid_header(const std::string& value) {
  if (value.find_first_of("\r\n") != std::string::npos || value.find('\0') != std::string::npos)
    throw std::runtime_error("invalid HTTP header");
}
}

Value script_http(const std::string& url, const Value& options, const std::string& destination,
                  const std::function<void()>& check) {
  if (!destination.empty()) {
    const auto connections = script_uint(options, "connections", 1);
    if (connections < 1 || connections > 16)
      throw std::runtime_error("download connections must be 1..16");
  }
  if (!destination.empty() &&
      (script_bool(options, "resume") || script_uint(options, "connections", 1) > 1))
    return script_download(url, options, destination, check);
  return script_http_request(url, options, destination, check);
}

void publish_http_download(const fs::path& temporary, const fs::path& path, bool overwrite) {
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), path.c_str(),
                   MOVEFILE_WRITE_THROUGH | (overwrite ? MOVEFILE_REPLACE_EXISTING : 0)))
    throw std::runtime_error("cannot publish download");
#else
  if (overwrite)
    fs::rename(temporary, path);
  else {
    if (::link(temporary.c_str(), path.c_str()) != 0)
      throw std::runtime_error("cannot publish download (destination exists or is unwritable)");
    fs::remove(temporary);
  }
#endif
}

Value script_http_request(const std::string& url, const Value& options,
                          const std::string& destination, const std::function<void()>& check,
                          HttpStream* stream) {
  if ((url.rfind("https://", 0) != 0 && url.rfind("http://", 0) != 0) ||
      url.find('\0') != std::string::npos)
    throw std::runtime_error("HTTP URL must use http:// or https://");
  Transfer transfer;
  transfer.stream = stream;
  transfer.check = check;
  transfer.timeout = script_uint(options, "timeoutMs", transfer.timeout);
  transfer.limit = script_uint(options, "maxBytes", transfer.limit);
  auto method = script_string(options, "method", "GET");
  valid_header(method);
  const auto body = script_string(options, "body");
  fs::path path, temporary;
  const bool overwrite = script_bool(options, "overwrite");
  if (!destination.empty()) {
    path = fs::absolute(fs::u8path(destination));
    if (!overwrite && fs::exists(path))
      throw std::runtime_error("download destination exists");
    temporary = path;
    temporary += "." + unique_id() + ".part";
    transfer.file.open(temporary, std::ios::binary);
    if (!transfer.file)
      throw std::runtime_error("cannot open download destination");
  }
  int status = 0;
  try {
    transfer.tick();
#ifdef _WIN32
    struct Internet {
      HINTERNET handle = nullptr;

      ~Internet() {
        if (handle)
          WinHttpCloseHandle(handle);
      }
    } session, connection, request;

    const auto wide_url = fs::u8path(url).wstring();
    URL_COMPONENTS parts{};
    parts.dwStructSize = sizeof(parts);
    parts.dwHostNameLength = parts.dwUrlPathLength = parts.dwExtraInfoLength = DWORD(-1);
    if (!WinHttpCrackUrl(wide_url.c_str(), DWORD(wide_url.size()), 0, &parts))
      throw std::runtime_error("invalid HTTP URL");
    const std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    std::wstring resource(parts.lpszUrlPath, parts.dwUrlPathLength);
    if (parts.dwExtraInfoLength)
      resource.append(parts.lpszExtraInfo, parts.dwExtraInfoLength);
    if (resource.empty())
      resource = L"/";
    session.handle = WinHttpOpen(L"slopfab-workflows/1", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                                 nullptr, nullptr, 0);
    if (!session.handle)
      throw std::runtime_error("cannot create HTTP session");
    const int timeout = int(
        std::min<uint64_t>(transfer.timeout ? transfer.timeout : 60000, stream ? 15000 : INT_MAX));
    WinHttpSetTimeouts(session.handle, timeout, timeout, timeout, timeout);
    connection.handle = WinHttpConnect(session.handle, host.c_str(), parts.nPort, 0);
    const auto wide_method = fs::u8path(method).wstring();
    request.handle =
        WinHttpOpenRequest(connection.handle, wide_method.c_str(), resource.c_str(), nullptr,
                           WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                           parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0);
    if (!request.handle)
      throw std::runtime_error("cannot create HTTP request");
    std::wstring headers;
    if (const auto* h = options.find("headers"))
      for (const auto& entry : h->as_object()) {
        valid_header(entry.first);
        valid_header(entry.second.as_string());
        headers += fs::u8path(entry.first + ": " + entry.second.as_string() + "\r\n").wstring();
      }
    if (body.size() > DWORD(-1))
      throw std::runtime_error("HTTP request body too large");
    if (!WinHttpSendRequest(request.handle, headers.c_str(), DWORD(headers.size()),
                            body.empty() ? WINHTTP_NO_REQUEST_DATA : const_cast<char*>(body.data()),
                            DWORD(body.size()), DWORD(body.size()), 0) ||
        !WinHttpReceiveResponse(request.handle, nullptr))
      throw std::runtime_error("HTTP request failed (Windows error " +
                               std::to_string(GetLastError()) + ")");
    DWORD code = 0, length = sizeof(code);
    if (!WinHttpQueryHeaders(request.handle, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             nullptr, &code, &length, nullptr))
      throw std::runtime_error("cannot read HTTP status");
    status = int(code);
    length = 0;
    WinHttpQueryHeaders(request.handle, WINHTTP_QUERY_RAW_HEADERS_CRLF, nullptr, nullptr, &length,
                        nullptr);
    if (length) {
      std::wstring raw(length / sizeof(wchar_t), L'\0');
      if (WinHttpQueryHeaders(request.handle, WINHTTP_QUERY_RAW_HEADERS_CRLF, nullptr, raw.data(),
                              &length, nullptr)) {
        raw.resize(wcslen(raw.c_str()));
        std::istringstream lines(fs::path(raw).u8string());
        std::string line;
        while (std::getline(lines, line))
          transfer.header(line);
      }
    }
    transfer.accept(status);
    char chunk[65536];
    DWORD received;
    for (;;) {
      transfer.tick();
      if (!WinHttpReadData(request.handle, chunk, sizeof(chunk), &received))
        throw std::runtime_error("HTTP response read failed");
      if (!received)
        break;
      transfer.write(chunk, received);
    }
#else
    static const int initialized = [] {
      return curl_global_init(CURL_GLOBAL_DEFAULT);
    }();
    if (initialized != CURLE_OK)
      throw std::runtime_error("cannot initialize HTTP client");

    struct Curl {
      CURL* handle = curl_easy_init();
      curl_slist* headers = nullptr;

      ~Curl() {
        if (headers)
          curl_slist_free_all(headers);
        if (handle)
          curl_easy_cleanup(handle);
      }
    } curl;

    if (!curl.handle)
      throw std::runtime_error("cannot create HTTP client");
    curl_easy_setopt(curl.handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.handle, CURLOPT_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
    curl_easy_setopt(curl.handle, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTP | CURLPROTO_HTTPS);
    curl_easy_setopt(curl.handle, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl.handle, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl.handle, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.handle, CURLOPT_TIMEOUT_MS,
                     long(std::min<uint64_t>(transfer.timeout, LONG_MAX)));
    curl_easy_setopt(curl.handle, CURLOPT_USERAGENT, "slopfab-workflows/1");
    curl_easy_setopt(curl.handle, CURLOPT_CUSTOMREQUEST, method.c_str());
    if (method == "HEAD")
      curl_easy_setopt(curl.handle, CURLOPT_NOBODY, 1L);
    if (!body.empty()) {
      curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDS, body.data());
      curl_easy_setopt(curl.handle, CURLOPT_POSTFIELDSIZE_LARGE, curl_off_t(body.size()));
    }
    if (const auto* h = options.find("headers"))
      for (const auto& entry : h->as_object()) {
        valid_header(entry.first);
        valid_header(entry.second.as_string());
        curl.headers = curl_slist_append(curl.headers,
                                         (entry.first + ": " + entry.second.as_string()).c_str());
      }
    curl_easy_setopt(curl.handle, CURLOPT_HTTPHEADER, curl.headers);
    curl_easy_setopt(curl.handle, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(
        curl.handle, CURLOPT_WRITEFUNCTION, +[](char* p, size_t s, size_t n, void* data) -> size_t {
          auto& t = *static_cast<Transfer*>(data);
          try {
            t.write(p, s * n);
            return s * n;
          } catch (...) {
            t.failure = std::current_exception();
            return 0;
          }
        });
    curl_easy_setopt(curl.handle, CURLOPT_HEADERDATA, &transfer);
    curl_easy_setopt(
        curl.handle, CURLOPT_HEADERFUNCTION,
        +[](char* p, size_t s, size_t n, void* data) -> size_t {
          auto& t = *static_cast<Transfer*>(data);
          try {
            t.header(std::string(p, s * n));
            return s * n;
          } catch (...) {
            t.failure = std::current_exception();
            return 0;
          }
        });
    curl_easy_setopt(curl.handle, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl.handle, CURLOPT_XFERINFODATA, &transfer);
    curl_easy_setopt(
        curl.handle, CURLOPT_XFERINFOFUNCTION,
        +[](void* data, curl_off_t, curl_off_t, curl_off_t, curl_off_t) -> int {
          auto& t = *static_cast<Transfer*>(data);
          try {
            t.tick();
            return 0;
          } catch (...) {
            t.failure = std::current_exception();
            return 1;
          }
        });
    const auto rc = curl_easy_perform(curl.handle);
    if (transfer.failure)
      std::rethrow_exception(transfer.failure);
    if (rc != CURLE_OK)
      throw std::runtime_error(curl_easy_strerror(rc));
    long code = 0;
    curl_easy_getinfo(curl.handle, CURLINFO_RESPONSE_CODE, &code);
    status = int(code);
    transfer.accept(status);
#endif
    transfer.tick();
    if (!destination.empty()) {
      if (status < 200 || status >= 300)
        throw std::runtime_error("download returned HTTP " + std::to_string(status));
      transfer.file.close();
      if (!transfer.file)
        throw std::runtime_error("cannot finish download");
      publish_http_download(temporary, path, overwrite);
    }
    return Value(Object{{"status", num(status)},
                        {"ok", Value(status >= 200 && status < 300)},
                        {"headers", Value(transfer.headers)},
                        {"body", str(transfer.body)},
                        {"bytes", num(double(transfer.bytes))},
                        {"path", str(destination.empty() ? "" : path.u8string())}});
  } catch (...) {
    transfer.file.close();
    if (!temporary.empty()) {
      std::error_code error;
      fs::remove(temporary, error);
    }
    throw;
  }
}
}
