// CUDA-free Windows launcher. It selects a toolkit-specific vidfab backend,
// prefixes that toolkit's bin directory to PATH, and forwards the remaining
// command line. It deliberately does not load nvcuda.dll: GPU architecture is
// selected by the backend fat binary after CUDA has initialized normally.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cwchar>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

bool file_exists(const std::wstring& path) {
  const DWORD attributes = GetFileAttributesW(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

std::wstring environment(const wchar_t* name) {
  const DWORD length = GetEnvironmentVariableW(name, nullptr, 0);
  if (length == 0) return {};
  std::wstring value(length, L'\0');
  GetEnvironmentVariableW(name, value.data(), length);
  value.resize(std::wcslen(value.c_str()));
  return value;
}

std::wstring executable_directory() {
  std::wstring path(32768, L'\0');
  const DWORD length = GetModuleFileNameW(nullptr, path.data(),
                                          static_cast<DWORD>(path.size()));
  if (length == 0 || length == path.size()) return {};
  path.resize(length);
  const size_t slash = path.find_last_of(L"\\/");
  return slash == std::wstring::npos ? L"." : path.substr(0, slash);
}

void add_root(std::vector<std::wstring>& roots, std::wstring root) {
  while (!root.empty() && (root.back() == L'\\' || root.back() == L'/'))
    root.pop_back();
  if (root.size() >= 4 && _wcsicmp(root.c_str() + root.size() - 4, L"\\bin") == 0)
    root.resize(root.size() - 4);
  if (!root.empty() &&
      std::find_if(roots.begin(), roots.end(), [&](const std::wstring& old) {
        return _wcsicmp(old.c_str(), root.c_str()) == 0;
      }) == roots.end())
    roots.push_back(std::move(root));
}

std::vector<std::wstring> toolkit_roots(int major) {
  std::vector<std::wstring> roots;
  add_root(roots, environment(major == 13 ? L"CUDA_PATH_V13_0" : L"CUDA_PATH_V12_8"));

  // Accept all CUDA_PATH_V12_x variables so a later CUDA 12 point release
  // needs no launcher update. Environment variables are process-local reads.
  const wchar_t* block = GetEnvironmentStringsW();
  if (block) {
    const std::wstring prefix = major == 13 ? L"CUDA_PATH_V13_" : L"CUDA_PATH_V12_";
    for (const wchar_t* entry = block; *entry; entry += std::wcslen(entry) + 1) {
      const wchar_t* equals = std::wcschr(entry, L'=');
      if (!equals) continue;
      std::wstring name(entry, equals);
      if (name.size() >= prefix.size() &&
          _wcsnicmp(name.c_str(), prefix.c_str(), prefix.size()) == 0)
        add_root(roots, equals + 1);
    }
    FreeEnvironmentStringsW(const_cast<wchar_t*>(block));
  }
  add_root(roots, environment(L"CUDA_PATH"));

  const std::wstring program_files = environment(L"ProgramFiles");
  if (!program_files.empty()) {
    if (major == 13) {
      for (int minor = 9; minor >= 0; --minor)
        add_root(roots, program_files + L"\\NVIDIA GPU Computing Toolkit\\CUDA\\v13." +
                            std::to_wstring(minor));
    } else {
      for (int minor = 9; minor >= 0; --minor)
        add_root(roots, program_files + L"\\NVIDIA GPU Computing Toolkit\\CUDA\\v12." +
                            std::to_wstring(minor));
    }
  }
  return roots;
}

std::wstring find_toolkit_bin(int major) {
  const std::wstring suffix = std::to_wstring(major) + L".dll";
  for (const std::wstring& root : toolkit_roots(major)) {
    const std::wstring bin = root + L"\\bin";
    if (file_exists(bin + L"\\cublas64_" + suffix) &&
        file_exists(bin + L"\\cublasLt64_" + suffix))
      return bin;
  }
  return {};
}

std::wstring quote(std::wstring value) {
  if (value.find_first_of(L" \t\"") == std::wstring::npos) return value;
  std::wstring result = L"\"";
  size_t slashes = 0;
  for (wchar_t ch : value) {
    if (ch == L'\\') {
      ++slashes;
    } else {
      if (ch == L'\"') result.append(slashes * 2 + 1, L'\\');
      else result.append(slashes, L'\\');
      slashes = 0;
      result += ch;
    }
  }
  result.append(slashes * 2, L'\\');
  result += L'\"';
  return result;
}

int fail(const std::wstring& message) {
  std::wcerr << L"vidfab: " << message << L'\n';
  return 1;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  std::wstring requested;
  std::vector<std::wstring> forwarded;
  for (int i = 1; i < argc; ++i) {
    const std::wstring argument = argv[i];
    constexpr wchar_t prefix[] = L"--cuda-version=";
    if (argument.rfind(prefix, 0) == 0) {
      requested = argument.substr(std::size(prefix) - 1);
    } else if (argument == L"--cuda-version") {
      if (++i >= argc) return fail(L"--cuda-version requires auto, 13, or 12");
      requested = argv[i];
    } else {
      forwarded.push_back(argument);
    }
  }
  if (requested.empty()) requested = environment(L"VIDFAB_CUDA_VERSION");
  if (requested.empty()) requested = L"auto";
  if (requested != L"auto" && requested != L"13" && requested != L"12")
    return fail(L"--cuda-version requires auto, 13, or 12");

  const std::wstring directory = executable_directory();
  if (directory.empty()) return fail(L"cannot determine launcher directory");
  const int order[] = {13, 12};
  int selected = 0;
  std::wstring backend;
  std::wstring toolkit_bin;
  for (int major : order) {
    if (requested != L"auto" && requested != std::to_wstring(major)) continue;
    const std::wstring candidate = directory + L"\\vidfab-cuda" +
                                   std::to_wstring(major) + L".exe";
    if (!file_exists(candidate)) continue;
    std::wstring bin = find_toolkit_bin(major);
    if (bin.empty()) continue;
    selected = major;
    backend = candidate;
    toolkit_bin = std::move(bin);
    break;
  }
  if (selected == 0) {
    return fail(L"no matching CUDA backend/toolkit found (looked for CUDA 13, then CUDA 12; "
                L"install the toolkit with cuBLAS or select --cuda-version=12|13)");
  }

  const std::wstring old_path = environment(L"PATH");
  const std::wstring child_path = toolkit_bin + (old_path.empty() ? L"" : L";" + old_path);
  if (!SetEnvironmentVariableW(L"PATH", child_path.c_str()))
    return fail(L"cannot prepare CUDA toolkit PATH");

  std::wstring command = quote(backend);
  for (const std::wstring& argument : forwarded) command += L" " + quote(argument);
  std::vector<wchar_t> mutable_command(command.begin(), command.end());
  mutable_command.push_back(L'\0');
  STARTUPINFOW startup{};
  startup.cb = sizeof(startup);
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(backend.c_str(), mutable_command.data(), nullptr, nullptr, TRUE, 0,
                      nullptr, nullptr, &startup, &process)) {
    return fail(L"cannot start CUDA " + std::to_wstring(selected) +
                L" backend (Windows error " + std::to_wstring(GetLastError()) + L")");
  }
  CloseHandle(process.hThread);
  WaitForSingleObject(process.hProcess, INFINITE);
  DWORD exit_code = 1;
  GetExitCodeProcess(process.hProcess, &exit_code);
  CloseHandle(process.hProcess);
  return static_cast<int>(exit_code);
}
