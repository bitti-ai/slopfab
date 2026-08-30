#pragma once

#include <string>
#include <vector>

namespace vidfab {

struct CudaLaunchCandidate {
  int major = 0;
  std::wstring backend;
  std::wstring toolkit_bin;
};

inline std::wstring cuda_version_request(const std::wstring& cli,
                                         const std::wstring& environment) {
  if (!cli.empty()) return cli;
  if (!environment.empty()) return environment;
  return L"auto";
}

inline const CudaLaunchCandidate* select_cuda_launch(
    const std::wstring& requested,
    const std::vector<CudaLaunchCandidate>& candidates) {
  for (const CudaLaunchCandidate& candidate : candidates) {
    if (candidate.backend.empty() || candidate.toolkit_bin.empty()) continue;
    if (requested == L"auto" || requested == std::to_wstring(candidate.major))
      return &candidate;
  }
  return nullptr;
}

// Windows command-line quoting as consumed by CommandLineToArgvW/the MSVC
// runtime. Empty strings require an explicit pair of quotes.
inline std::wstring quote_windows_argument(const std::wstring& value) {
  if (value.empty()) return L"\"\"";
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

inline std::wstring cuda_launch_command(const std::wstring& backend,
                                        const std::vector<std::wstring>& arguments) {
  std::wstring command = quote_windows_argument(backend);
  for (const std::wstring& argument : arguments)
    command += L" " + quote_windows_argument(argument);
  return command;
}

}  // namespace vidfab
