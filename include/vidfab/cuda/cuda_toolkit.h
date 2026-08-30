#pragma once

#include <string>
#include <vector>

namespace vidfab::cuda {

struct CudaToolkitCandidate {
  int major = 0;
  std::wstring bin;
};

inline std::wstring cuda_version_request(const std::wstring& explicit_request,
                                         const std::wstring& environment) {
  if (!explicit_request.empty()) return explicit_request;
  if (!environment.empty()) return environment;
  return L"auto";
}

inline const CudaToolkitCandidate* select_cuda_toolkit(
    const std::wstring& requested,
    const std::vector<CudaToolkitCandidate>& candidates) {
  for (const CudaToolkitCandidate& candidate : candidates) {
    if (candidate.bin.empty()) continue;
    if (requested == L"auto" || requested == std::to_wstring(candidate.major))
      return &candidate;
  }
  return nullptr;
}

inline bool cuda_version_matches_linked_toolkit(const std::wstring& requested,
                                                int linked_major) {
  return requested == L"auto" || requested == std::to_wstring(linked_major);
}

}  // namespace vidfab::cuda
