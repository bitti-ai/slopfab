// Reference-image decoding through the platform's own image stack.
//
// This is the decoder a build without FFmpeg uses. It exists because the
// alternative in that configuration is PPM only, and a host that wants to
// condition a run on a PNG should not have to transcode it first — while
// pulling in libpng, libjpeg and their transitive dependencies to avoid that
// would trade one runtime dependency for several build-time ones.
//
// Windows has a full image stack in the OS: WIC reads PNG, JPEG, BMP, GIF,
// TIFF, DDS and — with the Store codecs installed — HEIF and WebP, and it is
// already present on every machine this project targets. Nothing is shipped
// and nothing is linked that is not part of Windows.
//
// Elsewhere there is no equivalent single API, so this refuses rather than
// pretending: a Linux build without FFmpeg reads PPM, and says so.
#include "vidfab/image.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#if defined(_WIN32)

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <objbase.h>
#include <wincodec.h>

namespace vidfab {
namespace {

std::runtime_error wic_error(const std::string& path, const std::string& reason, HRESULT hr) {
  char code[32];
  std::snprintf(code, sizeof(code), " (hr=0x%08lX)", static_cast<unsigned long>(hr));
  return std::runtime_error("reference image '" + path + "': " + reason + code);
}

// Minimal COM pointer. <wrl/client.h> would do this better but is not present
// in every toolchain that can build this project, and the need here is three
// pointers deep.
template <typename T>
struct ComPtr {
  T* p = nullptr;
  ~ComPtr() {
    if (p != nullptr) p->Release();
  }
  ComPtr() = default;
  ComPtr(const ComPtr&) = delete;
  ComPtr& operator=(const ComPtr&) = delete;
  T** put() { return &p; }
  T* operator->() const { return p; }
};

// Balances whatever CoInitializeEx actually did, which is the part that is
// easy to get wrong in a library: the host may already have put this thread
// into an apartment, and it may be a different one than we asked for.
//
//   S_OK              we initialised it; we must uninitialise.
//   S_FALSE           already initialised on this thread, refcount raised;
//                     we must still uninitialise to lower it.
//   RPC_E_CHANGED_MODE  the thread is in the other apartment model. The
//                     refcount was *not* raised and calling CoUninitialize
//                     here would decrement someone else's — so we must not,
//                     and WIC works from either apartment anyway.
class ComScope {
 public:
  ComScope() {
    hr_ = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    owns_ = SUCCEEDED(hr_);
  }
  ~ComScope() {
    if (owns_) CoUninitialize();
  }
  ComScope(const ComScope&) = delete;
  ComScope& operator=(const ComScope&) = delete;
  bool usable() const { return SUCCEEDED(hr_) || hr_ == RPC_E_CHANGED_MODE; }
  HRESULT hr() const { return hr_; }

 private:
  HRESULT hr_ = S_OK;
  bool owns_ = false;
};

// MB_ERR_INVALID_CHARS is what makes the check below mean anything. Without
// it, MultiByteToWideChar substitutes U+FFFD for malformed input and returns a
// positive count, so a path in the active code page rather than UTF-8 — the
// mistake this diagnoses, and an easy one for a host to make — would be
// silently mangled and then reported as a missing file.
std::wstring widen(const std::string& utf8, const std::string& path) {
  if (utf8.empty()) throw std::runtime_error("reference image: empty path");
  const int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(),
                                         static_cast<int>(utf8.size()), nullptr, 0);
  if (needed <= 0) {
    throw wic_error(path, "path is not valid UTF-8", HRESULT_FROM_WIN32(GetLastError()));
  }
  std::wstring wide(static_cast<size_t>(needed), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(),
                      static_cast<int>(utf8.size()), &wide[0], needed);
  return wide;
}

}  // namespace

RGBImage load_platform_image(const std::string& path) {
  ComScope com;
  if (!com.usable()) throw wic_error(path, "cannot initialise COM", com.hr());

  ComPtr<IWICImagingFactory> factory;
  HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(factory.put()));
  if (FAILED(hr)) throw wic_error(path, "cannot create the WIC imaging factory", hr);

  const std::wstring wide = widen(path, path);
  ComPtr<IWICBitmapDecoder> decoder;
  hr = factory->CreateDecoderFromFilename(wide.c_str(), nullptr, GENERIC_READ,
                                          WICDecodeMetadataCacheOnDemand, decoder.put());
  if (FAILED(hr)) {
    // The common cases — a missing file and a format with no installed codec —
    // are worth separating, because the second is fixable by the user and the
    // first is a typo.
    if (hr == HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND) ||
        hr == HRESULT_FROM_WIN32(ERROR_PATH_NOT_FOUND)) {
      throw wic_error(path, "cannot open file", hr);
    }
    if (hr == WINCODEC_ERR_COMPONENTNOTFOUND) {
      throw wic_error(path, "no installed Windows codec reads this format", hr);
    }
    throw wic_error(path, "cannot decode this file", hr);
  }

  // Frame 0. An animation or a multi-page TIFF contributes its first image,
  // which is the same rule the FFmpeg path follows.
  ComPtr<IWICBitmapFrameDecode> frame;
  hr = decoder->GetFrame(0, frame.put());
  if (FAILED(hr)) throw wic_error(path, "cannot read the first frame", hr);

  // Whatever the file's own pixel format is — palettised, CMYK, 16-bit,
  // premultiplied — WIC converts it to packed 24-bit RGB here, so everything
  // downstream sees one layout. Alpha is dropped against black rather than
  // carried: the pipeline conditions on colour and has nowhere to put it.
  ComPtr<IWICFormatConverter> converter;
  hr = factory->CreateFormatConverter(converter.put());
  if (FAILED(hr)) throw wic_error(path, "cannot create a WIC format converter", hr);
  hr = converter->Initialize(frame.p, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone,
                             nullptr, 0.0, WICBitmapPaletteTypeCustom);
  if (FAILED(hr)) throw wic_error(path, "cannot convert this image to 24-bit RGB", hr);

  UINT width = 0;
  UINT height = 0;
  hr = converter->GetSize(&width, &height);
  if (FAILED(hr)) throw wic_error(path, "cannot read the image size", hr);
  if (width == 0 || height == 0) throw wic_error(path, "image has a zero dimension", E_FAIL);

  // Checked before the multiplication below rather than after it: at 3 bytes
  // per pixel a 2 GB image would otherwise wrap the stride computation.
  const uint64_t stride = static_cast<uint64_t>(width) * 3u;
  const uint64_t total = stride * height;
  if (stride > 0xFFFFFFFFull || total > 0xFFFFFFFFull) {
    throw wic_error(path, "image is too large to decode", E_OUTOFMEMORY);
  }

  RGBImage image;
  image.width = static_cast<int>(width);
  image.height = static_cast<int>(height);
  image.pixels.resize(static_cast<size_t>(total));
  hr = converter->CopyPixels(nullptr, static_cast<UINT>(stride), static_cast<UINT>(total),
                             image.pixels.data());
  if (FAILED(hr)) throw wic_error(path, "cannot copy the decoded pixels", hr);
  return image;
}

}  // namespace vidfab

#else  // !_WIN32

namespace vidfab {

RGBImage load_platform_image(const std::string& path) {
  throw std::runtime_error(
      "reference image '" + path +
      "': this build has no FFmpeg and this platform has no image decoder, so only binary PPM "
      "(P6) can be read; convert the image or build with -DVIDFAB_WITH_FFMPEG=ON");
}

}  // namespace vidfab

#endif  // _WIN32
