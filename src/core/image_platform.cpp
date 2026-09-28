// Reference-image decoding through the platform's own image stack.
//
// This is the decoder a build without FFmpeg uses. Linux uses libpng and
// libjpeg so ordinary reference images also work without an FFmpeg runtime.
//
// Windows has a full image stack in the OS: WIC reads PNG, JPEG, BMP, GIF,
// TIFF, DDS and — with the Store codecs installed — HEIF and WebP, and it is
// already present on every machine this project targets. Nothing is shipped
// and nothing is linked that is not part of Windows.
//
#include "slopfab/image.h"

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

namespace slopfab {
namespace {

std::runtime_error wic_error(const std::string& path, const std::string& reason, HRESULT hr) {
  char code[32];
  std::snprintf(code, sizeof(code), " (hr=0x%08lX)", static_cast<unsigned long>(hr));
  return std::runtime_error("reference image '" + path + "': " + reason + code);
}

// Minimal COM pointer. <wrl/client.h> would do this better but is not present
// in every toolchain that can build this project, and the need here is three
// pointers deep.
template <typename T> struct ComPtr {
  T* p = nullptr;

  ~ComPtr() {
    if (p != nullptr)
      p->Release();
  }

  ComPtr() = default;
  ComPtr(const ComPtr&) = delete;
  ComPtr& operator=(const ComPtr&) = delete;

  T** put() {
    return &p;
  }

  T* operator->() const {
    return p;
  }
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
    if (owns_)
      CoUninitialize();
  }

  ComScope(const ComScope&) = delete;
  ComScope& operator=(const ComScope&) = delete;

  bool usable() const {
    return SUCCEEDED(hr_) || hr_ == RPC_E_CHANGED_MODE;
  }

  HRESULT hr() const {
    return hr_;
  }

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
  if (utf8.empty())
    throw std::runtime_error("reference image: empty path");
  const int needed = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(),
                                         static_cast<int>(utf8.size()), nullptr, 0);
  if (needed <= 0) {
    throw wic_error(path, "path is not valid UTF-8", HRESULT_FROM_WIN32(GetLastError()));
  }
  std::wstring wide(static_cast<size_t>(needed), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.c_str(), static_cast<int>(utf8.size()),
                      &wide[0], needed);
  return wide;
}

} // namespace

RGBImage load_platform_image(const std::string& path) {
  ComScope com;
  if (!com.usable())
    throw wic_error(path, "cannot initialise COM", com.hr());

  ComPtr<IWICImagingFactory> factory;
  HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                IID_PPV_ARGS(factory.put()));
  if (FAILED(hr))
    throw wic_error(path, "cannot create the WIC imaging factory", hr);

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
  if (FAILED(hr))
    throw wic_error(path, "cannot read the first frame", hr);

  // Whatever the file's own pixel format is — palettised, CMYK, 16-bit,
  // premultiplied — WIC converts it to packed 24-bit RGB here, so everything
  // downstream sees one layout. Alpha is dropped against black rather than
  // carried: the pipeline conditions on colour and has nowhere to put it.
  ComPtr<IWICFormatConverter> converter;
  hr = factory->CreateFormatConverter(converter.put());
  if (FAILED(hr))
    throw wic_error(path, "cannot create a WIC format converter", hr);
  hr = converter->Initialize(frame.p, GUID_WICPixelFormat24bppRGB, WICBitmapDitherTypeNone, nullptr,
                             0.0, WICBitmapPaletteTypeCustom);
  if (FAILED(hr))
    throw wic_error(path, "cannot convert this image to 24-bit RGB", hr);

  UINT width = 0;
  UINT height = 0;
  hr = converter->GetSize(&width, &height);
  if (FAILED(hr))
    throw wic_error(path, "cannot read the image size", hr);
  if (width == 0 || height == 0)
    throw wic_error(path, "image has a zero dimension", E_FAIL);

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
  if (FAILED(hr))
    throw wic_error(path, "cannot copy the decoded pixels", hr);
  return image;
}

} // namespace slopfab

#elif defined(__linux__)

#include <csetjmp>
#include <limits>
#include <memory>
#include <utility>

#include <jpeglib.h>
#include <png.h>

namespace slopfab {
namespace {

size_t checked_image_size(uint64_t width, uint64_t height, const std::string& path) {
  if (width == 0 || height == 0 || width > std::numeric_limits<int>::max() ||
      height > std::numeric_limits<int>::max() || width * height > 0xFFFFFFFFull / 3u) {
    throw std::runtime_error("reference image '" + path + "': invalid or oversized dimensions");
  }
  return static_cast<size_t>(width * height * 3u);
}

RGBImage load_png(const std::string& path) {
  struct PngImage {
    png_image image{};
    PngImage() { image.version = PNG_IMAGE_VERSION; }
    ~PngImage() { png_image_free(&image); }
  } state;
  auto& png = state.image;
  if (!png_image_begin_read_from_file(&png, path.c_str()))
    throw std::runtime_error("reference image '" + path + "': " + png.message);
  const size_t size = checked_image_size(png.width, png.height, path);
  png.format = PNG_FORMAT_RGB;
  RGBImage image{static_cast<int>(png.width), static_cast<int>(png.height),
                 std::vector<uint8_t>(size)};
  const png_color background{0, 0, 0};
  if (!png_image_finish_read(&png, &background, image.pixels.data(), 0, nullptr))
    throw std::runtime_error("reference image '" + path + "': " + png.message);
  return image;
}

struct JpegError {
  jpeg_error_mgr manager{};
  std::jmp_buf jump;
  char message[JMSG_LENGTH_MAX]{};
};

void jpeg_failure(j_common_ptr info) {
  auto* error = reinterpret_cast<JpegError*>(info->err);
  info->err->format_message(info, error->message);
  std::longjmp(error->jump, 1);
}

// State lives on the heap: longjmp must neither skip C++ destructors nor
// invalidate automatic variables changed after setjmp. The owning pointer is
// created before setjmp and never modified; it releases every resource when
// the error branch throws, including partially decoded image memory.
struct JpegState {
  jpeg_decompress_struct info{};
  JpegError error;
  FILE* file = nullptr;
  bool created = false;
  RGBImage image;
  ~JpegState() {
    if (created)
      jpeg_destroy_decompress(&info);
    if (file != nullptr)
      std::fclose(file);
  }
};

RGBImage load_jpeg(const std::string& path) {
  const auto state = std::make_unique<JpegState>();
  state->file = std::fopen(path.c_str(), "rb");
  if (state->file == nullptr)
    throw std::runtime_error("reference image '" + path + "': cannot open file");
  state->info.err = jpeg_std_error(&state->error.manager);
  state->error.manager.error_exit = jpeg_failure;
  if (setjmp(state->error.jump))
    throw std::runtime_error("reference image '" + path + "': " + state->error.message);
  state->created = true;
  jpeg_create_decompress(&state->info);
  jpeg_stdio_src(&state->info, state->file);
  jpeg_read_header(&state->info, TRUE);
  // Grayscale and YCbCr inputs both become the same packed RGB layout.
  state->info.out_color_space = JCS_RGB;
  jpeg_start_decompress(&state->info);
  const size_t size = checked_image_size(state->info.output_width,
                                          state->info.output_height, path);
  state->image.width = static_cast<int>(state->info.output_width);
  state->image.height = static_cast<int>(state->info.output_height);
  state->image.pixels.resize(size);
  const size_t stride = static_cast<size_t>(state->image.width) * 3;
  while (state->info.output_scanline < state->info.output_height) {
    JSAMPROW row = state->image.pixels.data() + state->info.output_scanline * stride;
    jpeg_read_scanlines(&state->info, &row, 1);
  }
  jpeg_finish_decompress(&state->info);
  return std::move(state->image);
}

} // namespace

RGBImage load_platform_image(const std::string& path) {
  // Detect the file contents, allowing extensionless files and Unicode paths.
  const auto close_file = [](FILE* file) { std::fclose(file); };
  const auto file = std::unique_ptr<FILE, decltype(close_file)>(
      std::fopen(path.c_str(), "rb"), close_file);
  if (!file)
    throw std::runtime_error("reference image '" + path + "': cannot open file");
  unsigned char signature[8]{};
  const size_t count = std::fread(signature, 1, sizeof(signature), file.get());
  if (count == sizeof(signature) && png_sig_cmp(signature, 0, sizeof(signature)) == 0)
    return load_png(path);
  if (count >= 2 && signature[0] == 0xFF && signature[1] == 0xD8)
    return load_jpeg(path);
  throw std::runtime_error("reference image '" + path +
                           "': Linux image decoder supports PNG and JPEG; "
                           "use PPM or build with -DSLOPFAB_WITH_FFMPEG=ON for other formats");
}

} // namespace slopfab

#else

namespace slopfab {

RGBImage load_platform_image(const std::string& path) {
  throw std::runtime_error(
      "reference image '" + path +
      "': this build has no FFmpeg and this platform has no image decoder, so only binary PPM "
      "(P6) can be read; convert the image or build with -DSLOPFAB_WITH_FFMPEG=ON");
}

} // namespace slopfab

#endif // _WIN32
