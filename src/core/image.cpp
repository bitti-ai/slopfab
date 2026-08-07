#include "vidfab/image.h"

#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#endif

namespace vidfab {
namespace {

std::runtime_error image_error(const std::string& path, const std::string& reason) {
  return std::runtime_error("reference image '" + path + "': " + reason);
}

std::string ppm_token(std::istream& in) {
  std::string token;
  for (;;) {
    in >> std::ws;
    if (in.peek() != '#') break;
    in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  in >> token;
  return token;
}

RGBImage load_ppm(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw image_error(path, "cannot open file");
  if (ppm_token(in) != "P6") throw image_error(path, "not a binary PPM (P6)");
  RGBImage image;
  try {
    image.width = std::stoi(ppm_token(in));
    image.height = std::stoi(ppm_token(in));
    if (std::stoi(ppm_token(in)) != 255) throw image_error(path, "PPM max value must be 255");
  } catch (const std::invalid_argument&) {
    throw image_error(path, "invalid PPM header");
  } catch (const std::out_of_range&) {
    throw image_error(path, "PPM dimensions are out of range");
  }
  if (image.width <= 0 || image.height <= 0 ||
      static_cast<uint64_t>(image.width) * image.height >
          std::numeric_limits<size_t>::max() / 3) {
    throw image_error(path, "invalid image dimensions");
  }
  const int separator = in.get();
  if (separator == EOF || !std::isspace(static_cast<unsigned char>(separator))) {
    throw image_error(path, "missing whitespace after PPM header");
  }
  image.pixels.resize(static_cast<size_t>(image.width) * image.height * 3);
  in.read(reinterpret_cast<char*>(image.pixels.data()),
          static_cast<std::streamsize>(image.pixels.size()));
  if (in.gcount() != static_cast<std::streamsize>(image.pixels.size())) {
    throw image_error(path, "truncated pixel data");
  }
  return image;
}

#ifdef _WIN32
std::wstring widen_path(const std::string& path) {
  const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(),
                                    static_cast<int>(path.size()), nullptr, 0);
  if (n <= 0) throw image_error(path, "path is not valid UTF-8");
  std::wstring wide(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), static_cast<int>(path.size()),
                      wide.data(), n);
  return wide;
}

RGBImage load_wic(const std::string& path) {
  using Microsoft::WRL::ComPtr;
  const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  const bool uninitialize = SUCCEEDED(init);
  if (FAILED(init) && init != RPC_E_CHANGED_MODE) throw image_error(path, "cannot initialize COM");
  HRESULT hr;
  ComPtr<IWICImagingFactory> factory;
  hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                        IID_PPV_ARGS(&factory));
  ComPtr<IWICBitmapDecoder> decoder;
  if (SUCCEEDED(hr)) {
    const std::wstring wide = widen_path(path);
    hr = factory->CreateDecoderFromFilename(wide.c_str(), nullptr, GENERIC_READ,
                                            WICDecodeMetadataCacheOnDemand, &decoder);
  }
  ComPtr<IWICBitmapFrameDecode> frame;
  if (SUCCEEDED(hr)) hr = decoder->GetFrame(0, &frame);
  UINT width = 0, height = 0;
  if (SUCCEEDED(hr)) hr = frame->GetSize(&width, &height);
  if (SUCCEEDED(hr) && (width == 0 || height == 0 ||
                        static_cast<uint64_t>(width) * height >
                            std::numeric_limits<size_t>::max() / 3)) hr = E_INVALIDARG;
  ComPtr<IWICFormatConverter> converter;
  if (SUCCEEDED(hr)) hr = factory->CreateFormatConverter(&converter);
  if (SUCCEEDED(hr)) {
    hr = converter->Initialize(frame.Get(), GUID_WICPixelFormat24bppRGB,
                               WICBitmapDitherTypeNone, nullptr, 0.0,
                               WICBitmapPaletteTypeCustom);
  }
  RGBImage image;
  if (SUCCEEDED(hr)) {
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.pixels.resize(static_cast<size_t>(width) * height * 3);
    hr = converter->CopyPixels(nullptr, width * 3, static_cast<UINT>(image.pixels.size()),
                               image.pixels.data());
  }
  if (uninitialize) CoUninitialize();
  if (FAILED(hr)) throw image_error(path, "cannot decode image");
  return image;
}
#endif

}  // namespace

RGBImage load_reference_image(const std::string& path) {
  std::ifstream probe(path, std::ios::binary);
  char magic[2] = {};
  probe.read(magic, 2);
  if (!probe) throw image_error(path, "cannot open or read file");
  if (magic[0] == 'P' && magic[1] == '6') return load_ppm(path);
#ifdef _WIN32
  return load_wic(path);
#else
  throw image_error(path, "unsupported format; this build supports binary PPM (P6)");
#endif
}

}  // namespace vidfab
