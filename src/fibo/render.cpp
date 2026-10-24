#include "runtime.cuh"
#include <algorithm>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
namespace slopfab::fibo {
RGBImage render_text(const std::string& text) {
#ifdef _WIN32
  if (text.size() > 16384)
    throw std::invalid_argument("Fibo text_render entry exceeds 16384 bytes");
  int count =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
  if (!count)
    throw std::invalid_argument("Fibo text_render is not UTF-8");
  std::wstring wide(count, L' ');
  MultiByteToWideChar(CP_UTF8, 0, text.data(), int(text.size()), wide.data(), count);
  for (auto& c : wide)
    if (c == L'\r' || c == L'\n' || c == L'\t')
      c = L' ';
  HDC dc = CreateCompatibleDC(nullptr);
  HFONT font =
      CreateFontW(-20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                  CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH, L"Arial");
  if (!dc || !font) {
    if (font)
      DeleteObject(font);
    if (dc)
      DeleteDC(dc);
    throw std::runtime_error("Fibo text rasterizer initialization failed");
  }
  HGDIOBJ old_font = SelectObject(dc, font);
  int width = 1, lines = (count + 255) / 256;
  for (int i = 0; i < count; i += 256) {
    SIZE size{};
    GetTextExtentPoint32W(dc, wide.data() + i, std::min(256, count - i), &size);
    width = std::max(width, int(size.cx));
  }
  width += 4;
  int height = lines * 28 + 4;
  BITMAPINFO info{};
  info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
  info.bmiHeader.biWidth = width;
  info.bmiHeader.biHeight = -height;
  info.bmiHeader.biPlanes = 1;
  info.bmiHeader.biBitCount = 32;
  info.bmiHeader.biCompression = BI_RGB;
  void* bits = nullptr;
  HBITMAP bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
  if (!bitmap) {
    SelectObject(dc, old_font);
    DeleteObject(font);
    DeleteDC(dc);
    throw std::runtime_error("Fibo text image allocation failed");
  }
  HGDIOBJ old_bitmap = SelectObject(dc, bitmap);
  PatBlt(dc, 0, 0, width, height, WHITENESS);
  SetBkMode(dc, TRANSPARENT);
  SetTextColor(dc, RGB(0, 0, 0));
  for (int i = 0; i < count; i += 256)
    TextOutW(dc, 2, 2 + i / 256 * 28, wide.data() + i, std::min(256, count - i));
  GdiFlush();
  RGBImage out{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
  auto src = static_cast<const uint8_t*>(bits);
  for (size_t i = 0; i < size_t(width) * height; ++i)
    for (int c = 0; c < 3; ++c)
      out.pixels[i * 3 + c] = src[i * 4 + 2 - c];
  SelectObject(dc, old_bitmap);
  SelectObject(dc, old_font);
  DeleteObject(bitmap);
  DeleteObject(font);
  DeleteDC(dc);
  return out;
#else
  throw std::invalid_argument(
      "Fibo text_render glyph images currently require Windows; plain edit instructions work on CUDA platforms");
#endif
}
}
