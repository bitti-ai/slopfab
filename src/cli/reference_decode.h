#pragma once
#include <string>
#include "slopfab/reference_media.h"

namespace slopfab::cli {
// CLI-only file adapter. Uses ffprobe/ffmpeg executables; never linked into
// slopfab.dll. Video includes its first audio stream when present.
ReferenceMedia decode_reference_file(const std::string& path, bool video,
                                      const char* executable);
}
