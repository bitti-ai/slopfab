# ThorVG for Slopfab Motions

Upstream: https://github.com/thorvg/thorvg

Version: **v1.1.3**, commit `589bc4120f2097fb1b135e552c2ef39f6581612b`.
License: MIT; see LICENSE and retained source license headers.

The unmodified upstream source and headers are vendored for offline builds.
Slopfab builds only the common renderer, portable CPU engine, and SVG, raw, PNG,
TTF/OTF loaders via `cmake/ThorVG.cmake`. GPU backends, Lottie, media loaders,
savers and bindings are not linked. The generated config selects synchronous CPU
rendering and no SIMD requirement. The upstream Meson files are reference only.

Only the `slopfab` executable links the static target. The core and C API shared
library must not acquire a ThorVG dependency. Built-in PNG decoding includes
LodePNG; its license header is preserved in `src/loaders/png/tvgLodePng.h`.
