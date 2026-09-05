# Vulkan headers

This directory contains `vulkan_core.h`, `vk_platform.h`, and the core header's
generated `vk_video` dependencies from the official
Khronos Vulkan-Headers repository, pinned to SDK tag `vulkan-sdk-1.4.321.0`.
Only the implementation of the optional `slopfab_vulkan` target includes them;
slopfab's public Vulkan wrapper does not expose SDK types.

The headers are used solely for compile-time ABI declarations. At runtime,
slopfab dynamically opens the operating system's Vulkan loader (`vulkan-1.dll`,
`libvulkan.so.1`, or `libvulkan.1.dylib`). No SDK files are needed at runtime.

The upstream license is reproduced in `LICENSE.md`.
