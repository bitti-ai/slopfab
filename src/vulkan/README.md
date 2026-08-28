# Embedded output shader

`rgb_to_yuv.comp` converts planar fp32 RGB into packed BT.709 limited-range
YUV420 values. It is used only for output colour conversion; model inference
remains CUDA.

The checked-in SPIR-V was produced with Khronos glslang 16.5.0:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/rgb_to_yuv.comp -o src/vulkan/rgb_to_yuv.comp.spv
```

Expected SHA-256 digests (also checked by CMake before embedding):

```text
rgb_to_yuv.comp      77FA686E8C2CCCA0B62551211DEFB4D47DD941DD617627EFCE56DF35637D2C37
rgb_to_yuv.comp.spv  C4CEE251C15F54371387D2DBACB4637C76C0DBC5DC6C0663C1E1269C703E71E6
```

The runtime and build do not require a shader compiler or Vulkan SDK.
