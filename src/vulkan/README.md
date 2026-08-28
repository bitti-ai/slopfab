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
rgb_to_yuv.comp      71504BF881A3E7140AA8A4C0BAC1571679D8D173AE63F1C61D035EE4D88BB83A
rgb_to_yuv.comp.spv  746063A7EE311855BCA4FECB038ABC1E7C0835E67D18180133DA24154F759BBA
```

The runtime and build do not require a shader compiler or Vulkan SDK.
