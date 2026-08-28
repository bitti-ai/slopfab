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

## Tensor primitive shader

`tensor_add.comp` is the first neural-backend primitive. It performs one
ordered fp32 addition per element and is used through the reusable batched
tensor command path; copies use Vulkan transfer commands and need no shader.
The checked-in module was produced with the same Khronos glslang 16.5.0:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_add.comp -o src/vulkan/tensor_add.comp.spv
```

Expected SHA-256 digests (also pinned by CMake):

```text
tensor_add.comp      4378E3EDC139935EB4F62F64F7934F68242DF5BC07DA77141EDC0D6F1B34BB73
tensor_add.comp.spv  73F88EB5018B9CACA01CD80A60E5508A72C1A191FDC58318F387F157D73DD15B
```
