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
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_add.comp -o tensor_add.raw.spv
python tools/add_spirv_float_controls.py tensor_add.raw.spv src/vulkan/tensor_add.comp.spv src/vulkan/tensor_add_denorm.comp.spv
```

Expected SHA-256 digests (also pinned by CMake):

```text
tensor_add.comp      4378E3EDC139935EB4F62F64F7934F68242DF5BC07DA77141EDC0D6F1B34BB73
tensor_add.comp.spv          0E52BC03EED7D86E3254489E0F18B491C54857700BAFF2F6B19D89E70DE9B1BE
tensor_add_denorm.comp.spv   5909864E52688E5ABF9F38765901B693F32EDD995942925EE3A690AC6A5BC12E
```

The deterministic postprocessor adds explicit fp32 signed-zero/Inf/NaN and
round-to-nearest-even execution modes. The denorm variant additionally adds
`DenormPreserve`; it is selected only when the queried Vulkan 1.2 float-control
properties permit that mode. Thus CMake hashes exactly the modules executed by
the driver, not an untracked runtime transformation.
