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

`tensor_ops.comp` is the bounded neural-primitive module. One cached pipeline
provides fp32 add/add-bias, exact fp32-to/from-fp16 and bf16 conversion, fp32
transpose, bounds-safe trusted-index gather/scatter, head-major fp32 to
token-major bf16, and fp32 depth-to-space. Calls use the reusable batched tensor
path; exact copies use transfer commands and need no shader. The checked-in
modules were produced with the same Khronos glslang 16.5.0:

```text
glslang -V --target-env vulkan1.2 -S comp src/vulkan/tensor_ops.comp -o tensor_ops.raw.spv
python tools/add_spirv_float_controls.py tensor_ops.raw.spv src/vulkan/tensor_ops.comp.spv src/vulkan/tensor_ops_denorm.comp.spv
```

Expected SHA-256 digests (also pinned by CMake):

```text
tensor_ops.comp              0BCE01AE93BD3ABA5EDD396B21AF46621BEBFB60309E24BE2440130198BD424C
tensor_ops.comp.spv          A7D3450A9FE867257D64FF9E8A6824DAA351E36F0AA0E1F733EC13EA8C59926A
tensor_ops_denorm.comp.spv   2EC0F3B99AB49E055CACFF61DE3E15F6CEAD39F93879DAB24BE2EC6C9AC8B807
```

The deterministic postprocessor adds explicit fp32 signed-zero/Inf/NaN and
round-to-nearest-even execution modes. The denorm variant additionally adds
`DenormPreserve`; it is selected only when the queried Vulkan 1.2 float-control
properties permit that mode. fp16 and bf16 narrowing use CUDA's canonical NaN
value (`0x7fff`) and round-to-nearest-even. Thus CMake hashes exactly the
modules executed by the driver, not an untracked runtime transformation.
