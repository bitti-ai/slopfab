# Vulkan test shaders

`affine.comp.spv` is generated from `affine.comp` with the official Khronos
glslang 16.5.0 release:

```text
glslang -V --target-env vulkan1.2 -S comp affine.comp -o affine.comp.spv
```

The SPIR-V is checked in so configuring, building, and running slopfab needs no
shader compiler or Vulkan SDK. Regeneration is an explicit developer action;
the unit test loads the binary directly and the runtime validates its magic,
declared bindings, and local workgroup size before pipeline creation.

Pinned SHA-256 values (also checked by CMake, with source line endings
normalized to LF):

- `affine.comp`: `5ff2705b9c300adbdb476d393226f0dda0a3e23dd7b1eda9ac8095a9165345cd`
- `affine.comp.spv`: `75c8343ac68d52af2c5908db040af9768f6abbce9df0a82268d131afbfc2ede5`
