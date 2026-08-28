# Vulkan test shaders

`affine.comp.spv` is generated from `affine.comp` with the official Khronos
glslang 16.5.0 release:

```text
glslang -V --target-env vulkan1.2 -S comp affine.comp -o affine.comp.spv
```

The SPIR-V is checked in so configuring, building, and running vidfab needs no
shader compiler or Vulkan SDK. Regeneration is an explicit developer action;
the unit test loads the binary directly and the runtime validates its magic,
declared bindings, and local workgroup size before pipeline creation.
