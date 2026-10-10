# Test coverage

The large backend, transformer, and encoder suites are split by operator or
stage. Shared implementation details live in `detail/*_fixture.h`; test cases
remain registered exactly once. Keep fixture helpers private to their translation
unit rather than introducing externally linked definitions in those headers.

The split suites attach a coverage category to each case:

| Category | Purpose |
|---|---|
| `synthetic` | Generated inputs and numerical/operator contracts |
| `checkpoint` | Individual operators or stages using real checkpoint tensors |
| `integration` | Full model paths, replay captures, or multiple stages |
| `benchmark` | Opt-in timing and performance measurements |

CTest registers each available category separately while preserving the existing
synthetic suite names (`vulkan`, `kernels`, `tensor_backends`). For example:

```powershell
ctest --test-dir build --build-config Release -L synthetic --output-on-failure
ctest --test-dir build --build-config Release -L checkpoint --output-on-failure
ctest --test-dir build --build-config Release -L integration --output-on-failure
```

Configure with `-DSLOPFAB_TEST_BENCHMARKS=ON` to register benchmark suites, then
select `-L benchmark`. GPU suites use a shared CTest resource lock to avoid
competing for device memory when CTest runs in parallel.

For a direct executable invocation, `SLOPFAB_TEST_CATEGORY` selects one category
and `SLOPFAB_TEST_FILTER` selects case names by substring. Direct benchmark runs
also require `SLOPFAB_RUN_BENCHMARKS=1`. Individual expensive tests retain their
existing opt-in variables and report which one is missing.

Missing fixtures, insufficient VRAM, unsupported hardware, and disabled opt-ins
produce explicit `SKIP` messages. A suite with no checks and at least one skip
returns 77, which CTest displays as skipped. A filter matching no cases returns
an error. Mixed runs retain skip counts in their summaries; passing the remaining
checks does not establish coverage for skipped cases. Existing deferred checks
remain separately reported known defects.

To check reference-image encoding with a Comfy INT8 ConvRot VAE on CUDA and
Vulkan, run the real encoder regression from the repository root:

```powershell
$env:SLOPFAB_KEYFRAME_ENCODER_REAL = '1'
$env:SLOPFAB_INT8_VIDEO_VAE = 'path/to/minimax_h3_video_vae_int8_convrot.safetensors'
$env:SLOPFAB_TEST_FILTER = 'cuda_vulkan_keyframe_encoder_int8_convrot'
build/Release/slopfab_tensor_backend_tests.exe
```

This checks the mixed FP16/FP32 encoder against the FP16 checkpoint's golden
output, exact backend agreement, and persistent GPU memory accounting.

The host `unit` suite and the C API suite keep their existing registration and
carry suite-level labels. Their internal historical cases have not all been
reclassified by fixture requirements.
