# Official SM120 kernel AOT investigation

The released NVIDIA Sol-Attn SM120 implementation is Apache-2.0 Python CuTe
DSL, not CUDA C++. Its documented build/runtime requirements are:

- Python 3.10 or newer;
- PyTorch 2.10 or newer;
- CUDA 12.8 or newer;
- NVIDIA CuTe DSL / CUTLASS Python 4.5 or newer;
- `cuda-python`;
- contiguous BF16 `[batch, tokens, heads, 128]` tensors.

The implementation calls `cute.compile` on its first invocation with live
CuTe tensors, TMA copy atoms, and a CUDA stream. The resulting callable owns
the generated launch wrapper and TMA tensor-map argument construction. The
release does not provide a cubin, a C ABI, or an offline export script.

## Windows CUDA 13 result

Vidfab's supported build host is Windows with CUDA 13.0. Both matching package
lines were tested in a clean Python 3.10 virtual environment:

```text
pip install nvidia-cutlass-dsl==4.5.3 cuda-python==13.0.3
ERROR: No matching distribution found for nvidia-cutlass-dsl-libs-base==4.5.3

pip install nvidia-cutlass-dsl==4.7.0 cuda-python==13.3.1
ERROR: No matching distribution found for nvidia-cutlass-dsl-libs-cu12==4.7.0
```

The CuTe DSL binary packages have no Windows distribution. This machine also
has no WSL, Docker, or Podman environment in which the Linux generator can
run. Consequently the official source cannot currently be compiled here, and
checking in an untested generated cubin would not be a reproducible build.

## Native-port implications

The official SM120 schedule remains the executable design reference. A native
port must reproduce these mechanics rather than only its warp MMA operations:

- 128-thread M64/N64/D128 CTA;
- TMA descriptors and asynchronous Q/K/V/K-centroid/V-centroid pipelines;
- Q staged once and then register-resident;
- 64 centroid blocks routed together through a proxy MMA tile;
- CTA-local exact-index compaction;
- FP32 output MMA fragments kept in registers across routed groups;
- shared-memory aliasing in the output epilogue;
- raw-pointer host code that builds the CUDA tensor-map descriptors.

The current native WMMA backend remains the correctness fallback while this
SM120-specific port is developed. Its benchmark is intentionally reported as
slower than dense attention; it is not presented as the released kernel.
