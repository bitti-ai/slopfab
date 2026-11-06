"""Exercise real SeedVR2 checkpoints through the Vulkan streaming C ABI."""
import argparse
import ctypes as C
import math
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--library", required=True, type=Path)
p.add_argument("--dit", required=True, type=Path)
p.add_argument("--vae", required=True, type=Path)
p.add_argument("--compare-cuda", action="store_true")
p.add_argument("--width", type=int, default=32)
p.add_argument("--height", type=int, default=48)
p.add_argument("--frames", type=int, default=5)
a = p.parse_args()
lib = C.CDLL(str(a.library.resolve()))
lib.slopfab_last_error.restype = C.c_char_p


class Options(C.Structure):
    _fields_ = [("size", C.c_uint32), ("dit", C.c_char_p), ("vae", C.c_char_p),
                ("width", C.c_int32), ("height", C.c_int32), ("frames", C.c_int32),
                ("tile", C.c_int32), ("device", C.c_int32), ("color", C.c_int32),
                ("seed", C.c_uint64)]


Read = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(C.c_float), C.c_uint64)
Write = Read
Cancel = C.CFUNCTYPE(C.c_int, C.c_void_p)
Progress = C.CFUNCTYPE(None, C.c_void_p, C.c_char_p)
run = lib.slopfab_seedvr2_upscale_backend
run.argtypes = [C.POINTER(Options), C.c_int, Read, Write, Cancel, Progress,
                C.c_void_p, C.POINTER(C.c_uint64)]
o = Options(C.sizeof(Options), str(a.dit.resolve()).encode(), str(a.vae.resolve()).encode(),
            a.width, a.height, a.frames, 0, 0, 0, 666)
size = a.width * a.height * 3


def restore(backend, cancel=False):
    frame = 0
    result = []

    @Read
    def read(_, out, count):
        nonlocal frame
        if frame == a.frames:
            return 0
        assert count == size
        for i in range(size):
            out[i] = ((i * 7 + frame * 13) % 251) / 250
        frame += 1
        return 1

    @Write
    def write(_, out, count):
        assert count == size
        result.extend(out[:count])
        return 0

    @Progress
    def progress(_, message):
        print(("CUDA" if backend == 0 else "Vulkan"), message.decode(), flush=True)

    stop = Cancel(lambda _: int(cancel))
    written = C.c_uint64(999)
    code = run(C.byref(o), backend, read, write, stop, progress, None, C.byref(written))
    if cancel:
        assert code == -8 and written.value == 0
        return []
    if code:
        raise RuntimeError(lib.slopfab_last_error().decode())
    assert written.value == a.frames and len(result) == a.frames * size
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in result)
    assert max(result) - min(result) > .01
    return result


restore(1, cancel=True)
actual = restore(1)
if a.compare_cuda:
    reference = restore(0)
    error = sum((x-y)**2 for x, y in zip(actual, reference))
    relative = math.sqrt(error / sum(x*x for x in reference))
    print(f"Vulkan/CUDA pixel relative L2: {relative:.6f}", flush=True)
    assert relative < .1, "Vulkan/CUDA restoration differs by more than 10% relative L2"
print("SeedVR2 Vulkan streaming, temporal frames, pixels and cancellation passed")
