"""Real-weight DLL Sol-H3-VAE selection, still/video, progress and cancellation.

python tools/sol_h3_vae_smoke.py build/Release/slopfab.dll /path/sol_h3_vae.safetensors
"""
import argparse
import ctypes as C
import math
from pathlib import Path
import threading
from continuation_smoke import Output

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("library", type=Path)
p.add_argument("weights", type=Path)
p.add_argument("--root", type=Path, default=Path("."))
a = p.parse_args()
dll = C.CDLL(str(a.library.resolve()))
handle = C.c_void_p


def bind(name, args, result=C.c_int):
    fn = getattr(dll, "slopfab_" + name)
    fn.argtypes, fn.restype = args, result
    return fn


error = bind("last_error", [], C.c_char_p)


def check(status):
    if status:
        raise RuntimeError(f"{status}: {error().decode()}")


class Progress(C.Structure):
    _fields_ = [("stage", C.c_int), ("step", C.c_int), ("total", C.c_int), ("elapsed", C.c_double)]


Callback = C.CFUNCTYPE(None, C.POINTER(Progress), handle)
create = bind("request_create", [], handle)
free = bind("request_destroy", [handle], None)
model = bind("request_set_model_path", [handle, C.c_int, C.c_char_p])
start = bind("generation_start", [handle, handle, handle, C.POINTER(handle)])
wait = bind("generation_wait", [handle, C.c_int])
gen_error = bind("generation_error", [handle], C.c_char_p)
destroy = bind("generation_destroy", [handle], None)
cancel = bind("generation_cancel", [handle], None)
output = bind("generation_output", [handle, C.POINTER(Output)])

for still, stop in [(False, False), (False, True), (True, False)]:
    request = create()
    generation = handle()
    ready = threading.Event()
    events = []

    @Callback
    def progress(value, _):
        event = value.contents
        events.append((event.stage, event.step, event.total))
        if stop and event.stage == 5 and event.step >= 2:
            ready.wait()
            cancel(generation)

    try:
        check(bind("request_set_synthetic_latents", [handle, C.c_int])(request, 1))
        check(bind("request_set_resolution", [handle, C.c_int, C.c_int])(request, 64, 32))
        check(bind("request_set_frames", [handle, C.c_int])(request, 22))
        check(bind("request_set_still_image", [handle, C.c_int])(request, still))
        check(model(request, 3, str(a.root.resolve() / "weights/vae/minimax_h3_video_vae_fp16.safetensors").encode()))
        check(model(request, 5, str(a.weights.resolve()).encode()))
        check(model(request, 4, str(a.root.resolve() / "weights/vae/minimax_h3_audio_vae_fp32.safetensors").encode()))
        check(start(request, C.cast(progress, handle), None, C.byref(generation)))
        ready.set()
        free(request)
        request = None
        status = wait(generation, -1)
        if stop:
            assert status == -8, (status, gen_error(generation))
        else:
            if status:
                raise RuntimeError(gen_error(generation).decode())
            samples = Output()
            check(output(generation, C.byref(samples)))
            assert (samples.width, samples.height, samples.frames) == (64, 32, 1 if still else 22)
            assert all(math.isfinite(samples.video[i]) for i in range(samples.video_float_count))
            phase = [(done, total) for stage, done, total in events if stage == 5 and total == 24]
            assert phase == [(i, 24) for i in range(25)], phase
        print(f"PASS still={still}, cancelled={stop}", flush=True)
    finally:
        ready.set()
        if generation:
            destroy(generation)
        if request:
            free(request)
