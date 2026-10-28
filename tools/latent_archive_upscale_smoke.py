"""Upscale an existing H3 archive through the DLL and CLI with a real checkpoint.

python tools/latent_archive_upscale_smoke.py build/Release/slopfab.dll . cuda
Checks metadata, audio preservation, retained-source lifetime, continuation
planning, progress/cancellation, and decoding the enlarged archive without DiT.
"""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import threading

from continuation_smoke import read_archive

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("library", type=Path)
p.add_argument("root", type=Path)
p.add_argument("backend", choices=("cuda", "vulkan"))
p.add_argument("--frames", type=int, default=22)
a = p.parse_args()
if a.frames < 22 or a.frames % 17 != 5:
    p.error("frames must be 17*k+5 and at least 22")
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
arguments = [C.c_char_p, C.c_float, C.c_int, C.c_int, handle, handle, C.POINTER(handle)]
start_file = bind("latent_upscale_file_start", [C.c_char_p] + arguments)
start_gen = bind("latent_upscale_generation_start", [handle] + arguments)
wait = bind("generation_wait", [handle, C.c_int])
gen_error = bind("generation_error", [handle], C.c_char_p)
destroy = bind("generation_destroy", [handle], None)
cancel = bind("generation_cancel", [handle], None)
save = bind("generation_save_latents", [handle, C.c_char_p])
model = str(a.root.resolve() / "weights/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors").encode()
backend = int(a.backend == "vulkan")
jobs = []


def finish(job):
    status = wait(job, -1)
    if status:
        raise RuntimeError(gen_error(job).decode())


try:
    with tempfile.TemporaryDirectory(prefix="slopfab-archive-upscale-") as directory:
        work = Path(directory)
        source = work / "source.safetensors"
        video_count = ((a.frames - 5) // 17 * 5 + 2) * 192
        audio_count = ((a.frames * 5 + 1) // 3) * 64
        video = struct.pack(f"<{video_count}f", *(math.sin(i) * .1 for i in range(video_count)))
        audio = struct.pack(f"<{audio_count}f", *(math.cos(i) * .1 for i in range(audio_count)))
        header = {"__metadata__": {"slopfab_latents": "h3-av-v1", "width": "64", "height": "32",
                                   "frames": str(a.frames), "fps": "24", "sampled": "1"},
                  "video_rows": {"dtype": "F32", "shape": [video_count // 96, 96], "data_offsets": [0, len(video)]},
                  "audio_rows": {"dtype": "F32", "shape": [audio_count // 32, 32], "data_offsets": [len(video), len(video) + len(audio)]}}
        raw = json.dumps(header).encode()
        raw += b" " * (-len(raw) % 8)
        original = struct.pack("<Q", len(raw)) + raw + video + audio
        source.write_bytes(original)
        events = []

        @Callback
        def progress(value, _):
            v = value.contents
            events.append((v.stage, v.step, v.total))

        job = handle()
        check(start_file(str(source).encode(), model, 2, 1, backend, C.cast(progress, handle), None, C.byref(job)))
        jobs.append(job)
        finish(job)
        result = work / "enlarged.safetensors"
        check(save(job, str(result).encode()))
        meta, rows = read_archive(result)
        assert (meta["__metadata__"]["width"], meta["__metadata__"]["height"], meta["__metadata__"]["frames"]) == ("128", "64", str(a.frames))
        assert meta["video_rows"]["shape"] == [video_count // 96 * 4, 96]
        assert rows["audio_rows"] == audio
        assert all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", rows["video_rows"]))
        assert source.read_bytes() == original
        phase = [(step, total) for stage, step, total in events if stage == 9]
        assert phase and phase[-1][0] == phase[-1][1] > 0
        assert all(x[0] <= y[0] for x, y in zip(phase, phase[1:]))
        assert events[0][0] == 0 and events[-1][0] == 8
        assert not any(stage in (2, 3, 4, 5, 6) for stage, _, _ in events)
        request = bind("request_create", [], handle)()
        try:
            check(bind("request_set_continuation_generation", [handle, handle, C.c_int])(request, job, 22))
            description = handle()
            check(bind("describe_plan", [handle, C.POINTER(handle)])(request, C.byref(description)))
            try:
                assert "64 x 128" in C.string_at(description).decode()
            finally:
                bind("free_string", [handle], None)(description)
        finally:
            bind("request_destroy", [handle], None)(request)
        copy = handle()
        check(start_gen(job, model, 1, 1, backend, None, None, C.byref(copy)))
        jobs.append(copy)
        destroy(job)
        jobs.remove(job)
        source.unlink()
        finish(copy)
        check(save(copy, str(work / "copy.safetensors").encode()))
        assert read_archive(work / "copy.safetensors")[1] == rows
        source.write_bytes(original)
        # A cancellation inside the actual upscale, then a successful recovery.
        ready = threading.Event()
        cancelled = handle()

        @Callback
        def stop(value, _):
            if value.contents.stage == 9 and value.contents.step > 0:
                ready.wait()
                cancel(cancelled)

        check(start_file(str(source).encode(), model, 2, 1, backend, C.cast(stop, handle), None, C.byref(cancelled)))
        jobs.append(cancelled)
        ready.set()
        assert wait(cancelled, -1) == -8
        recovery = handle()
        check(start_file(str(source).encode(), model, 1, 1, backend, None, None, C.byref(recovery)))
        jobs.append(recovery)
        finish(recovery)
        exe = a.library.resolve().with_name("slopfab.exe" if a.library.suffix == ".dll" else "slopfab")
        cli = work / "cli.safetensors"
        subprocess.run([str(exe), "upscale-latents", "--input", str(source), "--out", str(cli),
                        "--model", model.decode(), "--scale", "2", "--inference-backend", a.backend], check=True)
        assert read_archive(cli)[1] == rows
        output = work / "decoded.y4m"
        subprocess.run([str(exe), "decode", "--latent", str(cli), "--vae",
                        str(a.root.resolve() / "weights/vae/minimax_h3_video_vae_fp16.safetensors"),
                        "--inference-backend", a.backend, "--out", str(output)], check=True)
        assert b"W128 H64" in output.read_bytes().split(b"\n", 1)[0]
        print(f"PASS {a.backend}: archive geometry/audio, retained lifetime, continuation, progress, cancellation, CLI and decode", flush=True)
finally:
    for job in jobs:
        destroy(job)
