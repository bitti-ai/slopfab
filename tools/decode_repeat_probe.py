"""Repeat real VAE decoding through the DLL, without conditioning/transformer.

python tools/decode_repeat_probe.py build/Release/slopfab.dll --runs 5
Use --lifetime retain to demonstrate output ownership, or release to keep only
latents (C API 1.25+). Synthetic input is deterministic noise, not a quality test.
Memory figures are process-wide; GPU memory and file cache are not included.
"""
import argparse
import ctypes as C
import hashlib
import json
import pathlib
import tempfile
import time

from continuation_smoke import Output


def memory():
    if hasattr(C, "WinDLL"):
        class Counters(C.Structure):
            _fields_ = [("cb", C.c_ulong), ("faults", C.c_ulong)] + [
                (name, C.c_size_t) for name in
                ("peak_working", "working", "peak_paged", "paged", "peak_nonpaged",
                 "nonpaged", "commit", "peak_commit", "private")]
        counters = Counters()
        counters.cb = C.sizeof(counters)
        query = C.WinDLL("psapi").GetProcessMemoryInfo
        query.argtypes = [C.c_void_p, C.POINTER(Counters), C.c_ulong]
        if not query(C.c_void_p(-1), C.byref(counters), counters.cb):
            raise C.WinError()
        return {name + "_mib": round(getattr(counters, name) / 1048576, 2)
                for name in ("working", "private", "peak_commit")}
    return {}


def digest(pointer, count):
    # A buffer view avoids copying a potentially multi-gigabyte output.
    if not count:
        return None
    view = (C.c_ubyte * (count * 4)).from_address(C.addressof(pointer.contents))
    return hashlib.sha256(view).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=pathlib.Path)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
    parser.add_argument("--lifetime", choices=("destroy", "retain", "release"), default="destroy")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    dll = C.CDLL(str(args.dll.resolve()))
    handle = C.c_void_p

    def bind(name, arguments, result=C.c_int):
        fn = getattr(dll, "slopfab_" + name)
        fn.argtypes, fn.restype = arguments, result
        return fn

    error = bind("last_error", [], C.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(f"{status}: {error().decode()}")

    create = bind("request_create", [], handle)
    request_destroy = bind("request_destroy", [handle], None)
    destroy = bind("generation_destroy", [handle], None)
    model = bind("request_set_model_path", [handle, C.c_int, C.c_char_p])
    resolution = bind("request_set_resolution", [handle, C.c_int, C.c_int])
    frames = bind("request_set_frames", [handle, C.c_int])
    backend = bind("request_set_inference_backend", [handle, C.c_int])
    synthetic = bind("request_set_synthetic_latents", [handle, C.c_int])
    verbose = bind("request_set_verbose", [handle, C.c_int])
    retain = bind("request_set_retain_latents", [handle, C.c_int])
    start = bind("generation_start", [handle, handle, handle, C.POINTER(handle)])
    wait = bind("generation_wait", [handle, C.c_int])
    status = bind("generation_status", [handle])
    gen_error = bind("generation_error", [handle], C.c_char_p)
    output = bind("generation_output", [handle, C.POINTER(Output)])
    release = bind("generation_release_samples", [handle]) if args.lifetime == "release" else None
    save = bind("generation_save_latents", [handle, C.c_char_p])
    rgba = bind("generation_frame_rgba8", [handle, C.c_int, C.c_void_p, C.c_size_t])
    handles = []
    expected = None
    print(json.dumps({"before": memory(), "arguments": vars(args)}, default=str), flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="slopfab-decode-") as tmp:
            for iteration in range(args.runs):
                request = create()
                if not request:
                    raise RuntimeError(error().decode())
                generation = handle()
                begin = time.perf_counter()
                try:
                    for kind, filename in ((3, "minimax_h3_video_vae_fp16.safetensors"),
                                           (4, "minimax_h3_audio_vae_fp32.safetensors")):
                        check(model(request, kind, str(args.root / "weights" / "vae" / filename).encode()))
                    check(resolution(request, args.width, args.height))
                    check(frames(request, args.frames))
                    check(backend(request, args.backend == "vulkan"))
                    check(synthetic(request, 1))
                    check(verbose(request, 0))
                    check(retain(request, args.lifetime == "release"))
                    check(start(request, None, None, C.byref(generation)))
                    handles.append(generation)
                finally:
                    request_destroy(request)
                while True:
                    code = wait(generation, 1000)
                    if code != -7:
                        break
                if code:
                    raise RuntimeError(gen_error(generation).decode())
                elapsed = time.perf_counter() - begin
                result = Output()
                check(output(generation, C.byref(result)))
                hashes = [digest(result.video, result.video_float_count),
                          digest(result.audio, result.audio_float_count)]
                if expected is None:
                    expected = hashes
                assert hashes == expected, "decoded samples changed across identical runs"
                record = {"run": iteration + 1, "wall_seconds": round(elapsed, 3),
                          "video_seconds": result.seconds_video_decode,
                          "audio_seconds": result.seconds_audio_decode,
                          "samples_mib": (result.video_float_count + result.audio_float_count) * 4 / 1048576,
                          "hashes": hashes, "before_cleanup": memory()}
                if release:
                    before, after = (pathlib.Path(tmp) / name for name in ("before.safetensors", "after.safetensors"))
                    check(save(generation, str(before).encode()))
                    check(release(generation))
                    check(release(generation))
                    check(status(generation))
                    assert output(generation, C.byref(Output())) == -2
                    pixel = (C.c_ubyte * 4)()
                    assert rgba(generation, 0, pixel, 4) == -2
                    check(save(generation, str(after).encode()))
                    assert before.read_bytes() == after.read_bytes(), "release changed retained latents"
                elif args.lifetime == "destroy":
                    destroy(handles.pop())
                record["after_cleanup"] = memory()
                print(json.dumps(record), flush=True)
    finally:
        for generation in handles:
            destroy(generation)
    print(json.dumps({"after_all_destroyed": memory()}), flush=True)


if __name__ == "__main__":
    main()
