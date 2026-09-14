"""Real-checkpoint FFmpeg-free DLL smoke run with video and PCM references.

Usage: python reference_generation_smoke.py slopfab.dll repository-root [vulkan]
No output files are written; asserts decoded frame/audio geometry and finiteness.
"""
import ctypes as C
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[2]).resolve()
dll = C.CDLL(str(pathlib.Path(sys.argv[1]).resolve()))
handle = C.c_void_p


def bind(name, arguments, result=C.c_int):
    fn = getattr(dll, name)
    fn.argtypes, fn.restype = arguments, result
    return fn


error = bind("slopfab_last_error", [], C.c_char_p)


def check(status):
    if status:
        raise RuntimeError(f"slopfab status {status}: {error().decode()}")


create = bind("slopfab_request_create", [], handle)
destroy = bind("slopfab_request_destroy", [handle], None)
video_create = bind("slopfab_reference_video_create", [C.c_double, C.POINTER(handle)])
video_destroy = bind("slopfab_reference_video_destroy", [handle], None)
append = bind("slopfab_reference_video_append_rgb24", [handle, C.POINTER(C.c_uint8), C.c_size_t, C.c_int, C.c_int, C.c_size_t, C.c_double])
audio = bind("slopfab_reference_video_set_audio_f32", [handle, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int, C.c_double])
attach = bind("slopfab_request_add_reference_video", [handle, handle])
standalone = bind("slopfab_request_add_reference_audio_f32", [handle, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int])
set_model = bind("slopfab_request_set_model_path", [handle, C.c_int, C.c_char_p])
set_prompt = bind("slopfab_request_set_prompt", [handle, C.c_char_p])
set_resolution = bind("slopfab_request_set_resolution", [handle, C.c_int, C.c_int])
set_frames = bind("slopfab_request_set_frames", [handle, C.c_int])
set_steps = bind("slopfab_request_set_steps", [handle, C.c_int])


class Progress(C.Structure):
    _fields_ = [("stage", C.c_int32), ("step", C.c_int32), ("total_steps", C.c_int32), ("elapsed_seconds", C.c_double)]


class Output(C.Structure):
    _fields_ = [("video", C.POINTER(C.c_float)), ("video_float_count", C.c_size_t),
                ("channels", C.c_int32), ("frames", C.c_int32), ("width", C.c_int32), ("height", C.c_int32),
                ("audio", C.POINTER(C.c_float)), ("audio_float_count", C.c_size_t),
                ("audio_channels", C.c_int32), ("audio_sample_rate", C.c_int32), ("audio_frames", C.c_int64),
                ("fps", C.c_double)] + [(name, C.c_double) for name in
                ("seconds_conditioning", "seconds_denoise", "seconds_video_decode", "seconds_audio_decode", "seconds_total")] + [
                ("steps_computed", C.c_int32), ("steps_skipped", C.c_int32)]


Callback = C.CFUNCTYPE(None, C.POINTER(Progress), handle)


@Callback
def progress(value, _):
    p = value.contents
    print(f"stage={p.stage} step={p.step} elapsed={p.elapsed_seconds:.1f}s", flush=True)


start = bind("slopfab_generation_start", [handle, Callback, handle, C.POINTER(handle)])
wait = bind("slopfab_generation_wait", [handle, C.c_int])
generation_error = bind("slopfab_generation_error", [handle], C.c_char_p)
generation_destroy = bind("slopfab_generation_destroy", [handle], None)
output = bind("slopfab_generation_output", [handle, C.POINTER(Output)])
request, video, generation = create(), handle(), handle()
try:
    check(video_create(2, C.byref(video)))
    for i in range(48):
        pixels = (C.c_uint8 * 48)(*([100 + i, 90, 60] * 16))
        check(append(video, pixels, len(pixels), 4, 4, 12, i / 24))
    pcm = (C.c_float * 64000)(*(.05 * math.sin(2 * math.pi * 440 * i / 32000) for i in range(64000)))
    check(audio(video, pcm, len(pcm), 1, 32000, 0))
    check(attach(request, video))
    check(standalone(request, pcm, len(pcm), 1, 32000))
    for model, path in [(0, "transformer/minimax_h3_ref2va_pruned_fp8_scaled.safetensors"),
                        (1, "text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
                        (3, "vae/minimax_h3_video_vae_fp16.safetensors"),
                        (4, "vae/minimax_h3_audio_vae_fp32.safetensors")]:
        check(set_model(request, model, str(root / "weights" / path).encode()))
    check(set_prompt(request, b"A warm red scene with gentle motion and a soft continuous tone."))
    check(set_resolution(request, 32, 32))
    check(set_frames(request, 22))
    check(set_steps(request, 2))
    if len(sys.argv) > 3 and sys.argv[3] == "vulkan":
        set_backend = bind("slopfab_request_set_inference_backend", [handle, C.c_int])
        check(set_backend(request, 1))
    check(start(request, progress, None, C.byref(generation)))
    # Inputs are released while the worker is active, exercising ownership.
    destroy(request)
    request = None
    video_destroy(video)
    video = handle()
    while True:
        status = wait(generation, 1000)
        if status == -7:
            continue
        if status:
            raise RuntimeError(generation_error(generation).decode())
        break
    result = Output()
    check(output(generation, C.byref(result)))
    assert (result.frames, result.width, result.height) == (22, 32, 32)
    assert result.audio_frames > 0 and result.audio_channels == 2
    assert all(math.isfinite(result.video[i]) for i in range(result.video_float_count))
    assert all(math.isfinite(result.audio[i]) for i in range(result.audio_float_count))
    print(f"PASS: {result.frames} frames, {result.audio_frames} stereo audio samples, {result.steps_computed} denoiser evaluation(s).")
finally:
    if generation:
        generation_destroy(generation)
    if video:
        video_destroy(video)
    if request:
        destroy(request)
