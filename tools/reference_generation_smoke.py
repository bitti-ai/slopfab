"""Real-checkpoint FFmpeg-free DLL smoke run with video and PCM references.

Usage: python reference_generation_smoke.py slopfab.dll repository-root [vulkan] [--reuse]
--reuse checks identical repeated output, a new seed, and skipped VAE encodes.
No output files are written; asserts decoded frame/audio geometry and finiteness.
"""
import ctypes as C
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[2]).resolve()
dll = C.CDLL(str(pathlib.Path(sys.argv[1]).resolve()))
handle = C.c_void_p
reuse = "--reuse" in sys.argv[3:]
reference_events = []
denoise_events = []


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
    if p.stage == 1: reference_events.append(p.elapsed_seconds)
    if p.stage == 4: denoise_events.append((p.step, p.total_steps))
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
    if "vulkan" in sys.argv[3:]:
        set_backend = bind("slopfab_request_set_inference_backend", [handle, C.c_int])
        check(set_backend(request, 1))
    set_seed = bind("slopfab_request_set_seed", [handle, C.c_uint64])
    if reuse:
        check(bind("slopfab_request_set_reuse_models", [handle, C.c_int])(request, 1))
    first_output = None
    first_reference_count = None
    runs = 3 if reuse else 1
    for run in range(runs):
        check(set_seed(request, 11 if run < 2 else 12))
        reference_events.clear()
        denoise_events.clear()
        check(start(request, progress, None, C.byref(generation)))
        # On the last run all input handles are released while work is active.
        if run == runs - 1:
            destroy(request)
            request = None
        if video:
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
        assert denoise_events and denoise_events[0][0] == -1, "missing denoising entry notification"
        assert denoise_events[0][1] > 0, "denoising entry must report its step count"
        assert [step for step, _ in denoise_events[1:]] == list(range(denoise_events[0][1]))
        assert (result.frames, result.width, result.height) == (22, 32, 32)
        assert result.audio_frames > 0 and result.audio_channels == 2
        assert all(math.isfinite(result.video[i]) for i in range(result.video_float_count))
        assert all(math.isfinite(result.audio[i]) for i in range(result.audio_float_count))
        video_bytes = C.string_at(result.video, result.video_float_count * C.sizeof(C.c_float))
        audio_bytes = C.string_at(result.audio, result.audio_float_count * C.sizeof(C.c_float))
        if run == 0:
            first_output = (video_bytes, audio_bytes)
            first_reference_count = len(reference_events)
        elif run == 1:
            assert first_output == (video_bytes, audio_bytes), "cache changed same-seed output"
        else:
            assert first_output[0] != video_bytes, "new seed did not change video"
        if run:
            assert len(reference_events) < first_reference_count, "cache did not skip VAE encoding"
        print(f"PASS run {run + 1}: {result.frames} frames, {result.audio_frames} stereo audio samples, "
              f"{result.steps_computed} denoiser evaluation(s), {len(reference_events)} reference events.", flush=True)
        generation_destroy(generation)
        generation = handle()
    if reuse:
        check(bind("slopfab_reused_models_clear", [])())

finally:
    if generation:
        generation_destroy(generation)
    if video:
        video_destroy(video)
    if request:
        destroy(request)
