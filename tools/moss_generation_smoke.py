"""Exercise MOSS through the C ABI with real local weights, pixels and cancellation."""
import argparse
import ctypes as C
from pathlib import Path
import math

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--library", required=True, type=Path)
parser.add_argument("--weights", default="weights", type=Path)
args = parser.parse_args()
dll = C.CDLL(str(args.library.resolve()))
handle = C.c_void_p


def bind(name, arguments, result=C.c_int):
    function = getattr(dll, name)
    function.argtypes, function.restype = arguments, result
    return function


error = bind("slopfab_last_error", [], C.c_char_p)


def check(code):
    if code:
        raise RuntimeError(f"{code}: {error().decode()}")


class Output(C.Structure):
    _fields_ = [("video", C.POINTER(C.c_float)), ("video_float_count", C.c_size_t),
                ("channels", C.c_int32), ("frames", C.c_int32), ("width", C.c_int32), ("height", C.c_int32),
                ("audio", C.POINTER(C.c_float)), ("audio_float_count", C.c_size_t),
                ("audio_channels", C.c_int32), ("audio_sample_rate", C.c_int32), ("audio_frames", C.c_int64),
                ("fps", C.c_double)] + [(name, C.c_double) for name in
                ("seconds_conditioning", "seconds_denoise", "seconds_video_decode", "seconds_audio_decode", "seconds_total")] + [
                ("steps_computed", C.c_int32), ("steps_skipped", C.c_int32)]


request = bind("slopfab_request_create", [], handle)()
generation = handle()
destroy = bind("slopfab_generation_destroy", [handle], None)
start = bind("slopfab_generation_start", [handle, handle, handle, C.POINTER(handle)])
wait = bind("slopfab_generation_wait", [handle, C.c_int])
try:
    check(bind("slopfab_request_set_model_family", [handle, C.c_char_p])(request, b"moss"))
    set_path = bind("slopfab_request_set_model_path", [handle, C.c_int, C.c_char_p])
    for kind, path in [(0, "transformer/moss_image_sft_int8_convrot.safetensors"),
                       (1, "text_encoder/qwen3.5_9b_int8_convrot.safetensors"),
                       (2, "text_encoder/qwen3.5_tokenizer.json"),
                       (3, "vae/flux2-vae_small.safetensors")]:
        check(set_path(request, kind, str((args.weights/path).resolve()).encode()))
    check(bind("slopfab_request_set_prompt", [handle, C.c_char_p])(request, b"A red apple"))
    check(bind("slopfab_request_set_resolution", [handle, C.c_int, C.c_int])(request, 64, 96))
    check(bind("slopfab_request_set_steps", [handle, C.c_int])(request, 1))
    check(bind("slopfab_request_set_inference_backend", [handle, C.c_int])(request, 0))
    check(bind("slopfab_request_set_attention", [handle, C.c_char_p])(request, b"flash2"))
    check(start(request, None, None, C.byref(generation)))
    status = wait(generation, -1)
    if status:
        raise RuntimeError(bind("slopfab_generation_error", [handle], C.c_char_p)(generation).decode())
    output = Output()
    check(bind("slopfab_generation_output", [handle, C.POINTER(Output)])(generation, C.byref(output)))
    assert (output.width, output.height, output.frames, output.channels) == (64, 96, 1, 3)
    assert output.audio_float_count == 0 and output.video_float_count == 64*96*3
    pixels = output.video[:output.video_float_count]
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in pixels)
    assert max(pixels)-min(pixels) > .01
    assert output.steps_computed == 1
    destroy(generation)
    generation = handle()
    check(start(request, None, None, C.byref(generation)))
    bind("slopfab_generation_cancel", [handle], None)(generation)
    assert wait(generation, -1) == -8, "expected cancellation"
    print("MOSS C API generation, pixel ownership and cancellation passed")
finally:
    if generation:
        destroy(generation)
    bind("slopfab_request_destroy", [handle], None)(request)
