"""Exercise Fibo through the C ABI with real local weights, pixels and cancellation."""
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
    check(bind("slopfab_request_set_model_family", [handle, C.c_char_p])(request, b"fibo2"))
    set_path = bind("slopfab_request_set_model_path", [handle, C.c_int, C.c_char_p])
    for kind, path in [(0, "transformer/fibo2_edit_turbo_int8_convrot.safetensors"),
                       (1, "text_encoder/qwen3vl_4b_int8_convrot.safetensors"),
                       (3, "vae/flux2-vae.safetensors")]:
        check(set_path(request, kind, str((args.weights/path).resolve()).encode()))
    check(bind("slopfab_request_set_model_family", [handle, C.c_char_p])(request, b""))
    check(bind("slopfab_request_set_prompt", [handle, C.c_char_p])(request, b"Make the object blue"))
    check(bind("slopfab_request_set_resolution", [handle, C.c_int, C.c_int])(request, 64, 96))
    check(bind("slopfab_request_set_steps", [handle, C.c_int])(request, 4))
    check(bind("slopfab_request_set_inference_backend", [handle, C.c_int])(request, 0))
    check(bind("slopfab_request_set_attention", [handle, C.c_char_p])(request, b"flash2"))
    source=bytes(v for y in range(96) for x in range(64) for v in (x*4,y*2,120))
    rgb=(C.c_uint8*len(source)).from_buffer_copy(source)
    set_edit=bind("slopfab_request_set_image_edit_rgb24",[handle,C.POINTER(C.c_uint8),C.c_size_t,C.c_int,C.c_int,C.c_size_t,C.c_int,C.c_int,C.c_int,C.c_int,C.c_float,C.c_int])
    check(set_edit(request,rgb,len(source),64,96,64*3,0,0,64,96,1.,0))
    coverage=bytes(255 if (x-32)**2+(y-48)**2<20**2 else 0 for y in range(96) for x in range(64))
    mask=(C.c_uint8*len(coverage)).from_buffer_copy(coverage)
    check(bind("slopfab_request_set_image_edit_mask_u8",[handle,C.POINTER(C.c_uint8),C.c_size_t,C.c_int,C.c_int,C.c_size_t])(request,mask,len(coverage),64,96,64))
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
    for i,alpha in enumerate(coverage):
        if alpha==0:
            for c in range(3):assert abs(pixels[c*64*96+i]-source[i*3+c]/255)<1e-7
    assert output.steps_computed == 4
    destroy(generation)
    generation = handle()
    check(start(request, None, None, C.byref(generation)))
    bind("slopfab_generation_cancel", [handle], None)(generation)
    assert wait(generation, -1) == -8, "expected cancellation"
    print("Fibo C API generation, pixel ownership and cancellation passed")
finally:
    if generation:
        destroy(generation)
    bind("slopfab_request_destroy", [handle], None)(request)
