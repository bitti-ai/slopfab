"""Run real outpainting and verify exact preservation of the source rectangle.

Requires Pillow and numpy. Supply a source image, output folder and model paths;
this writes a prepared canvas and generated PNG for visual seam inspection.
"""
import argparse
import ctypes as C
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
for name in ("library", "source", "output", "transformer", "text-encoder", "vae"):
    parser.add_argument("--" + name, required=True, type=Path)
parser.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--lora", type=Path)
parser.add_argument("--schedule", choices=("default", "dmad-4step"), default="default")
parser.add_argument("--strength", type=float, default=1)
args = parser.parse_args()
assert args.size >= 96 and args.size % 32 == 0
args.output.mkdir(parents=True, exist_ok=True)
source = Image.open(args.source).convert("RGB")
source.thumbnail((args.size * 2 // 3, args.size * 2 // 3), Image.Resampling.LANCZOS)
left, top = (args.size - source.width) // 2, (args.size - source.height) // 2
canvas = Image.new("RGB", (args.size, args.size), (127, 127, 127))
canvas.paste(source, (left, top))
canvas.save(args.output / "canvas.png")
pixels = np.array(canvas)
dll = C.CDLL(str(args.library.resolve()))
handle = C.c_void_p


def bind(name, arguments, result=C.c_int):
    fn = getattr(dll, name)
    fn.argtypes, fn.restype = arguments, result
    return fn


error = bind("slopfab_last_error", [], C.c_char_p)


def check(status):
    if status:
        raise RuntimeError(f"SlopFab {status}: {error().decode()}")


class Progress(C.Structure):
    _fields_ = [("stage", C.c_int32), ("step", C.c_int32), ("total_steps", C.c_int32),
                ("elapsed_seconds", C.c_double)]


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
    print(f"stage={p.stage} step={p.step}/{p.total_steps} elapsed={p.elapsed_seconds:.1f}s", flush=True)


request = bind("slopfab_request_create", [], handle)()
generation = handle()
try:
    check(bind("slopfab_request_set_verbose", [handle, C.c_int])(request, 1))
    set_model = bind("slopfab_request_set_model_path", [handle, C.c_int, C.c_char_p])
    for model, path in [(0, args.transformer), (1, args.text_encoder), (3, args.vae)]:
        check(set_model(request, model, str(path.resolve()).encode()))
    prompt = ("integrated_multimodal_description: [Shot 1] A wider view of the source scene. "
              "Continue the same surroundings beyond the visible edges, matching perspective, scale, lighting and textures. "
              "One continuous scene, with the original content in its existing position."
              "\n\noverall_soundscape: N/A\n\nnon_diegetic_music: N/A")
    check(bind("slopfab_request_set_prompt", [handle, C.c_char_p])(request, prompt.encode()))
    check(bind("slopfab_request_set_steps", [handle, C.c_int])(request, args.steps))
    if args.lora:
        check(bind("slopfab_request_add_lora", [handle, C.c_char_p, C.c_float])(
            request, str(args.lora.resolve()).encode(), 1))
    check(bind("slopfab_request_set_schedule", [handle, C.c_int])(
        request, 2 if args.schedule == "dmad-4step" else 0))
    check(bind("slopfab_request_set_seed", [handle, C.c_uint64])(request, 17))
    check(bind("slopfab_request_set_inference_backend", [handle, C.c_int])(request, int(args.backend == "vulkan")))
    check(bind("slopfab_request_set_attention", [handle, C.c_char_p])(request, b"sage2"))
    check(bind("slopfab_request_set_image_edit_path", [handle, C.c_char_p, C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.c_int])(
        request, str((args.output / "canvas.png").resolve()).encode(), left, top, source.width, source.height, args.strength, 0))
    check(bind("slopfab_request_set_image_edit_invert_mask", [handle, C.c_int])(request, 1))
    check(bind("slopfab_request_set_outpaint_blend_overlap", [handle, C.c_int])(request, 1))
    check(bind("slopfab_generation_start", [handle, Callback, handle, C.POINTER(handle)])(request, progress, None, C.byref(generation)))
    wait = bind("slopfab_generation_wait", [handle, C.c_int])
    generation_error = bind("slopfab_generation_error", [handle], C.c_char_p)
    while True:
        status = wait(generation, 1000)
        if status == -7:
            continue
        if status:
            raise RuntimeError(generation_error(generation).decode())
        break
    result = Output()
    check(bind("slopfab_generation_output", [handle, C.POINTER(Output)])(generation, C.byref(result)))
    assert (result.frames, result.width, result.height, result.channels) == (1, args.size, args.size, 3)
    values = np.ctypeslib.as_array(result.video, shape=(result.video_float_count,)).reshape(3, args.size, args.size).transpose(1, 2, 0)
    assert np.isfinite(values).all()
    expected_source = np.array(source).astype(np.float32) / np.float32(255)
    assert np.array_equal(values[top:top + source.height, left:left + source.width], expected_source)
    rgb = np.round(values.clip(0, 1) * 255).astype(np.uint8)
    assert np.array_equal(rgb[top:top + source.height, left:left + source.width], np.array(source))
    outside = np.ones((args.size, args.size), dtype=bool)
    outside[top:top + source.height, left:left + source.width] = False
    assert np.abs(rgb[outside].astype(float) - pixels[outside]).mean() > 1
    Image.fromarray(rgb).save(args.output / "result.png")
    print(f"PASS: source preserved, surround generated in {result.steps_computed} evaluations. Inspect {args.output / 'result.png'}", flush=True)
finally:
    if generation:
        bind("slopfab_generation_destroy", [handle], None)(generation)
    bind("slopfab_request_destroy", [handle], None)(request)
