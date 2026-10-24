"""Exercise pixel-mask editing through the DLL with real H3 checkpoints.

Uses a captured prompt embedding to avoid loading a text encoder. Checks exact
preservation, finite output, source/mask snapshot ownership and request lifetime.
Run once per backend; no third-party Python packages are needed.
"""
import argparse
import ctypes as C
import json
from pathlib import Path


class Output(C.Structure):
    _fields_ = [
        ("video", C.POINTER(C.c_float)), ("video_float_count", C.c_size_t),
        ("channels", C.c_int32), ("frames", C.c_int32),
        ("width", C.c_int32), ("height", C.c_int32),
        ("audio", C.POINTER(C.c_float)), ("audio_float_count", C.c_size_t),
        ("audio_channels", C.c_int32), ("audio_sample_rate", C.c_int32),
        ("audio_frames", C.c_int64), ("fps", C.c_double),
    ] + [(name, C.c_double) for name in (
        "seconds_conditioning", "seconds_denoise", "seconds_video_decode",
        "seconds_audio_decode", "seconds_total",
    )] + [("steps_computed", C.c_int32), ("steps_skipped", C.c_int32)]


def main():
    import math

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "transformer", "vae", "prompt-embedding"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--sampler", choices=("euler", "renoise"), default="euler")
    args = parser.parse_args()
    dll = C.CDLL(str(args.library.resolve()))
    handle = C.c_void_p

    def bind(name, arguments, result=C.c_int):
        fn = getattr(dll, "slopfab_" + name)
        fn.argtypes, fn.restype = arguments, result
        return fn

    error = bind("last_error", [], C.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(f"SlopFab {status}: {error().decode()}")

    w, h = 67, 65
    rgb_stride, mask_stride = w * 3 + 7, w + 5
    rgb = (C.c_uint8 * ((h - 1) * rgb_stride + w * 3))()
    mask = (C.c_uint8 * ((h - 1) * mask_stride + w))()
    source, coverage = [], []
    for y in range(h):
        for x in range(w):
            color = ((x * 7 + y * 3) % 256, (x * 3 + y * 5) % 256, (x + y * 7) % 256)
            source.append(color)
            for c, value in enumerate(color):
                rgb[y * rgb_stride + x * 3 + c] = value
            # A ring with a fully preserved central latent cell, soft edges,
            # and an isolated faint pixel on the unaligned source boundary.
            selected = 3 <= x < 61 and 3 <= y < 61 and not (16 <= x < 48 and 16 <= y < 48)
            value = (128 if x < 6 or y < 6 or x >= 58 or y >= 58 else 255) if selected else 0
            if x == w - 1 and y == h - 1:
                value = 1
            mask[y * mask_stride + x] = value
            coverage.append(value)

    request = bind("request_create", [], handle)()
    generation = handle()
    destroy_request = bind("request_destroy", [handle], None)
    try:
        set_model = bind("request_set_model_path", [handle, C.c_int, C.c_char_p])
        for kind, path in ((0, args.transformer), (3, args.vae)):
            check(set_model(request, kind, str(path.resolve()).encode()))
        check(bind("request_set_prompt_embedding_path", [handle, C.c_char_p])(
            request, str(args.prompt_embedding.resolve()).encode()))
        check(bind("request_set_verbose", [handle, C.c_int])(request, 1))
        check(bind("request_set_inference_backend", [handle, C.c_int])(
            request, int(args.backend == "vulkan")))
        check(bind("request_set_attention", [handle, C.c_char_p])(request, b"sage2"))
        check(bind("request_set_steps", [handle, C.c_int])(request, 3))
        check(bind("request_set_seed", [handle, C.c_uint64])(request, 17))
        check(bind("request_set_sampling_settings", [handle, C.c_char_p])(
            request, json.dumps({"version": 1, "sampler": args.sampler,
                                 "base_sigmas": [1, .5, 0]}).encode()))
        check(bind("request_set_image_edit_rgb24", [
            handle, C.POINTER(C.c_uint8), C.c_size_t, C.c_int, C.c_int, C.c_size_t,
            C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.c_int,
        ])(request, rgb, len(rgb), w, h, rgb_stride, 0, 0, w, h, .5, 0))
        check(bind("request_set_image_edit_mask_u8", [
            handle, C.POINTER(C.c_uint8), C.c_size_t, C.c_int, C.c_int, C.c_size_t,
        ])(request, mask, len(mask), w, h, mask_stride))
        C.memset(rgb, 0, len(rgb))
        C.memset(mask, 0, len(mask))
        check(bind("generation_start", [handle, handle, handle, C.POINTER(handle)])(
            request, None, None, C.byref(generation)))
        check(bind("request_clear_image_edit", [handle])(request))
        destroy_request(request)
        request = None
        wait = bind("generation_wait", [handle, C.c_int])
        while True:
            status = wait(generation, 1000)
            if status == -7:
                continue
            if status:
                detail = bind("generation_error", [handle], C.c_char_p)(generation)
                raise RuntimeError(detail.decode())
            break
        output = Output()
        check(bind("generation_output", [handle, C.POINTER(Output)])(generation, C.byref(output)))
        assert (output.frames, output.channels, output.width, output.height) == (1, 3, w, h)
        assert output.video_float_count == 3 * w * h
        assert output.steps_computed == 1
        changed = preserved = 0
        for p, alpha in enumerate(coverage):
            for c in range(3):
                value = output.video[c * w * h + p]
                assert math.isfinite(value), (p, c, value)
                original = C.c_float(source[p][c] / 255).value
                if alpha == 0:
                    assert value == original, (p, c, value, original)
                    preserved += 1
                else:
                    changed += abs(value - original) > 1e-5
        assert changed > 0
        print(f"PASS {args.backend}/{args.sampler}: {preserved} channels exactly preserved, "
              f"{changed} selected channels changed; caller buffers and request released.", flush=True)
    finally:
        if generation:
            bind("generation_destroy", [handle], None)(generation)
        if request:
            destroy_request(request)


if __name__ == "__main__":
    main()
