"""Opt-in real H3 DLL smoke test; uses only Python's standard library.

Exercises native calibration, request snapshots, worker callback order, cancellation,
error propagation and edited predictions through the exported C ABI. A synthetic
prompt embedding avoids loading a text encoder. This is not a visual-quality test.
"""
import argparse
import ctypes as C
import json
import math
from pathlib import Path
import struct
import tempfile


class Step(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("modality", C.c_int32),
        ("step", C.c_int32), ("total_steps", C.c_int32), ("evaluation", C.c_int32),
        ("sigma", C.c_float), ("sigma_next", C.c_float),
        ("frames", C.c_int32), ("height", C.c_int32), ("width", C.c_int32),
        ("row_width", C.c_int32), ("count", C.c_uint64),
        ("latent", C.POINTER(C.c_float)), ("denoised", C.POINTER(C.c_float)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
    args = parser.parse_args()
    lib = C.CDLL(str(Path(args.dll).resolve()))
    pointer, integer, string = C.c_void_p, C.c_int32, C.c_char_p

    def bind(name, restype, *argtypes):
        fn = getattr(lib, "slopfab_" + name)
        fn.restype, fn.argtypes = restype, argtypes
        return fn

    error = bind("last_error", string)
    create = bind("request_create", pointer)
    destroy = bind("request_destroy", None, pointer)
    set_path = bind("request_set_model_path", integer, pointer, integer, string)
    set_lcs = bind("request_set_lcs", integer, pointer, string, string)
    calibrate = bind("lcs_calibrate", integer, pointer, string, integer, integer, pointer, pointer)
    callback_type = C.CFUNCTYPE(C.c_int, C.POINTER(Step), pointer)
    set_callback = bind("request_set_sampling_callback", integer, pointer, callback_type, pointer)
    start = bind("generation_start", integer, pointer, pointer, pointer, C.POINTER(pointer))
    wait = bind("generation_wait", integer, pointer, integer)
    gen_error = bind("generation_error", string, pointer)
    gen_destroy = bind("generation_destroy", None, pointer)

    def check(status):
        if status != 0:
            raise RuntimeError(f"status {status}: {error().decode()}")

    with tempfile.TemporaryDirectory(prefix="slopfab-lcs-dll-") as directory:
        root = Path(directory)
        calibration, embedding = root / "lcs.safetensors", root / "prompt.safetensors"
        header = json.dumps({"prompt_embedding": {
            "dtype": "F32", "shape": [1, 5120], "data_offsets": [0, 5120 * 4]
        }}).encode()
        header += b" " * (-len(header) % 8)
        embedding.write_bytes(struct.pack("<Q", len(header)) + header + bytes(5120 * 4))
        request = create()
        assert request
        try:
            check(set_path(request, 3, str(Path(args.vae).resolve()).encode()))
            check(set_path(request, 0, str(Path(args.transformer).resolve()).encode()))
            check(bind("request_set_inference_backend", integer, pointer, integer)(request, args.backend == "vulkan"))
            check(bind("request_set_vulkan_arithmetic", integer, pointer, integer)(request, 1))
            check(bind("request_set_attention", integer, pointer, string)(request, b"flash2"))
            check(bind("request_set_prompt_embedding_path", integer, pointer, string)(request, str(embedding).encode()))
            check(bind("request_set_still_image", integer, pointer, integer)(request, 1))
            check(bind("request_set_resolution", integer, pointer, integer, integer)(request, 64, 64))
            check(bind("request_set_steps", integer, pointer, integer)(request, 2))
            check(bind("request_set_verbose", integer, pointer, integer)(request, 0))
            check(calibrate(request, str(calibration).encode(), 8, 64, None, None))
            print("Native DLL calibration passed", flush=True)
            captures = []
            # Cancellation avoids decoding: capture the same initial prediction,
            # first unchanged and then after native brightness correction.
            for mode in ("baseline", "native", "invalid"):
                if mode != "baseline":
                    check(set_lcs(request, str(calibration).encode(), b'{"brightness":0.1}'))
                calls, callback_errors = [], []

                @callback_type
                def callback(info, userdata):
                    try:
                        step = info.contents
                        assert step.struct_size == C.sizeof(Step)
                        assert step.modality == 0 and step.step == 0
                        assert step.total_steps == 1 and step.row_width == 96
                        assert step.frames == 1 and step.count == 384
                        values = list(step.denoised[:step.count])
                        assert all(math.isfinite(x) for x in values)
                        calls.append(values)
                        if mode == "invalid":
                            step.denoised[0] = float("nan")
                            return 0
                        return 1
                    except Exception as exc:
                        callback_errors.append(repr(exc))
                        return -1

                check(set_callback(request, callback, None))
                generation = pointer()
                check(start(request, None, None, C.byref(generation)))
                try:
                    status = wait(generation, -1)
                    assert not callback_errors, callback_errors
                    assert len(calls) == 1, gen_error(generation).decode()
                    if mode == "invalid":
                        assert status < 0 and b"non-finite" in gen_error(generation)
                    else:
                        assert status == -8, (status, gen_error(generation).decode())
                        captures.append(calls[0])
                finally:
                    gen_destroy(generation)
                print(f"DLL {mode} prediction/callback passed", flush=True)
            assert captures[0] != captures[1], "native LCS did not change the prediction"
        finally:
            destroy(request)
    print("Real DLL LCS and sampling callback smoke test passed", flush=True)


if __name__ == "__main__":
    main()
