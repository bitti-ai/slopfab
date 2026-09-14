"""Real-weight DLL continuation plumbing smoke test (not a quality benchmark).

python tools/continuation_smoke.py path/to/slopfab.dll . [cuda|vulkan]
Uses a fixed test embedding, 32x32 video and one model evaluation per window.
Tests retained-handle and file handoffs, normalized prefix preservation, archive
roundtripping, and finite video/audio on three consecutive generations.
"""
import ctypes as C
import json
import math
import pathlib
import struct
import sys
import tempfile


class Output(C.Structure):
    _fields_ = [("video", C.POINTER(C.c_float)), ("video_float_count", C.c_size_t),
                ("channels", C.c_int32), ("frames", C.c_int32), ("width", C.c_int32), ("height", C.c_int32),
                ("audio", C.POINTER(C.c_float)), ("audio_float_count", C.c_size_t),
                ("audio_channels", C.c_int32), ("audio_sample_rate", C.c_int32), ("audio_frames", C.c_int64),
                ("fps", C.c_double)] + [(name, C.c_double) for name in
                ("seconds_conditioning", "seconds_denoise", "seconds_video_decode", "seconds_audio_decode", "seconds_total")] + [
                ("steps_computed", C.c_int32), ("steps_skipped", C.c_int32)]


def read_archive(path):
    data = path.read_bytes()
    n, = struct.unpack_from("<Q", data)
    header = json.loads(data[8:8+n])
    tensors = {key: data[8+n+value["data_offsets"][0]:8+n+value["data_offsets"][1]]
               for key, value in header.items() if key != "__metadata__"}
    return header, tensors


def main():
    dll = C.CDLL(str(pathlib.Path(sys.argv[1]).resolve()))
    root = pathlib.Path(sys.argv[2]).resolve()
    backend = sys.argv[3] if len(sys.argv) > 3 else "cuda"
    assert backend in ("cuda", "vulkan")
    handle = C.c_void_p

    def bind(name, args, result=C.c_int):
        fn = getattr(dll, "slopfab_" + name)
        fn.argtypes, fn.restype = args, result
        return fn

    error = bind("last_error", [], C.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(f"{status}: {error().decode()}")

    create = bind("request_create", [], handle)
    destroy = bind("request_destroy", [handle], None)
    gen_destroy = bind("generation_destroy", [handle], None)
    set_model = bind("request_set_model_path", [handle, C.c_int, C.c_char_p])
    set_embedding = bind("request_set_prompt_embedding_path", [handle, C.c_char_p])
    set_resolution = bind("request_set_resolution", [handle, C.c_int, C.c_int])
    set_frames = bind("request_set_frames", [handle, C.c_int])
    set_steps = bind("request_set_steps", [handle, C.c_int])
    set_attention = bind("request_set_attention", [handle, C.c_char_p])
    set_backend = bind("request_set_inference_backend", [handle, C.c_int])
    retain = bind("request_set_retain_latents", [handle, C.c_int])
    save = bind("request_set_save_latents", [handle, C.c_char_p])
    from_gen = bind("request_set_continuation_generation", [handle, handle, C.c_int])
    from_file = bind("request_set_continuation_file", [handle, C.c_char_p, C.c_int])
    save_gen = bind("generation_save_latents", [handle, C.c_char_p])
    start = bind("generation_start", [handle, handle, handle, C.POINTER(handle)])
    wait = bind("generation_wait", [handle, C.c_int])
    gen_error = bind("generation_error", [handle], C.c_char_p)
    output = bind("generation_output", [handle, C.POINTER(Output)])

    previous = handle()
    request = None
    generation = handle()
    with tempfile.TemporaryDirectory(prefix="slopfab-continuation-smoke-") as tmp:
        work = pathlib.Path(tmp)
        embedding = work / "embedding.safetensors"
        header = json.dumps({"prompt_embedding": {"dtype": "F32", "shape": [1, 5120], "data_offsets": [0, 20480]}}).encode()
        header += b" " * (-len(header) % 8)
        embedding.write_bytes(struct.pack("<Q", len(header)) + header +
                              struct.pack("<5120f", *(.1 * math.sin(i) for i in range(5120))))
        prior_tensors = None
        try:
            for iteration, expected_frames in enumerate((22, 39, 56)):
                request = create()
                for kind, path in [(0, "transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors"),
                                   (3, "vae/minimax_h3_video_vae_fp16.safetensors"),
                                   (4, "vae/minimax_h3_audio_vae_fp32.safetensors")]:
                    check(set_model(request, kind, str(root / "weights" / path).encode()))
                check(set_embedding(request, str(embedding).encode()))
                check(set_resolution(request, 32, 32))
                check(set_frames(request, 22 if iteration == 0 else 17))
                check(set_steps(request, 2))
                check(set_attention(request, b"flash2"))
                check(set_backend(request, backend == "vulkan"))
                check(retain(request, 1))
                archive = work / f"clip-{iteration}.safetensors"
                check(save(request, str(archive).encode()))
                if iteration == 1:
                    check(from_gen(request, previous, 22))
                elif iteration == 2:
                    prior_file = work / "clip-1.safetensors"
                    check(from_file(request, str(prior_file).encode(), 22))
                    prior_file.unlink()
                if previous:
                    gen_destroy(previous)
                    previous = handle()
                print(f"{backend}: generation {iteration}, expected {expected_frames} joined frames", flush=True)
                check(start(request, None, None, C.byref(generation)))
                destroy(request)
                request = None
                while True:
                    status = wait(generation, 1000)
                    if status == -7:
                        continue
                    if status:
                        raise RuntimeError(gen_error(generation).decode())
                    break
                result = Output()
                check(output(generation, C.byref(result)))
                assert (result.frames, result.width, result.height) == (expected_frames, 32, 32)
                assert result.audio_channels == 2 and result.audio_frames > 0
                assert all(math.isfinite(result.video[i]) for i in range(result.video_float_count))
                assert all(math.isfinite(result.audio[i]) for i in range(result.audio_float_count))
                saved_copy = work / "copy.safetensors"
                check(save_gen(generation, str(saved_copy).encode()))
                assert archive.read_bytes() == saved_copy.read_bytes()
                meta, tensors = read_archive(archive)
                assert int(meta["__metadata__"]["frames"]) == expected_frames
                if prior_tensors:
                    assert tensors["video_rows"].startswith(prior_tensors["video_rows"])
                    old = prior_tensors["audio_rows"]
                    new = tensors["audio_rows"]
                    for channel in range(2):
                        assert new[channel*len(new)//2:channel*len(new)//2+len(old)//2] == old[channel*len(old)//2:(channel+1)*len(old)//2]
                prior_tensors = tensors
                previous, generation = generation, handle()
                print(f"PASS: {result.frames} frames, {result.audio_frames} stereo samples; prefix and archive exact", flush=True)
        finally:
            if request:
                destroy(request)
            if generation:
                gen_destroy(generation)
            if previous:
                gen_destroy(previous)


if __name__ == "__main__":
    main()
