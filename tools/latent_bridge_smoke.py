"""Real-weight retained-latent bridge plumbing test, not a quality benchmark.

python tools/latent_bridge_smoke.py build/Release/slopfab.dll . [cuda|vulkan]
Uses a tiny synthetic embedding/reference with real Ref2VA and VAE weights.
Checks two retained sources, destroyed-handle ownership, persistent references,
file handoff, editable margins, exact retained regions and finite joined output.
"""
import ctypes as C
import json
import math
import pathlib
import struct
import sys
import tempfile

from continuation_smoke import Output, read_archive


def write_tensor(path, name, shape, values, metadata=None):
    payload = struct.pack(f"<{len(values)}f", *values)
    header = {name: {"dtype": "F32", "shape": shape, "data_offsets": [0, len(payload)]}}
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


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
    set_seed = bind("request_set_seed", [handle, C.c_uint64])
    set_attention = bind("request_set_attention", [handle, C.c_char_p])
    set_backend = bind("request_set_inference_backend", [handle, C.c_int])
    retain = bind("request_set_retain_latents", [handle, C.c_int])
    save = bind("request_set_save_latents", [handle, C.c_char_p])
    add_ref = bind("request_add_refmod", [handle, C.c_char_p, C.c_float, C.c_int])
    from_gen = bind("request_set_latent_bridge_generations", [handle, handle, handle, C.c_int, C.c_int, C.c_int])
    from_file = bind("request_set_latent_bridge_files", [handle, C.c_char_p, C.c_char_p, C.c_int, C.c_int, C.c_int])
    describe = bind("describe_plan", [handle, C.POINTER(handle)])
    free_string = bind("free_string", [handle], None)
    start = bind("generation_start", [handle, handle, handle, C.POINTER(handle)])
    wait = bind("generation_wait", [handle, C.c_int])
    gen_error = bind("generation_error", [handle], C.c_char_p)
    output = bind("generation_output", [handle, C.POINTER(Output)])
    generations = []
    request = create()
    try:
        with tempfile.TemporaryDirectory(prefix="slopfab-bridge-smoke-") as tmp:
            work = pathlib.Path(tmp)
            embedding = work / "embedding.safetensors"
            write_tensor(embedding, "prompt_embedding", [1, 5120], [.1 * math.sin(i) for i in range(5120)])
            reference = work / "original.safetensors"
            write_tensor(reference, "latent", [1, 24, 1, 2, 2], [.1 * math.cos(i) for i in range(96)],
                         {"refmod_meta": json.dumps({"kind": "image", "name": "original", "latent_h": 2, "latent_w": 2, "latent_t": 1})})
            for kind, path in [(0, "transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors"),
                               (3, "vae/minimax_h3_video_vae_fp16.safetensors"),
                               (4, "vae/minimax_h3_audio_vae_fp32.safetensors")]:
                check(set_model(request, kind, str(root / "weights" / path).encode()))
            check(set_embedding(request, str(embedding).encode()))
            check(set_resolution(request, 32, 32))
            check(set_steps(request, 3))
            check(set_attention(request, b"flash2"))
            check(set_backend(request, backend == "vulkan"))
            check(retain(request, 1))
            check(add_ref(request, str(reference).encode(), 1, 1))

            def run(name, frames, expected, seed):
                check(set_frames(request, frames))
                check(set_seed(request, seed))
                archive = work / (name + ".safetensors")
                check(save(request, str(archive).encode()))
                generation = handle()
                print(f"{backend}: {name}, expected {expected} joined frames", flush=True)
                check(start(request, None, None, C.byref(generation)))
                generations.append(generation)
                while True:
                    status = wait(generation, 1000)
                    if status == -7:
                        continue
                    if status:
                        raise RuntimeError(gen_error(generation).decode())
                    break
                result = Output()
                check(output(generation, C.byref(result)))
                assert (result.frames, result.width, result.height) == (expected, 32, 32)
                assert result.audio_channels == 2 and result.audio_frames > 0
                assert all(math.isfinite(result.video[i]) for i in range(result.video_float_count))
                assert all(math.isfinite(result.audio[i]) for i in range(result.audio_float_count))
                return generation, read_archive(archive)[1]

            left, left_rows = run("left", 56, 56, 11)
            right, right_rows = run("right", 73, 73, 29)
            check(from_gen(request, left, right, 17, 34, 22))
            for generation in generations:
                gen_destroy(generation)
            generations.clear()
            # Original reference remains in the request after endpoint attachment.
            text = handle()
            check(describe(request, C.byref(text)))
            try:
                assert "refmod" in C.string_at(text).decode()
            finally:
                free_string(text)
            _, joined = run("joined", 13, 158, 47)

            def check_regions(rows):
                # 56-frame left minus 17-frame margin: 12 retained video latents.
                # 73-frame right minus 34-frame head: drop 10 video latents.
                assert rows["video_rows"][:12 * 96 * 4] == left_rows["video_rows"][:12 * 96 * 4]
                assert rows["video_rows"].endswith(right_rows["video_rows"][10 * 96 * 4:])
                assert rows["video_rows"][12 * 96 * 4:17 * 96 * 4] != left_rows["video_rows"][12 * 96 * 4:]
                for c in range(2):
                    old_left = left_rows["audio_rows"]
                    old_right = right_rows["audio_rows"]
                    out = rows["audio_rows"]
                    assert out[c * len(out)//2:c * len(out)//2 + 65 * 32 * 4] == old_left[c * len(old_left)//2:c * len(old_left)//2 + 65 * 32 * 4]
                    suffix = old_right[c * len(old_right)//2 + 57 * 32 * 4:(c + 1) * len(old_right)//2]
                    assert out[(c + 1) * len(out)//2 - len(suffix):(c + 1) * len(out)//2] == suffix
                assert len(rows["audio_rows"]) == 263 * 64 * 4  # round(158 * 40/24)

            check_regions(joined)
            check(from_file(request, str(work / "left.safetensors").encode(),
                            str(work / "right.safetensors").encode(), 17, 34, 22))
            (work / "left.safetensors").unlink()
            (work / "right.safetensors").unlink()
            _, repeated = run("file-joined", 13, 158, 47)
            check_regions(repeated)
            assert joined == repeated
            print("PASS: retained/file bridges agree; original regions exact; joined video/audio finite", flush=True)
    finally:
        destroy(request)
        for generation in generations:
            gen_destroy(generation)


if __name__ == "__main__":
    main()
