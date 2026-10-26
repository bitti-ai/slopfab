"""Real-weight latent ending DLL test; tiny inputs, not a visual quality benchmark.

python tools/latent_end_anchor_smoke.py build/Release/slopfab.dll . [cuda|vulkan]
Exercises a one-sided lead-in, continuation to another ending, retained-handle
and file ownership, unequal context lengths, and further forward continuation.
"""
import ctypes as C
import math
import pathlib
import sys
import tempfile

from continuation_smoke import Output, read_archive
from latent_bridge_smoke import write_tensor


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
    end_gen = bind("request_set_end_anchor_generation", [handle, handle, C.c_int])
    end_file = bind("request_set_end_anchor_file", [handle, C.c_char_p, C.c_int])
    clear_end = bind("request_clear_end_anchor_latents", [handle])
    begin_gen = bind("request_set_continuation_generation", [handle, handle, C.c_int])
    begin_file = bind("request_set_continuation_file", [handle, C.c_char_p, C.c_int])
    start = bind("generation_start", [handle, handle, handle, C.POINTER(handle)])
    wait = bind("generation_wait", [handle, C.c_int])
    gen_error = bind("generation_error", [handle], C.c_char_p)
    output = bind("generation_output", [handle, C.POINTER(Output)])
    generations = []
    request = create()
    try:
        with tempfile.TemporaryDirectory(prefix="slopfab-latent-ending-") as tmp:
            work = pathlib.Path(tmp)
            embedding = work / "embedding.safetensors"
            write_tensor(embedding, "prompt_embedding", [1, 5120], [.1 * math.sin(i) for i in range(5120)])
            for kind, path in [(0, "transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors"),
                               (3, "vae/minimax_h3_video_vae_fp16.safetensors"),
                               (4, "vae/minimax_h3_audio_vae_fp32.safetensors")]:
                check(set_model(request, kind, str(root / "weights" / path).encode()))
            check(set_embedding(request, str(embedding).encode()))
            check(set_resolution(request, 32, 32))
            check(set_steps(request, 3))
            check(set_attention(request, b"flash2"))
            check(set_backend(request, backend == "vulkan"))
            check(retain(request, 1))

            def release(generation):
                gen_destroy(generation)
                generations.remove(generation)

            def run(name, frames, expected, seed):
                check(set_frames(request, frames))
                check(set_seed(request, seed))
                archive = work / (name + ".safetensors")
                check(save(request, str(archive).encode()))
                generation = handle()
                print(f"{backend}: {name}, expected {expected} frames", flush=True)
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

            def preserved(rows, source, suffix):
                assert (rows["video_rows"].endswith(source["video_rows"]) if suffix
                        else rows["video_rows"].startswith(source["video_rows"]))
                for c in range(2):
                    a, b = rows["audio_rows"], source["audio_rows"]
                    target = a[c * len(a)//2:(c + 1) * len(a)//2]
                    original = b[c * len(b)//2:(c + 1) * len(b)//2]
                    assert target.endswith(original) if suffix else target.startswith(original)

            ending, end_rows = run("ending", 39, 39, 11)
            check(end_gen(request, ending, 22))
            release(ending)  # request owns the snapshot
            # Invalid replacement must leave the previous ending attached.
            assert end_file(request, str(work / "ending.safetensors").encode(), 56) != 0
            lead, lead_rows = run("lead", 17, 56, 23)
            preserved(lead_rows, end_rows, True)
            print("PASS: one-sided lead-in preserves complete ending video and stereo audio", flush=True)

            check(clear_end(request))
            next_end, next_end_rows = run("next-ending", 22, 22, 29)
            check(begin_gen(request, lead, 22))
            check(end_gen(request, next_end, 5))  # different beginning/ending context sizes
            release(lead)
            release(next_end)
            joined_gen, joined = run("joined", 13, 107, 47)
            preserved(joined, lead_rows, False)
            preserved(joined, next_end_rows, True)
            print("PASS: continuation preserves both complete source clips", flush=True)

            check(begin_file(request, str(work / "lead.safetensors").encode(), 22))
            check(end_file(request, str(work / "next-ending.safetensors").encode(), 5))
            (work / "lead.safetensors").unlink()
            (work / "next-ending.safetensors").unlink()
            repeated_gen, repeated = run("file-joined", 13, 107, 47)
            assert repeated == joined
            print("PASS: file snapshots survive deletion and match retained-generation sources", flush=True)
            release(repeated_gen)

            check(begin_gen(request, joined_gen, 22))
            release(joined_gen)
            check(clear_end(request))
            _, extended = run("extended", 17, 124, 61)
            preserved(extended, joined, False)
            print("PASS: joined output supports further ordinary continuation; all output finite", flush=True)
    finally:
        destroy(request)
        for generation in generations:
            gen_destroy(generation)


if __name__ == "__main__":
    main()
