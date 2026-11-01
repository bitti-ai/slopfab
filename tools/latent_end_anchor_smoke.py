"""Real-weight latent ending DLL test; tiny inputs, not a visual quality benchmark.

python tools/latent_end_anchor_smoke.py build/Release/slopfab.dll . [cuda|vulkan]
Add --source-noise 0.7 to test source initialization and the pure-noise endpoint.
Use --keystone instead to test persistent original-clip anchors and late locking.
Exercises a one-sided lead-in, continuation to another ending, retained-handle
and file ownership, unequal context lengths, and further forward continuation.
"""
import argparse
import ctypes as C
import math
import pathlib
import tempfile

from continuation_smoke import Output, read_archive
from latent_bridge_smoke import write_tensor


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
    parser.add_argument("library", type=pathlib.Path)
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("backend", nargs="?", default="cuda", choices=("cuda", "vulkan"))
    parser.add_argument("--source-noise", type=float)
    parser.add_argument("--keystone", action="store_true")
    parser.add_argument("--steps", type=int, default=3, help="Denoising updates")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be at least one")
    if args.source_noise is not None and not 0 < args.source_noise < 1:
        parser.error("source-noise must be in (0,1) to test a nontrivial mixture")
    if args.keystone and (args.source_noise is not None or args.steps < 2):
        parser.error("keystone needs at least two updates and no source-noise")
    dll = C.CDLL(str(args.library.resolve()))
    root, backend = args.root.resolve(), args.backend
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
    source_init = bind("request_set_continuation_source_init", [handle, C.c_int, C.c_float])
    key_gen = bind("request_set_keystone_generation", [handle, handle, C.c_int, C.c_int])
    key_file = bind("request_set_keystone_file", [handle, C.c_char_p, C.c_int, C.c_int])
    clear_key = bind("request_clear_keystone", [handle])
    lock_overlap = bind("request_set_continuation_lock_overlap", [handle, C.c_int])
    callback_type = C.CFUNCTYPE(C.c_int, C.POINTER(Step), handle)
    set_callback = bind("request_set_sampling_callback", [handle, callback_type, handle])
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
            check(set_steps(request, args.steps))
            check(set_attention(request, b"flash2"))
            check(set_backend(request, backend == "vulkan"))
            check(retain(request, 1))
            snapshots = {}
            current_steps = []

            @callback_type
            def capture(pointer, _):
                step = pointer.contents
                current_steps.append((step.modality, step.step, step.frames, step.count,
                                      C.string_at(step.denoised, step.count * 4),
                                      step.sigma, C.string_at(step.latent, step.count * 4)))
                return 0

            if args.keystone:
                check(set_callback(request, capture, None))

            def release(generation):
                gen_destroy(generation)
                generations.remove(generation)

            def run(name, frames, expected, seed):
                current_steps.clear()
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
                snapshots[name] = list(current_steps)
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
            if args.keystone:
                check(begin_gen(request, ending, 22))
                check(lock_overlap(request, 1))
                plain_gen, _ = run("locked-without-keystone", 17, 56, 23)
                release(plain_gen)
                check(key_gen(request, ending, 0, 1))
                baseline_gen, baseline = run("front-only", 17, 56, 23)
                release(baseline_gen)
                plain_first = next(s for s in snapshots["locked-without-keystone"] if s[:2] == (0, 0))
                key_first = next(s for s in snapshots["front-only"] if s[:2] == (0, 0))
                # At full noise the target lock carries no source signal. Only
                # the corrected clean guide can make Keystone affect this call.
                assert plain_first[5] == key_first[5] == 1.0
                assert plain_first[6] == key_first[6]
                held_bytes = 7 * 96 * 4  # 22 frames, one packed row per latent at 32x32
                assert plain_first[4][held_bytes:] != key_first[4][held_bytes:], \
                    "Keystone must influence new video before the noisy target lock becomes visible"
                check(key_gen(request, ending, 1, 1))
                release(ending)
                joined_gen, joined = run("keystone", 17, 56, 23)
                preserved(joined, end_rows, False)
                assert joined["video_rows"] != baseline["video_rows"]
                before, after = snapshots["front-only"], snapshots["keystone"]
                assert before and after
                assert [s[:4] for s in before] == [s[:4] for s in after]
                # Until activation, the model sees exactly the same sequence.
                assert [s for s in before if s[1] == 0] == [s for s in after if s[1] == 0]
                assert before != after
                check(key_file(request, str(work / "ending.safetensors").encode(), 1, 1))
                (work / "ending.safetensors").unlink()
                repeat_gen, repeated = run("keystone-file", 17, 56, 23)
                assert repeated == joined
                release(repeat_gen)
                check(begin_gen(request, joined_gen, 22))
                release(joined_gen)
                next_gen, extended = run("keystone-next", 17, 73, 31)
                preserved(extended, joined, False)
                check(begin_gen(request, next_gen, 22))
                release(next_gen)
                check(clear_key(request))
                _, plain = run("plain-after-keystone", 17, 90, 37)
                preserved(plain, extended, False)
                print("PASS: keystone retained/file parity, owned snapshots, call-only context, "
                      "front-guide influence at full noise, identical pre-capstone predictions, "
                      "unchanged sampler lengths, "
                      "persistent source and further ordinary continuation", flush=True)
                return
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
            if args.source_noise is not None:
                check(source_init(request, 1, 1))
                pure_gen, pure_rows = run("source-noise-one", 13, 107, 47)
                assert pure_rows == joined, "noise strength 1 must preserve existing locked generation"
                release(pure_gen)
                release(joined_gen)
                baseline = joined
                check(source_init(request, 1, args.source_noise))
                joined_gen, joined = run("source-initialized", 13, 107, 47)
                assert joined["video_rows"] != baseline["video_rows"], "source initialization must affect new video"
                print("PASS: source initialization changes generated video; strength 1 matches baseline", flush=True)
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
