"""Exercise both music families through the C ABI with real local checkpoints.

Checks audio ownership, geometry, finite samples, seeded repeatability,
progress and cancellation. Inference runs inside the DLL.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import threading
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--library", required=True, type=Path)
p.add_argument("--weights", default="weights", type=Path)
p.add_argument("--family", choices=("yue2", "minimax_music3", "both"), default="both")
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--output-dir", type=Path, help="Save raw float32 audio for repeatability diagnosis")
p.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
p.add_argument("--duration", type=float, default=.08, help="Audio frame budget in seconds")
p.add_argument("--steps", type=int, default=1, help="Flow integration steps")
p.add_argument("--planning", choices=("off", "melody", "full"), default="off", help="YuE2 ABC planning")
p.add_argument("--abc-max-tokens", type=int, default=4096, help="YuE2 ABC planning token budget")
p.add_argument("--timings-json", type=Path, help="Save per-stage timings for performance comparisons")
p.add_argument("--cancel-stages", action="store_true", help="Also cancel inside conditioning/flow and check recovery")
a = p.parse_args()
if not math.isfinite(a.duration) or not 0 < a.duration <= 300 or a.steps < 1 or a.repeats < 2:
    p.error("duration must be in (0,300], steps positive, and repeats at least two")
timings = []
dll = C.CDLL(str(a.library.resolve()))
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


class Progress(C.Structure):
    _fields_ = [("stage", C.c_int32), ("step", C.c_int32), ("total_steps", C.c_int32),
                ("elapsed_seconds", C.c_double)]


ProgressCallback = C.CFUNCTYPE(None, C.POINTER(Progress), handle)


def check_music_progress(events, steps_computed, family):
    stages = [event[0] for i, event in enumerate(events) if i == 0 or event[0] != events[i - 1][0]]
    assert stages == [0, 3, 2, 4, 6, 7, 8], f"unexpected {family} stage transitions: {stages}"
    elapsed = [event[3] for event in events]
    assert all(math.isfinite(t) and t >= 0 for t in elapsed) and elapsed == sorted(elapsed)
    for stage in (2, 4):
        phase = [(step, total) for s, step, total, _ in events if s == stage]
        assert phase[0][0] == -1 and phase[0][1] > 0, f"missing stage {stage} entry budget"
        if stage == 2:
            budget = math.ceil(a.duration * 25)
            if family == "yue2" and a.planning != "off":
                budget += a.abc_max_tokens
            assert phase[0][1] == budget, "conditioning budget includes only output tokens/frames"
        last_step, last_total = -1, phase[0][1]
        for step, total in phase:
            assert step in (last_step, last_step + 1), f"stage {stage} skipped or reset progress"
            assert -1 <= step < total <= last_total, f"invalid stage {stage} progress: {step}/{total}"
            if stage == 4:
                assert total == steps_computed, "flow budget must span all chunks"
            last_step, last_total = step, total
        assert last_step + 1 == last_total, f"stage {stage} never reported completion"
        if stage == 2 and family == "minimax_music3":
            frames = last_step + 1
            windows = 1 if frames <= 200 else (frames - 101) // 100 + 1
            assert steps_computed == windows * a.steps, "flow total must match actual acoustic windows"


create = bind("slopfab_request_create", [], handle)
destroy = bind("slopfab_generation_destroy", [handle], None)
cancel = bind("slopfab_generation_cancel", [handle], None)
start = bind("slopfab_generation_start", [handle, handle, handle, C.POINTER(handle)])
wait = bind("slopfab_generation_wait", [handle, C.c_int])
paths = {"yue2": [(0, "yue2_3b_int8_convrot.safetensors")], "minimax_music3": [
    (0, "minimax_music3_dit_int8_convrot.safetensors"),
    (1, "text_encoder/minimax_music3_text_encoder_pruned_int8_convrot.safetensors"),
    (4, "minimax_music3_dav.safetensors")]}
for family in paths if a.family == "both" else [a.family]:
    request, generation = create(), handle()
    try:
        for kind, path in paths[family]:
            check(bind("slopfab_request_set_model_path", [handle, C.c_int, C.c_char_p])(
                request, kind, str((a.weights / path).resolve()).encode()))
        # Leave the family empty to exercise checkpoint detection in the DLL.
        check(bind("slopfab_request_set_prompt", [handle, C.c_char_p])(request, b"Warm instrumental piano"))
        settings = {"duration": a.duration}
        if family == "yue2":
            settings.update(planning=a.planning, abc_max_tokens=a.abc_max_tokens)
        check(bind("slopfab_request_set_music_settings", [handle, C.c_char_p])(request, json.dumps(settings).encode()))
        check(bind("slopfab_request_set_steps", [handle, C.c_int])(request, a.steps))
        check(bind("slopfab_request_set_seed", [handle, C.c_uint64])(request, 42))
        check(bind("slopfab_request_set_verbose", [handle, C.c_int])(request, 0))
        check(bind("slopfab_request_set_inference_backend", [handle, C.c_int])(request, 1 if a.backend == "vulkan" else 0))
        if a.backend == "vulkan":
            check(bind("slopfab_request_set_vulkan_arithmetic", [handle, C.c_int])(request, 1))
        digests = []
        first_samples = None
        for repeat in range(a.repeats):
            events = []

            @ProgressCallback
            def record_progress(progress, _):
                value = progress.contents
                events.append((value.stage, value.step, value.total_steps, value.elapsed_seconds))

            callback = C.cast(record_progress, handle)
            check(start(request, callback, None, C.byref(generation)))
            if wait(generation, -1):
                raise RuntimeError(bind("slopfab_generation_error", [handle], C.c_char_p)(generation).decode())
            output = Output()
            check(bind("slopfab_generation_output", [handle, C.POINTER(Output)])(generation, C.byref(output)))
            assert output.video_float_count == 0 and not output.video
            assert output.audio_channels == 2
            assert output.audio_sample_rate == (48000 if family == "yue2" else 44100)
            if a.duration == .08:
                assert output.audio_frames == (3776 if family == "yue2" else 3072)
            else:
                # Longer generations can stop at EOS; codec/window crops vary.
                assert 0 < output.audio_frames <= (math.ceil(a.duration * 25) / 25 + 1) * output.audio_sample_rate
            assert output.audio_float_count == output.audio_frames * 2
            samples = output.audio[:output.audio_float_count]
            assert all(math.isfinite(v) for v in samples) and max(samples) != min(samples)
            assert output.steps_computed >= a.steps and output.steps_skipped == 0
            if a.duration == .08:
                assert output.steps_computed == a.steps
            check_music_progress(events, output.steps_computed, family)
            timing = {"family": family, "backend": a.backend, "repeat": repeat,
                      "duration": a.duration, "steps": a.steps, "audio_frames": output.audio_frames}
            if family == "yue2":
                timing.update(planning=a.planning, abc_max_tokens=a.abc_max_tokens)
            timing.update({name: getattr(output, name) for name in
                           ("seconds_conditioning", "seconds_denoise", "seconds_audio_decode", "seconds_total")})
            timings.append(timing)
            print(json.dumps(timing), flush=True)
            digests.append(hashlib.sha256(C.string_at(output.audio, output.audio_float_count * 4)).hexdigest())
            if a.output_dir:
                a.output_dir.mkdir(parents=True, exist_ok=True)
                (a.output_dir / f"{a.backend}-{family}-{repeat}.f32").write_bytes(C.string_at(output.audio, output.audio_float_count * 4))
            if first_samples is None:
                first_samples = samples
            elif digests[-1] != digests[0]:
                difference = max(abs(x - y) for x, y in zip(first_samples, samples))
                raise AssertionError(f"{a.backend} {family}: same-seed samples differ on repeat {repeat}; max error {difference}")
            destroy(generation)
            generation = handle()
        assert len(digests) >= 2, "at least two repeats required"
        check(start(request, None, None, C.byref(generation)))
        cancel(generation)
        assert wait(generation, -1) == -8, "expected cancellation"
        destroy(generation)
        generation = handle()
        if a.cancel_stages:
            settings = {"duration": .08}
            if family == "yue2":
                settings["planning"] = "off"
            check(bind("slopfab_request_set_music_settings", [handle, C.c_char_p])(request, json.dumps(settings).encode()))
            check(bind("slopfab_request_set_steps", [handle, C.c_int])(request, 1))
            for stage in (2, 4):  # conditioning, denoising
                ready = threading.Event()
                visits = [0]

                @ProgressCallback
                def on_progress(progress, _):
                    if progress.contents.stage == stage:
                        visits[0] += 1
                        # Cancel after several blocks, with active GPU temporaries.
                        if visits[0] == 8:
                            ready.wait()
                            cancel(generation)

                check(start(request, C.cast(on_progress, handle), None, C.byref(generation)))
                ready.set()
                assert wait(generation, -1) == -8 and visits[0] >= 8, f"expected cancellation inside stage {stage}"
                destroy(generation)
                generation = handle()
            # Ensure cancellation released the generation claim and GPU state.
            check(start(request, None, None, C.byref(generation)))
            check(wait(generation, -1))
            destroy(generation)
            generation = handle()
            print(f"{a.backend} {family}: in-stage cancellation and recovery passed", flush=True)
        print(f"{a.backend} {family}: conditioning and completed flow progress passed", flush=True)
        print(f"{a.backend} {family}: audio geometry, finite samples, seed repeatability and cancellation passed", flush=True)
    finally:
        if generation:
            destroy(generation)
        bind("slopfab_request_destroy", [handle], None)(request)
if a.timings_json:
    a.timings_json.parent.mkdir(parents=True, exist_ok=True)
    a.timings_json.write_text(json.dumps(timings, indent=2) + "\n", encoding="utf-8")
