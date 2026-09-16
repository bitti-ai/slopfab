"""Render an Animate comparison through the DLL, with no Qwen or tokenizer.

Requires NumPy, FFmpeg, the Viggle transformer/LoRA, converted fixed embedding,
and floating-point video/audio VAEs in the repository's weights directory.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess

import numpy as np


class Progress(C.Structure):
    _fields_ = [("stage", C.c_int32), ("step", C.c_int32),
                ("total_steps", C.c_int32), ("elapsed_seconds", C.c_double)]


class Output(C.Structure):
    _fields_ = [("video", C.POINTER(C.c_float)), ("video_float_count", C.c_size_t),
               ("channels", C.c_int32), ("frames", C.c_int32), ("width", C.c_int32), ("height", C.c_int32),
               ("audio", C.POINTER(C.c_float)), ("audio_float_count", C.c_size_t),
               ("audio_channels", C.c_int32), ("audio_sample_rate", C.c_int32), ("audio_frames", C.c_int64),
               ("fps", C.c_double)] + [(name, C.c_double) for name in
               ("seconds_conditioning", "seconds_denoise", "seconds_video_decode", "seconds_audio_decode", "seconds_total")] + [
               ("steps_computed", C.c_int32), ("steps_skipped", C.c_int32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("driving", type=Path)
    parser.add_argument("repainted", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=1248)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), default="cuda")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ffmpeg = str(root / "external/ffmpeg/bin/ffmpeg.exe")
    ffprobe = str(root / "external/ffmpeg/bin/ffprobe.exe")
    frames = max(22, ((args.frames - 5 + 16) // 17) * 17 + 5)
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32 or frames > 345:
        raise ValueError("Use a positive canvas divisible by 32 and at most 345 aligned frames")
    audio_path = args.output.with_suffix(".wav")
    latent_path = args.output.with_suffix(".safetensors")
    if any(p.exists() for p in (args.output, audio_path, latent_path)):
        raise FileExistsError("Choose new output paths; existing comparison artifacts are preserved")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(args.driving)]))
    sound = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    duration = min(float(probe["format"]["duration"]), (frames + 1) / 24, 15)
    if duration < 2:
        raise ValueError("Driving video must be at least two seconds")
    rate, channels = int(sound["sample_rate"]), int(sound["channels"])
    if channels not in (1, 2):
        raise ValueError("Driving soundtrack must be mono or stereo")
    dll = C.CDLL(str(args.dll.resolve()))
    handle = C.c_void_p

    def bind(name, arguments, result=C.c_int):
        fn = getattr(dll, name)
        fn.argtypes, fn.restype = arguments, result
        return fn

    error = bind("slopfab_last_error", [], C.c_char_p)

    def check(status):
        if status:
            raise RuntimeError(f"SlopFab {status}: {error().decode()}")

    request = bind("slopfab_request_create", [], handle)()
    video, generation = handle(), handle()
    try:
        check(bind("slopfab_request_set_animate", [handle, C.c_int, C.c_int])(request, 1, 1))
        check(bind("slopfab_request_set_resolution", [handle, C.c_int, C.c_int])(request, args.width, args.height))
        check(bind("slopfab_request_set_frames", [handle, C.c_int])(request, frames))
        check(bind("slopfab_request_set_seed", [handle, C.c_uint64])(request, args.seed))
        check(bind("slopfab_request_set_inference_backend", [handle, C.c_int])(request, int(args.backend == "vulkan")))
        check(bind("slopfab_request_set_verbose", [handle, C.c_int])(request, 1))
        set_model = bind("slopfab_request_set_model_path", [handle, C.c_int, C.c_char_p])
        for kind, relative in [(0, "transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors"),
                               (1, "must-not-load-qwen.safetensors"), (2, "must-not-load-tokenizer.json"),
                               (3, "vae/minimax_h3_video_vae_fp16.safetensors"),
                               (4, "vae/minimax_h3_audio_vae_fp32.safetensors")]:
            check(set_model(request, kind, str(root / "weights" / relative).encode()))
        check(bind("slopfab_request_set_prompt_embedding_path", [handle, C.c_char_p])(
            request, str(root / "weights/conditioning/viggle_animate.safetensors").encode()))
        check(bind("slopfab_request_add_lora", [handle, C.c_char_p, C.c_float])(
            request, str(root / "weights/loras/viggle_animate_distillation_bf16.safetensors").encode(), 1))
        check(bind("slopfab_request_add_reference_image", [handle, C.c_char_p])(request, str(args.repainted.resolve()).encode()))
        check(bind("slopfab_request_set_save_latents", [handle, C.c_char_p])(request, str(latent_path.resolve()).encode()))
        check(bind("slopfab_reference_video_create", [C.c_double, C.POINTER(handle)])(duration, C.byref(video)))
        append = bind("slopfab_reference_video_append_rgb24",
                      [handle, C.POINTER(C.c_uint8), C.c_size_t, C.c_int, C.c_int, C.c_size_t, C.c_double])
        # Both implementations can use this same 24-fps, target-sized reference.
        command = [ffmpeg, "-v", "error", "-i", str(args.driving), "-t", str(duration), "-an",
                   "-vf", f"fps=24,scale={args.width}:{args.height}:flags=lanczos", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
        frame_bytes = args.width * args.height * 3
        with subprocess.Popen(command, stdout=subprocess.PIPE) as decoder:
            index = 0
            while True:
                raw = decoder.stdout.read(frame_bytes)
                if not raw:
                    break
                if len(raw) != frame_bytes:
                    raise RuntimeError("Truncated decoded frame")
                if index / 24 < duration:
                    pixels = (C.c_uint8 * frame_bytes).from_buffer_copy(raw)
                    check(append(video, pixels, frame_bytes, args.width, args.height, args.width * 3, index / 24))
                index += 1
            if decoder.wait():
                raise RuntimeError("Video decode failed")
        pcm = np.frombuffer(subprocess.check_output([ffmpeg, "-v", "error", "-i", str(args.driving),
            "-t", str(duration), "-vn", "-f", "f32le", "pipe:1"]), dtype="<f4").copy()
        pcm = np.clip(pcm[:int(duration * rate) * channels], -1, 1)
        check(bind("slopfab_reference_video_set_audio_f32", [handle, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int, C.c_double])(
            video, pcm.ctypes.data_as(C.POINTER(C.c_float)), pcm.size, channels, rate, 0))
        check(bind("slopfab_request_add_reference_video", [handle, handle])(request, video))
        callback_type = C.CFUNCTYPE(None, C.POINTER(Progress), handle)

        @callback_type
        def progress(value, _):
            p = value.contents
            print(f"stage={p.stage} step={p.step}/{p.total_steps} elapsed={p.elapsed_seconds:.1f}s", flush=True)

        check(bind("slopfab_generation_start", [handle, callback_type, handle, C.POINTER(handle)])(request, progress, None, C.byref(generation)))
        wait = bind("slopfab_generation_wait", [handle, C.c_int])
        while True:
            status = wait(generation, 1000)
            if status == -7:
                continue
            if status:
                raise RuntimeError(bind("slopfab_generation_error", [handle], C.c_char_p)(generation).decode())
            break
        output = Output()
        check(bind("slopfab_generation_output", [handle, C.POINTER(Output)])(generation, C.byref(output)))
        assert (output.frames, output.width, output.height) == (frames, args.width, args.height)
        assert output.steps_computed == 3 and output.steps_skipped == 0
        assert output.video_float_count == output.channels * frames * args.width * args.height
        pixels = np.ctypeslib.as_array(output.video, shape=(output.video_float_count,))
        for offset in range(0, pixels.size, 1024 * 1024):
            assert np.isfinite(pixels[offset:offset + 1024 * 1024]).all()
        assert output.audio_frames > 0 and output.audio_channels == 2
        audio = np.ctypeslib.as_array(output.audio, shape=(output.audio_float_count,))
        assert np.isfinite(audio).all()
        subprocess.run([ffmpeg, "-v", "error", "-n", "-f", "f32le", "-ar", str(output.audio_sample_rate),
                        "-ac", str(output.audio_channels), "-i", "pipe:0", "-c:a", "pcm_f32le", str(audio_path)],
                       input=audio.astype("<f4").tobytes(), check=True)
        rgba = (C.c_uint8 * (args.width * args.height * 4))()
        get_frame = bind("slopfab_generation_frame_rgba8", [handle, C.c_int, C.POINTER(C.c_uint8), C.c_size_t])
        with subprocess.Popen([ffmpeg, "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgba", "-s",
                               f"{args.width}x{args.height}", "-r", "24", "-i", "pipe:0", "-i", str(audio_path),
                               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(args.output)],
                              stdin=subprocess.PIPE) as encoder:
            for index in range(frames):
                check(get_frame(generation, index, rgba, len(rgba)))
                encoder.stdin.write(bytes(rgba))
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("Output encode failed")
        print(f"Runtime checks passed: {frames} frames, three evaluations, pinned audio; {args.output}. "
              "Visual quality requires separate inspection.", flush=True)
    finally:
        if generation:
            bind("slopfab_generation_destroy", [handle], None)(generation)
        if video:
            bind("slopfab_reference_video_destroy", [handle], None)(video)
        if request:
            bind("slopfab_request_destroy", [handle], None)(request)


if __name__ == "__main__":
    main()
