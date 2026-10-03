"""Real-VAE C ABI export checks; no third-party Python packages required.

Usage: python tools/refmod_export_smoke.py LIBRARY REPO [cuda|vulkan] [OUTPUT_DIR]
Creates image/video/audio RefMods and a mixed bundle from raw buffers, verifies
their tensor layouts, finiteness, deterministic member equality and reloading.
Only the video/audio VAE checkpoints are set on the export requests.
"""
import ctypes as C
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

library, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
backend = sys.argv[3] if len(sys.argv) > 3 else "cuda"
scratch = tempfile.TemporaryDirectory(prefix="slopfab-media-export-")
output = Path(sys.argv[4]).resolve() if len(sys.argv) > 4 else Path(scratch.name)
output.mkdir(parents=True, exist_ok=True)
dll = C.CDLL(str(library))
H = C.c_void_p


def bind(name, args, result=C.c_int):
    fn = getattr(dll, "slopfab_" + name)
    fn.argtypes, fn.restype = args, result
    return fn


error = bind("last_error", [], C.c_char_p)
create = bind("request_create", [], H)
destroy = bind("request_destroy", [H], None)
set_backend = bind("request_set_inference_backend", [H, C.c_int])
set_model = bind("request_set_model_path", [H, C.c_int, C.c_char_p])
add_image = bind("request_add_reference_image", [H, C.c_char_p])
add_audio = bind("request_add_reference_audio_f32", [H, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int])
video_create = bind("reference_video_create", [C.c_double, C.POINTER(H)])
video_destroy = bind("reference_video_destroy", [H], None)
video_frame = bind("reference_video_append_rgb24", [H, C.POINTER(C.c_uint8), C.c_size_t, C.c_int, C.c_int, C.c_size_t, C.c_double])
video_audio = bind("reference_video_set_audio_f32", [H, C.POINTER(C.c_float), C.c_size_t, C.c_int, C.c_int, C.c_double])
add_video = bind("request_add_reference_video", [H, H])
export = bind("export_refmod", [H, C.c_char_p, C.c_char_p, C.c_char_p, C.c_int])
add_mod = bind("request_add_refmod", [H, C.c_char_p, C.c_float, C.c_int])


def check(status):
    if status:
        raise RuntimeError(f"status {status}: {error().decode()}")


def archive(path):
    data = path.read_bytes()
    n, = struct.unpack_from("<Q", data)
    header = json.loads(data[8:8+n])
    meta = json.loads(header["__metadata__"]["refmod_meta"])
    tensors = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        assert info["dtype"] == "F32", info
        begin, end = info["data_offsets"]
        payload = data[8+n+begin:8+n+end]
        assert len(payload) == math.prod(info["shape"]) * 4
        assert all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", payload))
        assert any(v[0] != 0 for v in struct.iter_unpack("<f", payload))
        tensors[key] = (info["shape"], payload)
    return meta, tensors


image = output / "input.ppm"
pixels = bytes((x * 7 + y * 3 + c * 53) % 256 for y in range(32) for x in range(32) for c in range(3))
image.write_bytes(b"P6\n32 32\n255\n" + pixels)
# Mono at 16 kHz exercises stereo expansion and resampling to the H3 codec rate.
pcm = (C.c_float * 32000)(*(.2 * math.sin(i * 2 * math.pi * 220 / 16000) for i in range(32000)))
handles = []
video = H()
try:
    check(video_create(2.0, C.byref(video)))
    for i in range(4):
        frame = (C.c_uint8 * len(pixels)).from_buffer_copy(bytes((v + i * 13) % 256 for v in pixels))
        check(video_frame(video, frame, len(frame), 32, 32, 96, i * .5))

    def request(kinds, soundtrack=False):
        req = create()
        assert req
        handles.append(req)
        check(set_backend(req, int(backend == "vulkan")))
        if "image" in kinds or "video" in kinds:
            check(set_model(req, 3, str(root / "weights/vae/minimax_h3_video_vae_fp16.safetensors").encode()))
        if "audio" in kinds or soundtrack:
            check(set_model(req, 4, str(root / "weights/vae/minimax_h3_audio_vae_fp32.safetensors").encode()))
        if "image" in kinds:
            check(add_image(req, str(image).encode()))
        if "video" in kinds:
            if soundtrack:
                check(video_audio(video, pcm, len(pcm), 1, 16000, 0))
            check(add_video(req, video))
        if "audio" in kinds:
            check(add_audio(req, pcm, len(pcm), 1, 16000))
        return req

    expected = {"image": [1, 24, 1, 2, 2], "video": [1, 24, 12, 2, 2], "audio": [1, 32, 2, 80]}
    standalone = {}
    for kind in expected:
        req = request([kind])
        path = output / f"{kind}.safetensors"
        print(f"encoding {kind} ({backend})", flush=True)
        check(export(req, str(path).encode(), b"raw", b"export smoke", 32))
        meta, tensors = archive(path)
        assert meta["kind"] == kind and meta["_format_version"] == 4
        assert meta["description"] == "export smoke"
        assert tensors["latent"][0] == expected[kind]
        standalone[kind] = tensors["latent"]
        if kind == "audio":
            samples = struct.unpack("<" + "f" * (len(tensors["latent"][1]) // 4), tensors["latent"][1])
            for c in range(32):
                assert samples[c*160:c*160+80] == samples[c*160+80:c*160+160]

    req = request(["image", "video", "audio"], soundtrack=True)
    mixed = output / "mixed.safetensors"
    print(f"encoding mixed assets and video soundtrack ({backend})", flush=True)
    check(export(req, str(mixed).encode(), b"mixed", b"raw media", 32))
    meta, tensors = archive(mixed)
    assert meta["kind"] == "bundle" and meta["_format_version"] == 5
    assert [m["kind"] for m in meta["members"]] == ["image", "video", "audio", "audio"]
    for i, kind in enumerate(["image", "video", "audio", "audio"]):
        assert tensors[f"ref_{i}"] == standalone[kind], (i, kind)
    image.unlink()
    loader = create()
    handles.append(loader)
    for name in ["image", "video", "audio", "mixed"]:
        check(add_mod(loader, str(output / f"{name}.safetensors").encode(), 1, 1))
    print(f"PASS: {backend} raw image/video/audio exports, soundtrack, mixed member equality, and reload; {output}", flush=True)
finally:
    for req in handles:
        destroy(req)
    if video:
        video_destroy(video)
    scratch.cleanup()
