"""Run H3 Image rendering checks sequentially on CUDA and portable Vulkan.

Requires Pillow, numpy, local weights, and a source image. Optional --start-at
accepts an ISO timestamp with timezone. Waits for low GPU memory use before
each case; never terminates other applications. Artifacts and individual logs
are retained, with a summary.json updated after each case.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exe", "library", "transformer", "text-encoder", "vae", "source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start-at", type=datetime.fromisoformat)
    parser.add_argument("--max-used-mib", type=int, default=5000)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--case", action="append", help="Only run this named case; repeatable")
    args = parser.parse_args()
    if args.start_at and args.start_at.tzinfo is None:
        parser.error("--start-at must include a timezone offset")
    for name in ("exe", "library", "transformer", "text_encoder", "vae", "source"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            parser.error(f"missing {name}: {path}")
        setattr(args, name, path)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    config = args.output / "settings.json"
    config.write_text('{"schema_version":1,"weight_paths":[]}', encoding="utf-8")
    env = dict(os.environ, SLOPFAB_NO_AUTO_UPDATE="1")
    # Normalize only the fixture dimensions; the actual edit must preserve these bytes.
    source = Image.open(args.source).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    source_path = args.output / "source.png"
    source.save(source_path)
    rgb = np.array(source)
    reference_path = args.output / "blue-reference.png"
    Image.new("RGB", (256, 256), (20, 60, 220)).save(reference_path)
    yy, xx = np.mgrid[:512, :512]
    radius = np.hypot(xx - 256, yy - 256)
    coverage = np.where((radius < 190) & (radius > 48), 255, 0).astype(np.uint8)
    coverage[(radius >= 180) & (radius < 190)] = 128
    mask_path = args.output / "mask.png"
    Image.fromarray(coverage).save(mask_path)
    common = [str(args.exe), "--config", str(config), "image", "generate", "--model", "h3_image",
              "--transformer", str(args.transformer), "--text-encoder", str(args.text_encoder),
              "--vae", str(args.vae), "--seed", "42"]
    tools = Path(__file__).resolve().parent
    capi = ["--library", str(args.library), "--model", "h3_image", "--transformer", str(args.transformer),
            "--text-encoder", str(args.text_encoder), "--vae", str(args.vae)]
    cases = []
    for backend in ("cuda", "vulkan"):
        backend_flags = ["--inference-backend", backend]
        if backend == "vulkan":
            backend_flags += ["--vulkan-arithmetic", "portable"]
        for kind in ("text", "edit", "multi", "rectangle", "mask", "outpaint"):
            name = f"{backend}-{kind}"
            output = args.output / f"{name}.png"
            command = common + backend_flags + ["--out", str(output)]
            if kind == "outpaint":
                command += ["--edit-image", str(source_path), "--outpaint", "128,128,128,128",
                            "--prompt", "Extend the scene in <Picture 1>. Continue the table and background, matching perspective and lighting."]
            else:
                command += ["--resolution", "512x512"]
                if kind == "text":
                    command += ["--prompt", "A studio photograph of a red ceramic teapot on a wooden table."]
                else:
                    command += ["--edit-image", str(source_path), "--prompt",
                                "Edit <Picture 1>. Change the red ceramic teapot to cobalt blue. Keep its shape, table, background and lighting."]
                    if kind == "multi":
                        command += ["--reference-image", str(reference_path)]
                        command[command.index("--prompt") + 1] = "Edit <Picture 1>. Give the teapot the blue color from <Picture 2>. Keep its shape and surroundings."
                    elif kind == "rectangle":
                        command += ["--edit-box", "96,64,320,384", "--edit-feather", "8"]
                    elif kind == "mask":
                        command += ["--edit-mask", str(mask_path)]
            cases.append((name, command, kind, output))
        cases.append((f"{backend}-capi-mask", [sys.executable, str(tools / "image_edit_mask_smoke.py"),
                      *capi, "--backend", backend], None, None))
        cases.append((f"{backend}-capi-outpaint", [sys.executable, str(tools / "outpaint_generation_smoke.py"),
                      *capi, "--backend", backend, "--source", str(source_path),
                      "--output", str(args.output / f"{backend}-capi-outpaint")], None, None))
    names = {case[0] for case in cases}
    if args.case and not set(args.case) <= names:
        parser.error("unknown case: " + str(set(args.case) - names))
    results = []

    def report():
        (args.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Artifacts: {args.output}", flush=True)
    if args.start_at:
        print(f"Waiting until {args.start_at.isoformat()}", flush=True)
        while datetime.now(args.start_at.tzinfo) < args.start_at:
            time.sleep(min(10, max(.1, (args.start_at - datetime.now(args.start_at.tzinfo)).total_seconds())))
    for name, command, kind, output in cases:
        if args.case and name not in args.case:
            continue
        started_wait = time.monotonic()
        quiet = 0
        while quiet < 3:
            memory = subprocess.check_output(["nvidia-smi", f"--id={args.gpu_index}",
                "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            quiet = quiet + 1 if int(memory.strip()) < args.max_used_mib else 0
            if time.monotonic() - started_wait > args.wait_seconds:
                raise RuntimeError(f"GPU remained busy before {name}; results retained in {args.output}")
            if quiet < 3:
                time.sleep(5)
        print(f"START {name} {datetime.now().astimezone().isoformat()}", flush=True)
        started = time.monotonic()
        result = {"case": name, "command": command, "ok": False}
        try:
            with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, env=env, cwd=args.output, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=1800)
            result["returncode"] = completed.returncode
            if completed.returncode:
                raise RuntimeError(f"process returned {completed.returncode}; see {name}.log")
            if output:
                actual = np.array(Image.open(output).convert("RGB"))
                expected_shape = (768, 768, 3) if kind == "outpaint" else rgb.shape
                assert actual.shape == expected_shape, (actual.shape, expected_shape)
                assert actual.std() > 1, "degenerate image"
                if kind in ("rectangle", "mask"):
                    editable = coverage != 0 if kind == "mask" else ((xx >= 96) & (xx < 416) & (yy >= 64) & (yy < 448))
                    assert np.array_equal(actual[~editable], rgb[~editable]), "source pixels changed outside mask"
                    assert np.any(actual[editable] != rgb[editable]), "selected pixels did not change"
                if kind == "outpaint":
                    assert np.array_equal(actual[136:632, 136:632], rgb[8:504, 8:504]), "source interior changed"
            result["ok"] = True
        except Exception as error:
            result["error"] = str(error)
        result["seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
        report()
        print(f"{'PASS' if result['ok'] else 'FAIL'} {name}: {result.get('error', '')} ({result['seconds']} s)", flush=True)
    if not all(result["ok"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
