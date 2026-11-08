"""Run the upstream Sol-H3-VAE reference; optionally compare native CLI output.

Requires torch, safetensors and diffusers with MiniMax-H3 support. --reference-code
is the downloaded HF repository's code/sol_h3_vae folder, revision
6a8a33fc61e092281f32bf28cde236760e2caaa4. No Python dependency is used by the DLL.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-code", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cli")
    parser.add_argument("--backend", choices=["cuda", "vulkan"], default="cuda")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cases = [(1, 2, 3), (5, 4, 6), (6, 17, 19), (11, 3, 5)]
    if not args.compare_only:
        package = types.ModuleType("sol_h3_vae")
        package.__path__ = [str(Path(args.reference_code).resolve())]
        sys.modules["sol_h3_vae"] = package
        from sol_h3_vae.decoder import SolH3VAEDecoder
        decoder = SolH3VAEDecoder(args.config, args.weights, engine="ref")
        for t, h, w in cases:
            torch.manual_seed(1234 + t)
            z = torch.randn(1, t, 24, h, w, device="cuda") * .4
            rgb = decoder.processor.revert_tensor(decoder.decode(z))
            name = f"t{t}-h{h}-w{w}"
            save_file({"latent": z[0].permute(1, 0, 2, 3).contiguous().cpu()}, output / f"{name}-latent.safetensors")
            save_file({"pixels": rgb[0].contiguous().cpu()}, output / f"{name}-ref.safetensors")
            print(name, tuple(rgb.shape), "reference saved", flush=True)
        del decoder, z, rgb
        torch.cuda.empty_cache()
    if not args.cli:
        return
    results = []
    for t, h, w in cases:
        name = f"t{t}-h{h}-w{w}"
        native = output / f"{name}-native-{args.backend}.safetensors"
        subprocess.run([args.cli, 'video', 'decode', "--vae", args.weights, "--inference-backend", args.backend,
                        "--vulkan-arithmetic", "portable",
                        "--latent", str(output / f"{name}-latent.safetensors"),
                        "--dump", str(native), "--out", str(output / f"{name}.y4m")], check=True)
        ref = load_file(output / f"{name}-ref.safetensors")["pixels"].numpy()
        got = load_file(native)["pixels"].numpy()
        assert got.shape == ref.shape
        assert np.isfinite(got).all()
        error = got.astype(np.float64) - ref.astype(np.float64)
        mse = np.mean(error ** 2)
        psnr = float(-10 * np.log10(max(mse, 1e-30)))
        results.append(dict(case=name, psnr=psnr, max_error=float(np.abs(error).max())))
        print(results[-1], flush=True)
        assert psnr >= 40, f"reference mismatch: {results[-1]}"
    (output / f"comparison-{args.backend}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
