"""Export standard HTDemucs safetensors and compare native inference to upstream.

Developer-only dependencies: torch, demucs==4.0.1, safetensors, numpy.
Export: python tools/demucs_reference.py export --out weights/htdemucs.safetensors
Verify: python tools/demucs_reference.py verify --exe build/Release/slopfab_demucs_tests.exe
Use --real for numerical verification with the released pretrained checkpoint.
Only export/--real downloads and deserializes the official upstream checkpoint.
The application itself reads safetensors directly and does not use Python.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "verify"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--exe", type=Path)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file, load_file
    from demucs.htdemucs import HTDemucs
    from demucs.apply import apply_model
    torch.set_num_threads(args.threads)
    torch.manual_seed(423)
    if args.action == "export" or args.real:
        from demucs.pretrained import get_model
        model = get_model("htdemucs").models[0].cpu().float().eval()
    else:
        model = HTDemucs(["drums", "bass", "other", "vocals"], channels=8,
                         bottom_channels=32, t_layers=2, dconv_mode=3,
                         segment=0.08).eval()
    state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    metadata = {"architecture": "htdemucs", "sources": "drums,bass,other,vocals",
                "samplerate": "44100", "upstream": "facebookresearch/demucs 4.0.1"}
    if args.action == "export":
        if args.out is None:
            parser.error("export requires --out")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_file(state, str(args.out), metadata=metadata)
        print(args.out)
        return
    if args.exe is None:
        parser.error("verify requires --exe")
    # Include non-multiple lengths, stereo asymmetry, DC and a partial tail.
    segment = 0.08
    model.segment = segment
    length = int(segment * 44100)
    wave = torch.randn(1, 2, length) * 0.1 + 0.03
    mix = torch.randn(2, length + 1719) * 0.15 + 0.02
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std()
    with torch.no_grad():
        spec = model._magnitude(model._spec(wave))[0]
        inverse = model._ispec(model._spec(wave), length)[0]
        output = model(wave)[0].reshape(4, 2, length)
        separated = apply_model(model, ((mix - mean) / std)[None], shifts=0,
                                split=True, overlap=0.25)[0] * std + mean
        separated = separated.transpose(1, 2).contiguous()
    with tempfile.TemporaryDirectory(prefix="slopfab-demucs-reference-") as directory:
        root = Path(directory)
        save_file(state, str(root / "model.safetensors"), metadata=metadata)
        save_file({"pcm": wave[0], "mixed": mix.t().contiguous().flatten()},
                  str(root / "input.safetensors"))
        subprocess.run([str(args.exe.resolve()), "probe", str(root / "model.safetensors"),
                        str(root / "input.safetensors"), str(root / "output.safetensors"),
                        str(segment)], check=True,
                       env={**os.environ, "OMP_NUM_THREADS": str(args.threads)})
        actual = load_file(str(root / "output.safetensors"))
        for key, reference in {"spec": spec, "inverse": inverse, "output": output,
                               "separated": separated}.items():
            error = (actual[key] - reference).abs().max().item()
            print(f"{key}: maximum absolute error {error:.8g}")
            torch.testing.assert_close(actual[key], reference, atol=3e-4, rtol=3e-3)


if __name__ == "__main__":
    main()
