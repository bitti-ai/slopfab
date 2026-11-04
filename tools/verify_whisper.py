"""Compare native Whisper features/encoder/logits against a local HF checkpoint.

Development only: requires PyTorch, Transformers 4.57.6, safetensors and NumPy.
No runtime Python dependency. Supply a downloaded model folder and a mono 16 kHz
PCM16 WAV; this script never downloads weights. Also checks cached decoding.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import wave

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", required=True)
p.add_argument("--wav", required=True)
p.add_argument("--probe", required=True)
p.add_argument("--backend", choices=["cuda", "vulkan", "cpu"], default="cuda")
args = p.parse_args()
torch.set_num_threads(8)
with wave.open(args.wav) as f:
    assert f.getframerate() == 16000 and f.getnchannels() == 1 and f.getsampwidth() == 2
    pcm = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32) / 32768
feature = WhisperFeatureExtractor.from_pretrained(args.model, local_files_only=True)
mel = feature._torch_extract_fbank_features(np.pad(pcm, (0, 480000))[None, :], "cpu")[0].T.copy()
model = WhisperForConditionalGeneration.from_pretrained(
    args.model, local_files_only=True, attn_implementation="eager").float().eval()
config = model.generation_config
tokens = [config.decoder_start_token_id]
if config.is_multilingual:
    tokens += [config.lang_to_id["<|en|>"], config.task_to_id["transcribe"]]
ids = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    encoder = model.model.encoder(torch.from_numpy(mel[:3000].T.copy())[None, :]).last_hidden_state
    logits = model(encoder_outputs=(encoder,), decoder_input_ids=ids).logits[0, -1]
    cached = model(encoder_outputs=(encoder,), decoder_input_ids=torch.cat([ids, torch.tensor([[123]])], dim=1)).logits[0, -1]
expected = {"mel": torch.from_numpy(mel), "encoder": encoder[0], "logits": logits, "cached_logits": cached}
with tempfile.TemporaryDirectory(prefix="whisper-parity-") as temp:
    root = Path(temp)
    save_file({"pcm": torch.from_numpy(pcm), "tokens": ids[0].float()}, str(root / "input.safetensors"))
    subprocess.run([str(Path(args.probe).resolve()), "probe", args.backend,
                    str(Path(args.model).resolve() / "model.safetensors"),
                    str(root / "input.safetensors"), str(root / "output.safetensors")], check=True)
    actual = load_file(str(root / "output.safetensors"))
    for name, reference in expected.items():
        difference = (actual[name] - reference).abs()
        maximum, mean = difference.max().item(), difference.mean().item()
        print(json.dumps({"tensor": name, "max_error": maximum, "mean_error": mean}))
        # Backend reductions and erf implementations differ, but errors must stay
        # small relative to real logits. The frontend is checked more tightly.
        assert maximum < (0.002 if name == "mel" else 0.03), (name, maximum)
        if name.endswith("logits"):
            assert actual[name].argmax() == reference.argmax(), name
print("PASS: reference features, encoder, logits and KV cache")
