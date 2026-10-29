"""Create parity fixtures using the upstream TAEHV implementation (test only).

Usage: python tools/taeh3_reference.py upstream/taehv.py taeh3.safetensors output.safetensors
Requires PyTorch and safetensors. The application does not use Python.
Reference: madebyollin/taehv at 62f7591f59dfbb4c3c02b7a621d180a9eeaba26c.
"""
import importlib.util
import sys
import torch
from safetensors.torch import load_file, save_file

spec = importlib.util.spec_from_file_location("taehv", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = module.TAEHV(None, patch_size=2, latent_channels=24,
                    decoder_time_upscale=(False, False, False),
                    decoder_space_upscale=(False, False, False))
model.is_h3 = True
model.load_state_dict(model.patch_tgrow_layers(load_file(sys.argv[2])))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
frames, height, width = map(int, sys.argv[4:7]) if len(sys.argv) == 7 else (7, 3, 4)
x = torch.sin(torch.arange(24 * frames * height * width, dtype=torch.float32) * .013)
x = x.reshape(24, frames, height, width)
with torch.inference_mode():
    y = model.decode_video(x.permute(1, 0, 2, 3)[None].to(device), parallel=False, show_progress_bar=False)[0]
rgb = (y.permute(0, 2, 3, 1).clamp(0, 1) * 255 + .5).floor().cpu().contiguous()
save_file({"latents": x, "rgb": rgb}, sys.argv[3])
print(tuple(y.shape))
