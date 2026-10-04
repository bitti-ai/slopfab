"""Generate numerical fixtures from the supplied x4plus safetensors.

Requires torch and safetensors only for development, never for slopfab runtime.
Architecture: https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
Padding/tiling: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
Usage: python tools/realesrgan_reference.py MODEL.safetensors OUTPUT.safetensors
"""
import argparse

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    weights = load_file(args.model)

    def conv(x, name, activate=False):
        x = F.conv2d(x, weights[name + ".weight"], weights[name + ".bias"], padding=1)
        return F.leaky_relu(x, 0.2) if activate else x

    def network(x):
        first = conv(x, "conv_first")
        x = first
        for block in range(23):
            skip = x
            for rdb in range(1, 4):
                prefix = f"body.{block}.rdb{rdb}.conv"
                features = [x]
                for layer in range(1, 5):
                    features.append(conv(torch.cat(features, 1), prefix + str(layer), True))
                x = x + 0.2 * conv(torch.cat(features, 1), prefix + "5")
            x = skip + 0.2 * x
        x = first + conv(x, "conv_body")
        for layer in ("conv_up1", "conv_up2"):
            x = conv(F.interpolate(x, scale_factor=2, mode="nearest"), layer, True)
        return conv(conv(x, "conv_hr", True), "conv_last")

    torch.manual_seed(123)
    images = torch.rand(2, 3, 11, 13)
    padded = F.pad(images, (0, 3, 0, 3), mode="reflect")
    with torch.inference_mode():
        full = network(padded)[:, :, :44, :52].clamp(0, 1)
        tiled = torch.empty(2, 3, 56, 64)
        for y in range(0, 14, 7):
            for x in range(0, 16, 7):
                x0, y0 = max(0, x - 3), max(0, y - 3)
                xe, ye = min(16, x + 7), min(14, y + 7)
                patch = network(padded[:, :, y0:min(14, ye + 3), x0:min(16, xe + 3)])
                tiled[:, :, y*4:ye*4, x*4:xe*4] = patch[:, :, (y-y0)*4:(ye-y0)*4, (x-x0)*4:(xe-x0)*4]
        save_file({"input": images.permute(1, 0, 2, 3).contiguous(),
                   "full": full.permute(1, 0, 2, 3).contiguous(),
                   "tiled": tiled[:, :, :44, :52].clamp(0, 1).permute(1, 0, 2, 3).contiguous()}, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
