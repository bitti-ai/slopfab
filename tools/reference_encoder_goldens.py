"""Independent CPU PyTorch authority for the new reference encoder graphs.

Formulas follow Diffusers c419dac0152186060246c93a095bc1bfaea342b3
autoencoder_kl_minimax_h3{,_audio}.py. Uses the original checkpoint names.
Run with video-checkpoint audio-checkpoint output.safetensors [audio-samples].
"""
import sys
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(8)


def weights(path, prefixes):
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k).float() for k in f.keys() if k.startswith(prefixes)}


@torch.no_grad()
def video(x, w):
    def conv(x, name, stride=(1, 1, 1), down=False):
        weight = w[name + ".weight"]
        k = weight.shape[-1]
        if k == 3:
            x = F.pad(x, (0, 1, 0, 1, 0, 0) if down else (1, 1, 1, 1, 0, 0), mode="reflect")
            x = F.pad(x, (0, 0, 0, 0, 2, 0))
        return F.conv3d(x, weight, w[name + ".bias"], stride=stride)

    def norm(x, name):
        b, c, t, h, width = x.shape
        flat = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, width)
        flat = F.silu(F.group_norm(flat, 32, w[name + ".weight"], w[name + ".bias"], 1e-6))
        return flat.reshape(b, t, c, h, width).permute(0, 2, 1, 3, 4)

    x = conv(x, "encoder.conv_in")
    channels = [128, 256, 256, 512, 512, 1024]
    for level, cout in enumerate(channels):
        for block in range(2):
            p = f"encoder.down.{level}.block.{block}"
            residual = x if x.shape[1] == cout else conv(x, p + ".nin_shortcut")
            y = conv(norm(x, p + ".norm1"), p + ".conv1")
            x = residual + conv(norm(y, p + ".norm2"), p + ".conv2")
        if level < 4:
            x = conv(x, f"encoder.down.{level}.downsample.conv", (2 if level in (1, 2) else 1, 2, 2), True)
    return conv(conv(norm(x, "encoder.norm_out"), "encoder.conv_out"), "quant_conv")


@torch.no_grad()
def audio(x, w):
    def conv(x, n, stride=1, padding=0, dilation=1):
        return F.conv1d(x, w[n + ".weight"], w[n + ".bias"], stride, padding, dilation)

    def snake(x, n):
        a = w[n + ".alpha"]
        return x + torch.sin(a * x).square() / (a + 1e-9)

    def norm(x, n):
        return F.layer_norm(x, (x.shape[-1],), w[n + ".weight"], w[n + ".bias"], 1e-5)

    def linear(x, n):
        return F.linear(x, w[n + ".weight"], w[n + ".bias"])

    x = F.pad(x, (0, (-x.shape[-1]) % 800))
    x = conv(x, "encoder.block.0", padding=3)
    for stage, stride in enumerate([2, 4, 4, 5, 5], 1):
        p = f"encoder.block.{stage}.block."
        for block, dilation in enumerate([1, 3, 9]):
            n = p + f"{block}.block."
            y = conv(snake(x, n + "0"), n + "1", padding=3 * dilation, dilation=dilation)
            x = x + conv(snake(y, n + "2"), n + "3")
        x = conv(snake(x, p + "3"), p + "4", stride=stride, padding=math.ceil(stride / 2))
    x = conv(snake(x, "encoder.block.6"), "encoder.block.7", padding=1).transpose(1, 2)
    bias = torch.cat([w["pre_block.attn." + k] for k in ("q_bias", "zero_k_bias", "v_bias")])
    qkv = F.linear(norm(x, "pre_block.norm1"), w["pre_block.attn.qkv.weight"], bias)
    b, t, _ = qkv.shape
    q, k, v = qkv.reshape(b, t, 3, 8, 256).permute(2, 0, 3, 1, 4).unbind()
    # Explicit math attention, including head averaging then adaptive pooling.
    scores = (q @ k.transpose(-1, -2)) / 16
    scores.masked_fill_(torch.ones(t, t, dtype=torch.bool).triu(1), -float("inf"))
    attn = F.adaptive_avg_pool1d((scores.softmax(-1) @ v).mean(1), 32)
    x = linear(norm(x, "pre_block.norm3"), "pre_block.proj") + linear(attn, "pre_block.attn.proj")
    y = norm(norm(x, "pre_block.norm2"), "pre_block.mlp.norm")
    y = F.gelu(linear(y, "pre_block.mlp.w0"), approximate="tanh") * linear(y, "pre_block.mlp.w1")
    x = x + linear(y, "pre_block.mlp.w2")
    return conv(x.transpose(1, 2), "mean_proj")


if __name__ == "__main__":
    torch.manual_seed(123)
    vi = torch.randn(1, 3, 17, 32, 32)
    ai = torch.randn(2, 1, int(sys.argv[4]) if len(sys.argv) > 4 else 1601) * .05
    print("Computing temporal video authority", flush=True)
    vw = weights(sys.argv[1], ("encoder.", "quant_conv."))
    vo = video(vi, vw)
    # Two independent 17-frame chunks, with the last input repeated and only
    # the final three latent frames removed after concatenation.
    rgb = torch.randint(0, 256, (22, 32, 32, 3), dtype=torch.uint8)
    pixels = rgb.float().permute(3, 0, 1, 2)[None] / 255
    pixels = (pixels - torch.tensor([.485, .456, .406])[None, :, None, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None, None]
    tail = torch.cat([pixels[:, :, 17:], pixels[:, :, -1:].expand(-1, -1, 12, -1, -1)], 2)
    moments = torch.cat([video(pixels[:, :, :17], vw), video(tail, vw)], 2)[:, :, :-3]
    mean, logvar = moments.chunk(2, 1)
    torch.manual_seed(42)
    latent = (mean + (.5 * logvar.clamp(-30, 20)).exp() * torch.randn(mean.shape)).half().float()
    lm, ls = torch.arange(24).float() * .01, torch.arange(24).float() * .01 + .8
    latent = (latent - lm[None, :, None, None, None]) / ls[None, :, None, None, None]
    rows = latent.reshape(1, 24, 7, 1, 2, 1, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(7, 96)
    print("Computing audio authority", flush=True)
    ao = audio(ai, weights(sys.argv[2], ("encoder.", "pre_block.", "mean_proj.")))
    # Torchaudio's default sinc_interp_hann kernel expressed with independent
    # polyphase Conv1d (the C++ implementation evaluates each output sample).
    ri = (torch.sin(torch.arange(88200, dtype=torch.float64) * (2 * math.pi * 440 / 44100)) * .2).float()
    orig, new = 441, 320
    base = min(orig, new) * .99
    width = math.ceil(6 * orig / base)
    index = torch.arange(-width, width + orig, dtype=torch.float64)[None, None] / orig
    t = (index - torch.arange(new, dtype=torch.float64)[:, None, None] / new) * base
    t = t.clamp(-6, 6)
    kernel = (torch.sinc(t) * torch.cos(t * math.pi / 12).square() * base / orig).float()
    ro = F.conv1d(F.pad(ri[None, None], (width, width + orig)), kernel, stride=orig)
    ro = ro.transpose(1, 2).reshape(-1)[:64000]
    save_file({"video_input": vi.contiguous(), "video_moments": vo.contiguous(),
               "audio_input": ai.contiguous(), "audio_mean": ao.contiguous(),
               "resample_input": ri, "resample_output": ro.repeat(2),
               "video_rgb": rgb, "video_rows": rows.contiguous(),
               "video_latent_mean": lm, "video_latent_std": ls}, sys.argv[3])
    print("Saved", sys.argv[3], flush=True)
