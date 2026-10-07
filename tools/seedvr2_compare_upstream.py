"""Compare native captures with the pinned ByteDance architecture, using SDPA.

Requires the development environment (torch, diffusers==0.35.1, einops,
omegaconf, rotary-embedding-torch, torchvision, safetensors) and an upstream
checkout at e4de8c24441a67e1b7df56abea10645059bb1185. No reference code is changed.
The FlashAttention dependency is replaced with FP32 PyTorch SDPA; RMSNorm uses
the Apex single-rounding formula. Captured noise/conditioning isolates arithmetic
from the different native/PyTorch RNGs. Outputs report relative L2 and correlation.
With --temporal-output, also decode noise minus the independent prediction and
save native/reference pre-color RGB as raw THWC float32, with shape and temporal
metrics in upstream-comparison.json. Use an untiled capture for this comparison;
the output preserves decoder overshoot and spatial padding without clamping.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import types
from importlib.machinery import ModuleSpec
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument('--upstream', required=True)
ap.add_argument('--capture', required=True)
ap.add_argument('--transformer', required=True)
ap.add_argument('--vae', required=True)
ap.add_argument('--temporal-output', action='store_true',
                help='Decode the full reference prediction and save pre-color THWC RGB float32 outputs')
args = ap.parse_args()
sys.path.insert(0, args.upstream)
root = Path(args.capture)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False


def flash(q, k, v, cu_seqlens_q, cu_seqlens_k, **kwargs):
    out = []
    for a, b, c, d in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:], cu_seqlens_k[:-1], cu_seqlens_k[1:]):
        qi, ki, vi = [x[int(lo):int(hi)].transpose(0, 1).unsqueeze(0)
                      for x, lo, hi in [(q, a, b), (k, c, d), (v, c, d)]]
        # FP32 authority avoids BF16 softmax-probability rounding differences
        # between FlashAttention, cuDNN and slopfab's FP16-probability kernel.
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out.append(F.scaled_dot_product_attention(qi.float(), ki.float(), vi.float()).to(q.dtype).squeeze(0).transpose(0, 1))
    return torch.cat(out)


stub = types.ModuleType('flash_attn')
stub.__spec__ = ModuleSpec('flash_attn', loader=None)
stub.flash_attn_varlen_func = flash
sys.modules['flash_attn'] = stub
from models.dit_v2.nadit import NaDiT
from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKL

# Match Apex FusedRMSNorm's single final rounding of the affine result. The
# diffusers unfused RMSNorm rounds before multiplying by its learned weight.
class FusedRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


apex = types.ModuleType('apex.normalization')
apex.FusedRMSNorm = FusedRMSNorm
sys.modules['apex.normalization'] = apex

report = {}


def captured(name):
    return load_file(str(root / (name + '.safetensors')))[name]


def compare(name, value):
    native = captured(name).flatten().float()
    reference = value.detach().cpu().flatten().float()
    assert native.shape == reference.shape, (name, native.shape, reference.shape)
    rel = float(torch.linalg.vector_norm(native - reference) / torch.linalg.vector_norm(reference))
    corr = float(torch.corrcoef(torch.stack([reference, native]))[0, 1])
    report[name] = dict(relative_l2=rel, correlation=corr, max_abs=float((native-reference).abs().max()))
    print(name, report[name], flush=True)


def load_vae():
    vae = VideoAutoencoderKL(in_channels=3, out_channels=3,
        down_block_types=('DownEncoderBlock3D',)*4, up_block_types=('UpDecoderBlock3D',)*4,
        block_out_channels=(128,256,512,512), layers_per_block=2, latent_channels=16,
        norm_num_groups=32, temporal_scale_num=2, inflation_mode='pad',
        use_quant_conv=False, use_post_quant_conv=False)
    vae.load_state_dict(load_file(args.vae), strict=True)
    return vae.eval().to(device='cuda', dtype=torch.bfloat16)


vae = load_vae()
x = captured('vae_input').permute(3,0,1,2).unsqueeze(0).cuda().bfloat16()
y = vae.encoder(x)
compare('vae_moments', y.squeeze(0).permute(1,2,3,0))
x = captured('vae_latent').permute(3,0,1,2).unsqueeze(0).cuda().bfloat16()
y = vae.decoder(x)
compare('vae_decoded', y.squeeze(0).permute(1,2,3,0))
del vae, x, y
gc.collect(); torch.cuda.empty_cache()

with torch.device('meta'):
    dit = NaDiT(vid_in_channels=33, vid_out_channels=16, vid_dim=2560,
        txt_in_dim=5120, txt_dim=2560, emb_dim=15360, heads=20, head_dim=128,
        expand_ratio=4, norm='fusedrms', norm_eps=1e-5, ada='single', qk_bias=False,
        qk_norm='fusedrms', patch_size=(1,2,2), num_layers=32, block_type='mmdit_sr',
        mm_layers=10, mlp_type='swiglu', window=[(4,3,3)]*32,
        window_method=['720pwin_by_size_bysize','720pswin_by_size_bysize']*16,
        rope_type='mmrope3d', rope_dim=128, vid_out_norm='fusedrms')
sd = load_file(args.transformer)
text = sd.pop('positive_conditioning').cuda().bfloat16()
sd.pop('negative_conditioning')
dit.load_state_dict(sd, strict=True, assign=True)
for module in dit.modules():
    for name, buffer in module._buffers.items():
        if buffer is not None and buffer.is_meta:
            module._buffers[name] = torch.zeros(buffer.shape, dtype=buffer.dtype)
dit = dit.eval().to(device='cuda', dtype=torch.bfloat16)
# Rotary position arithmetic keeps the checkpoint's frequency values in FP32.
for i, block in enumerate(dit.blocks):
    block.attn.rope.rope.freqs = sd[f'blocks.{i}.attn.rope.rope.freqs'].float().cuda()
    # The reference materializes an 8 GiB 1024x128x128 position table per block.
    # Trim unused language-position rows; values at every used index are identical.
    geometry = captured('dit_patches').shape
    limits = (len(text) + geometry[0], geometry[1], geometry[2])
    rotary = block.attn.rope.rope
    block.attn.rope.get_axial_freqs = lambda *dims, rotary=rotary, limits=limits: rotary.get_axial_freqs(
        *(min(d, limit) for d, limit in zip(dims, limits)))
del sd
dit.emb_in.register_forward_hook(lambda m, a, y: compare('dit_embedding', y))
for i in [0, 31]:
    dit.blocks[i].register_forward_hook(lambda m,a,y,i=i: compare(f'dit_block_{i}',y[0]))
patches = captured('dit_patches')
t, h, w, _ = patches.shape
video = patches.reshape(t,h,w,2,2,33).permute(0,1,3,2,4,5).reshape(-1,33).cuda().bfloat16()
result = dit(video, text, torch.tensor([[t,h*2,w*2]],device='cuda'),
             torch.tensor([[len(text)]],device='cuda'), 1000.0).vid_sample
prediction = result.reshape(t,h,2,w,2,16).permute(0,1,3,2,4,5).reshape(t,h,w,64)
compare('dit_prediction', prediction)
if args.temporal_output:
    # Keep the original FP32 diffusion noise: native subtraction happens before
    # the VAE's BF16 upload, while only the DiT input is rounded to BF16.
    noise = patches.reshape(t,h,w,2,2,33).permute(0,1,3,2,4,5).reshape(t,h*2,w*2,33)[..., :16]
    reference_latent = (noise - result.detach().cpu().float().reshape(t,h*2,w*2,16)) / 0.9152
# Isolate every block from accumulated BF16 perturbations. This is stricter
# than only comparing images, where output projection suppresses hidden drift.
from common.cache import Cache
emb = captured('dit_embedding').reshape(1, -1).cuda().bfloat16()
vid_shape = torch.tensor([[t,h,w]], device='cuda')
txt_shape = torch.tensor([[len(text)]], device='cuda')
block_errors = []
for i, block in enumerate(dit.blocks):
    block._forward_hooks.clear()
    vi = captured('dit_video_in' if i == 0 else f'dit_block_{i-1}').reshape(-1,2560).cuda().bfloat16()
    ti = captured('dit_text_in' if i == 0 else f'dit_text_{i-1}').reshape(-1,2560).cuda().bfloat16()
    expected = block(vi, ti, vid_shape, txt_shape, emb, Cache())[0].float().cpu().flatten()
    native = captured(f'dit_block_{i}').flatten()
    err = float(torch.linalg.vector_norm(native-expected)/torch.linalg.vector_norm(expected))
    block_errors.append(err)
report['isolated_blocks'] = {'relative_l2': block_errors, 'maximum': max(block_errors)}
print('isolated block maximum relative L2:', max(block_errors), flush=True)
if args.temporal_output:
    # Free the DiT before reloading the VAE to keep this useful on 12 GB cards.
    del dit, block, text, video, result, prediction, emb, vi, ti
    gc.collect(); torch.cuda.empty_cache()
    compare('vae_latent', reference_latent)
    vae = load_vae()
    decoded = vae.decoder(reference_latent.permute(3,0,1,2).unsqueeze(0).cuda().bfloat16())
    reference_rgb = decoded.squeeze(0).permute(1,2,3,0).float().cpu() * 0.5 + 0.5
    native_rgb = captured('vae_decoded').float() * 0.5 + 0.5
    assert native_rgb.shape == reference_rgb.shape

    def temporal_metrics(rgb):
        adjacent = (rgb[1:] - rgb[:-1]).square().mean((1,2,3)).sqrt()
        return dict(adjacent_rmse=adjacent.tolist(),
                    mean_adjacent_rmse=float(adjacent.mean()) if len(adjacent) else None,
                    max_adjacent_rmse=float(adjacent.max()) if len(adjacent) else None,
                    temporal_std_rms=float((rgb-rgb.mean(0)).square().mean().sqrt()))

    difference = native_rgb - reference_rgb
    report['temporal_output'] = dict(
        shape=list(native_rgb.shape), layout='THWC', dtype='float32',
        range='decoder * 0.5 + 0.5, unclipped, before color matching',
        native=temporal_metrics(native_rgb), reference=temporal_metrics(reference_rgb),
        native_reference_frame_rmse=difference.square().mean((1,2,3)).sqrt().tolist(),
        native_reference_rmse=float(difference.square().mean().sqrt()),
        adjacent_difference_rmse=float((difference[1:]-difference[:-1]).square().mean().sqrt())
            if len(difference) > 1 else None)
    for name, rgb in [('native', native_rgb), ('reference', reference_rgb)]:
        rgb.contiguous().numpy().tofile(root / (name + '-precolor-rgb.f32'))
    print('temporal output:', report['temporal_output'], flush=True)
(root/'upstream-comparison.json').write_text(json.dumps(report, indent=2))
if max(block_errors) > 0.015:
    raise SystemExit('An isolated block exceeded relative L2 0.015')
if any(report[k]['relative_l2'] > 0.1 or report[k]['correlation'] < 0.99
       for k in ['vae_moments','vae_decoded','dit_embedding','dit_prediction']):
    raise SystemExit('An output exceeded relative L2 0.1 / correlation 0.99 limits')
