"""Generate independent PyTorch operator fixtures for the native SeedVR2 port.

Development only. Requires torch and safetensors; CPU is sufficient.
  python tools/seedvr2_goldens.py build/seedvr2-goldens.safetensors
  set SLOPFAB_SEEDVR2_FIXTURE=...absolute path...
  ctest --test-dir build -C Release -R seedvr2_cuda --output-on-failure
"""
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

torch.manual_seed(20261004)
out = {}


def rand(shape, scale=0.1):
    return (torch.randn(shape) * scale).bfloat16()


def save(name, x):
    out[name] = x.contiguous().float()


def thwc(x):
    return x.squeeze(0).permute(1, 2, 3, 0).contiguous()


x = rand((1, 32, 5, 8, 10))
save('input', thwc(x))
for name, kt, down, stride in [('conv', 3, False, 1), ('down', 3, True, 2), ('spatial', 1, True, 1)]:
    w = rand((64, 32, kt, 3, 3), 0.02)
    b = rand((64,), 0.01)
    save(name + '.weight', w)
    save(name + '.bias', b)
    padded = torch.cat([x[:, :, :1].repeat(1, 1, kt - 1, 1, 1), x], dim=2)
    padded = F.pad(padded, (0, 1, 0, 1) if down else (1, 1, 1, 1))
    # Native GEMM rounds its result before adding bias, like the DiT's linear ops.
    y = F.conv3d(padded.float(), w.float(), stride=(stride, 2 if down else 1, 2 if down else 1))
    y = (y.bfloat16() + b.view(1, -1, 1, 1, 1)).bfloat16()
    save(name + '.expected', thwc(y))

w, b = rand((32,), 0.1) + 1, rand((32,), 0.01)
save('norm.weight', w); save('norm.bias', b)
y = x.permute(0, 2, 1, 3, 4).reshape(5, 32, 8, 10)
y = F.group_norm(y.float(), 32, w.float(), b.float(), 1e-6).bfloat16()
y = F.silu(y).reshape(1, 5, 32, 8, 10).permute(0, 2, 1, 3, 4)
save('norm.expected', thwc(y))

q = rand((1, 1, 7, 256))
w = rand((256,), 0.1) + 1
save('rms.input', q); save('rms.weight', w)
y = q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + 1e-5) * w.float()
save('rms.expected', y.bfloat16())
lw, lb = rand((128,256)), rand((128,), 2)
save('linear.weight', lw); save('linear.bias', lb)
save('linear.expected', F.linear(q.float(), lw.float(), lb.float()).bfloat16())
emb = rand((1, 1, 1, 1536))
shift, scale, gate = rand((256,)), rand((256,)) + 1, rand((256,))
save('emb', emb)
save('ada_shift', shift); save('ada_scale', scale); save('ada_gate', gate)
mod = emb.reshape(256, 2, 3)[:, 1]
save('mod.expected', (q * (mod[:, 1] + scale) + (mod[:, 0] + shift)).bfloat16())
save('gate.expected', (q * (mod[:, 2] + gate)).bfloat16())

u = torch.arange(3 * 2 * 3 * 32, dtype=torch.float32).reshape(3, 2, 3, 32).bfloat16()
save('up.input', u)
up = u.reshape(3, 2, 3, 2, 2, 2, 4).permute(0, 5, 1, 3, 2, 4, 6).reshape(6, 4, 6, 4)
save('up.expected', torch.cat([up[:1], up[2:]], dim=0))

save_file(out, sys.argv[1])
print(f'Wrote {len(out)} tensors to {sys.argv[1]}')
