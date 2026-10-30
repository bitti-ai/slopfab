"""Prepare exact Hyperflow two-time modulation for a pruned H3 checkpoint.

Requires torch, numpy and safetensors. Reads a local Diffusers transformer
directory (--source), or downloads only the necessary tensor byte ranges from
the official model at the adapter's pinned revision. Inference is offline.
The base and adapter are never modified. The companion is strength-specific.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import urllib.request

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


def remote_bytes(url, start=None, end=None):
    headers = {}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
        # Keep CDN caches from reusing a different range of the same shard.
        url += f'?slopfab_range={start}-{end}'
    token = os.environ.get('HF_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        if start is not None and (r.status != 206 or
                not r.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/')):
            raise ValueError('Server did not honor the tensor byte range')
        data = r.read()
    if start is not None and len(data) != end - start + 1:
        raise ValueError('Incomplete tensor download')
    return data


class Source:
    def __init__(self, source, revision, variant):
        self.local = Path(source) if source else None
        folder = 'transformer_ref' if variant == 'ref2va' else 'transformer'
        self.url = f'https://huggingface.co/MiniMaxAI/MiniMax-H3/resolve/{revision}/{folder}/'
        name = 'diffusion_pytorch_model.safetensors.index.json'
        self.index = json.loads((self.local / name).read_bytes() if self.local else
                                remote_bytes(self.url + name))['weight_map']
        self.headers = {}
        # Headers are small; fetch once before worker threads read projections.
        if not self.local:
            names = {v for k, v in self.index.items() if 'adaln_proj.linear' in k or
                     k.startswith(('time_embedder.', 'norm_out.linear.'))}
            for name in sorted(names):
                url = self.url + name
                length = struct.unpack('<Q', remote_bytes(url, 0, 7))[0]
                self.headers[name] = (8 + length, json.loads(remote_bytes(url, 8, 7 + length)))

    def tensor(self, key):
        name = self.index[key]
        if self.local:
            with safe_open(self.local / name, framework='pt') as f:
                return f.get_tensor(key)
        offset, header = self.headers[name]
        info = header[key]
        start, end = info['data_offsets']
        data = bytearray(remote_bytes(self.url + name, offset + start, offset + end - 1))
        dtype = {'F32': torch.float32, 'BF16': torch.bfloat16, 'F16': torch.float16}[info['dtype']]
        return torch.frombuffer(data, dtype=dtype).reshape(info['shape'])


def time_pairs(metadata):
    sigmas = torch.tensor(json.loads(metadata['hyperflow_sigmas']), dtype=torch.float32)
    if (sigmas.shape != (9,) or sigmas[0] != 1 or sigmas[-1] != 0 or
            not torch.isfinite(sigmas).all() or not (sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError('Expected a finite, decreasing eight-interval sigma grid')
    pairs = []
    for key in ('hyperflow_video_shift', 'hyperflow_audio_shift'):
        shift = float(metadata[key])
        if not math.isfinite(shift) or shift <= 0:
            raise ValueError('Invalid sigma shift')
        grid = shift * sigmas / (1 + (shift - 1) * sigmas)
        pairs.append(torch.stack((1 - grid[:-1], 1 - grid[1:]), dim=1))
    pairs.append(torch.tensor([[.999, .999], [1, 1]], dtype=torch.float32))
    return torch.cat(pairs)


def sinusoid(times, dim):
    # Diffusers Timesteps: flip_sin_to_cos=True, downscale_freq_shift=0.
    exponent = -math.log(10000) * torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    phase = times[:, None] * exponent.exp()[None, :]
    return torch.cat((phase.cos(), phase.sin()), dim=1)


def time_embedding(source, adapter, times, endpoint, alpha, strength):
    prefix = 'transformer.' + ('endpoint_time_embedder' if endpoint else 'time_embedder')
    x = sinusoid(times, source['linear_1.weight'].shape[1])
    for layer in ('linear_1', 'linear_2'):
        a = adapter.get_tensor(f'{prefix}.{layer}.lora_A.weight').float()
        b = adapter.get_tensor(f'{prefix}.{layer}.lora_B.weight').float()
        w, bias = source[layer + '.weight'].float(), source[layer + '.bias'].float()
        if a.shape != (int(adapter.metadata()['lora_rank']), w.shape[1]) or b.shape != (w.shape[0], a.shape[0]):
            raise ValueError('Invalid time-embedder LoRA shape')
        x = F.linear(x, w, bias) + F.linear(F.linear(x, a), b) * (strength * alpha / a.shape[0])
        if layer == 'linear_1':
            x = F.silu(x)
    return x


def prepare(args):
    if not math.isfinite(args.strength) or args.strength == 0:
        raise ValueError('Prepare a finite, nonzero adapter strength')
    digest_state = hashlib.sha256()
    with args.adapter.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest_state.update(chunk)
    digest = digest_state.hexdigest()
    output = args.output or args.adapter.with_suffix(f'.hyperflow-{args.variant}.safetensors')
    cache = output.with_suffix('.parts') if not args.source else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    with safe_open(args.adapter, framework='pt') as adapter:
        m = adapter.metadata()
        if m.get('hyperflow') != 'true' or m.get('hyperflow_version') != '1.0' or m.get('base_model') != 'MiniMaxAI/MiniMax-H3':
            raise ValueError('Expected a Hyperflow 1.0 H3 adapter')
        pairs = time_pairs(m)
        print('Reading original time embedder and full AdaLN projections', flush=True)
        source = Source(args.source, m['base_model_revision'], args.variant)
        mlp = {name: source.tensor('time_embedder.' + name) for name in
               ('linear_1.weight', 'linear_1.bias', 'linear_2.weight', 'linear_2.bias')}
        alpha, gate = float(m['lora_alpha']), float(m['hyperflow_gate'])
        if not math.isfinite(alpha) or not math.isfinite(gate):
            raise ValueError('Non-finite Hyperflow parameters')
        t = time_embedding(mlp, adapter, pairs[:, 0], False, alpha, args.strength)
        r = time_embedding(mlp, adapter, pairs[:, 1], True, alpha, args.strength)
        activated = F.silu(t + gate * (r - t))
    tensors = {'pairs': pairs}
    targets = [(f'transformer_blocks.{i}.adaln_proj.linear', f'blocks.{i}.adaln_proj.linear')
               for i in range(50)] + [('norm_out.linear', 'final_layer.adaln_proj.linear')]
    # At most two full projections live in memory; no whole-model download.
    def projection(target):
        upstream, native = target
        cache_path = cache / (native + '.safetensors') if cache else None
        cache_metadata = {'adapter_sha256': digest, 'variant': args.variant,
                          'strength': str(args.strength), 'version': '1',
                          'revision': m['base_model_revision']}
        if cache_path and cache_path.exists():
            with safe_open(cache_path, framework='pt') as saved:
                if saved.metadata() == cache_metadata:
                    return native, saved.get_tensor('modulation')
        w, b = source.tensor(upstream + '.weight'), source.tensor(upstream + '.bias')
        # Match the official BF16 projection, including its output rounding.
        result = F.linear(activated.to(w.dtype), w, b).float().T.contiguous()
        if not torch.isfinite(result).all():
            raise ValueError('Non-finite modulation: ' + upstream)
        if cache_path:
            save_file({'modulation': result}, str(cache_path) + '.tmp', metadata=cache_metadata)
            os.replace(str(cache_path) + '.tmp', cache_path)
        return native, result
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Submit in bounded batches; Executor.map would retain completed large tensors.
        for i in range(0, len(targets), 2):
            for name, tensor in pool.map(projection, targets[i:i+2]):
                tensors[name] = tensor
                print(f'Prepared {name} ({len(tensors)-1}/51)', flush=True)
    metadata = {'slopfab.hyperflow': '1', 'variant': args.variant,
                'base_model_revision': m['base_model_revision'], 'adapter_sha256': digest,
                'strength': str(args.strength)}
    if output.resolve() == args.adapter.resolve():
        raise ValueError('Companion output must not replace the adapter')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata=metadata)
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    print(f'Wrote {output} ({output.stat().st_size / 2**20:.1f} MiB)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Local original Diffusers transformer directory')
    parser.add_argument('--variant', choices=['fl2va', 'ref2va'], default='fl2va')
    parser.add_argument('--strength', type=float, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=8)
    prepare(parser.parse_args())
