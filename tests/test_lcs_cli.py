"""Model-free validation of native LCS CLI wiring and optional calibration data."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())


def run(*args, success=True):
    result = subprocess.run([exe, 'generate', '--transformer', 'missing-lcs-test.safetensors',
                             '--dry-run', *map(str, args)], capture_output=True, text=True)
    assert (result.returncode == 0) == success, (args, result.stdout, result.stderr)


def archive(path, extra=False):
    anchors = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    tensors = {'basis': ([96, 3], [float(i//3 == i%3) for i in range(288)]),
               'mean': ([96], [0.]*96), 'anchor_lcs': ([8, 3], anchors),
               'latent_mean': ([24], [0.]*24), 'latent_std': ([24], [1.]*24),
               'explained_variance': ([1], [1.])}
    if extra:
        tensors['drift_direction'] = ([96], [float(i == 3) for i in range(96)])
        tensors['drift_reference'] = ([2], [0., .1])
        tensors['color_reference'] = ([5, 2, 3], [-1., 0., 1.]*10)
        tensors['edge_reference'] = ([5], [.1]*5)
    header = {'__metadata__': {'slopfab_lcs': '1', 'vae_sha256': 'a'*64}}
    payload = b''
    for name, (shape, values) in tensors.items():
        raw = struct.pack('<' + 'f'*len(values), *values)
        header[name] = dict(dtype='F32', shape=shape, data_offsets=[len(payload), len(payload)+len(raw)])
        payload += raw
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


run('--lcs-settings', '{}', success=False)
with tempfile.TemporaryDirectory(prefix='slopfab-lcs-cli-') as directory:
    path = Path(directory)/'lcs.safetensors'
    archive(path)
    run('--lcs', path)
    run('--lcs', path, '--lcs-settings', '{"highlight_strength":0.5,"highlight_knee":0.78}')
    run('--lcs', path, '--lcs-settings', '{"drift_strength":0.5}', success=False)
    run('--lcs', path, '--lcs-settings', '{"color_stability":0.5}', success=False)
    run('--lcs', path, '--lcs-settings', '{"edge_stability":0.5}', success=False)
    archive(path, extra=True)
    run('--lcs', path, '--lcs-settings', '{"drift_strength":0.5}')
    run('--lcs', path, '--lcs-settings', '{"color_stability":0.5,"edge_stability":0.5}')
    for settings in ('{"highlight_strength":2}', '{"highlight_knee":-1}', '{"drift_max":-1}',
                     '{"color_stability":2}', '{"edge_stability":-1}', '{"color_max_shift":2}',
                     '{"typo":1}', '[]', '{broken'):
        run('--lcs', path, '--lcs-settings', settings, success=False)
    run('--lcs', path, '--synthetic-latents', success=False)
print('LCS CLI checks passed')
