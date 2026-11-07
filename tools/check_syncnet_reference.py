"""Compare test_syncnet's optional real-model outputs with upstream PyTorch.

Pass a directory containing the pinned weights, upstream SyncNetModel.py and
native-*.f32 from slopfab_syncnet_tests DIRECTORY BACKEND. Developer-only;
requires torch, numpy, python_speech_features. No Python dependency at runtime.
"""
import hashlib
import pathlib
import sys

import numpy as np
import torch
import python_speech_features

from syncnet_manifest import MODELS

root = pathlib.Path(sys.argv[1]).resolve()
for filename, (_, digest) in MODELS.items():
    assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == digest
assert hashlib.sha256((root / 'SyncNetModel.py').read_bytes()).hexdigest() == (
    '41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22')
sys.path.insert(0, str(root))
from SyncNetModel import S


def compare(name, reference, atol=2e-4):
    native = np.fromfile(root / f'native-{name}.f32', np.float32)
    reference = np.asarray(reference, np.float32).ravel()
    print(name, 'max absolute error', abs(native - reference).max())
    np.testing.assert_allclose(native, reference, atol=atol, rtol=2e-4)


net = S()
net.load_state_dict(torch.load(root / 'syncnet_v2.model', map_location='cpu', weights_only=True))
net.eval()
with torch.no_grad():
    a = torch.tensor((np.arange(260) % 37 - 18).astype(np.float32)).reshape(1, 1, 13, 20)
    lip = torch.tensor((np.arange(224*224*15) % 251).astype(np.float32))
    lip = lip.reshape(224, 224, 3, 5).permute(2, 3, 0, 1).unsqueeze(0)
    compare('audio', net.forward_aud(a).numpy())
    compare('lip', net.forward_lip(lip).numpy())
    w = torch.load(root / 'sfd_face.pth', map_location='cpu', weights_only=True)
    x = torch.tensor((np.arange(64*96*3) % 251 - 117).astype(np.float32))
    x = x.reshape(64, 96, 3).permute(2, 0, 1).unsqueeze(0)
    out = torch.nn.functional.conv2d(x, w['vgg.0.weight'], w['vgg.0.bias'], padding=1).relu()
    compare('detector-first', out.permute(0, 2, 3, 1).numpy(), 3e-4)
pcm = (np.arange(16327)*7919 % 60001 - 30000).astype(np.int16)
compare('mfcc', python_speech_features.mfcc(pcm, samplerate=16000), 1e-4)
