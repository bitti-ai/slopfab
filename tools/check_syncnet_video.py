"""Optional real-video SyncNet regression (torch/numpy/python_speech_features/FFmpeg).

Usage: python tools/check_syncnet_video.py REFERENCE_DIRECTORY SLOPFAB_EXECUTABLE
The directory contains the hash-locked models, upstream SyncNetModel.py and
https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/example.avi.
All generated test artifacts stay in that directory.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import python_speech_features
import torch
from syncnet_manifest import MODELS

root = Path(sys.argv[1]).resolve()
exe = str(Path(sys.argv[2]).resolve())
ffmpeg = shutil.which('ffmpeg')
for filename, (_, digest) in MODELS.items():
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest() == digest
assert hashlib.sha256((root/'SyncNetModel.py').read_bytes()).hexdigest() == '41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22'
sys.path.insert(0, str(root))
from SyncNetModel import S


def analyze(video, name, *options):
    command = [exe, 'video', 'syncnet', video, '--model', 'syncnet_v2.model', '--duration', '4',
               '--backend', 'vulkan', '--out', name, '--overwrite', *options]
    p = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, (p.stdout, p.stderr)
    return json.loads(p.stdout)


base = analyze('example.avi', 'cropped-report.json', '--cropped')['tracks'][0]
video = subprocess.check_output([ffmpeg, '-v', 'error', '-i', str(root/'example.avi'), '-t', '4',
                                '-vf', 'fps=25', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
audio = subprocess.check_output([ffmpeg, '-v', 'error', '-i', str(root/'example.avi'), '-t', '4',
                                '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'])
frames = torch.from_numpy(np.frombuffer(video, np.uint8).copy().reshape(-1, 224, 224, 3)).permute(3, 0, 1, 2).float()
mfcc = torch.tensor(python_speech_features.mfcc(np.frombuffer(audio, '<i2'), samplerate=16000).T, dtype=torch.float32)
net = S()
net.load_state_dict(torch.load(root/'syncnet_v2.model', map_location='cpu', weights_only=True))
net.eval()
torch.set_num_threads(8)
n = min(frames.shape[1], len(audio)//1280)-5
visual, sound = [], []
with torch.no_grad():
    for start in range(0, n, 16):
        indices = range(start, min(start+16, n))
        visual.append(net.forward_lip(torch.stack([frames[:, i:i+5] for i in indices])))
        sound.append(net.forward_aud(torch.stack([mfcc[:, i*4:i*4+20] for i in indices]).unsqueeze(1)))
v, a = torch.cat(visual), torch.cat(sound)
padded = torch.nn.functional.pad(a, (0, 0, 15, 15))
distances = torch.stack([torch.nn.functional.pairwise_distance(v, padded[i:i+n]).mean() for i in range(31)])
best = int(distances.argmin())
assert base['offset_frames'] == 15-best
np.testing.assert_allclose(base['shift_distances'], distances.numpy(), atol=3e-4, rtol=3e-4)
print('Upstream video parity:', base['offset_frames'], 'frames; LSE-D', base['lse_d'])

subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(root/'example.avi'), '-t', '4', '-af',
                'adelay=160:all=1', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(root/'delayed.mkv')], check=True)
delayed = analyze('delayed.mkv', 'delayed-report.json', '--cropped')['tracks'][0]
assert delayed['offset_frames'] == base['offset_frames']-4, (base, delayed)
print('160 ms audio delay changes offset by -4 frames')

subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(root/'example.avi'), '-t', '4', '-vf',
                'pad=640:360:208:68', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(root/'scene.mkv')], check=True)
tracked = analyze('scene.mkv', 'tracked-report.json', '--face-model', 'sfd_face.pth')['tracks']
assert len(tracked) == 1 and len(tracked[0]['detections']) > 10
assert tracked[0]['offset_frames'] == base['offset_frames']
print('S3FD full-scene detection and tracking:', len(tracked[0]['detections']), 'detections')

# Safetensors is an architecture-validated alternative, with no file hash pin.
state = torch.load(root/'sfd_face.pth', map_location='cpu', weights_only=True)
header, chunks, offset = {}, [], 0
for key, value in state.items():
    data = value.numpy().astype('<f4').tobytes()
    header[key] = {'dtype': 'F32', 'shape': list(value.shape), 'data_offsets': [offset, offset+len(data)]}
    chunks.append(data)
    offset += len(data)
encoded = json.dumps(header).encode()
encoded += b' ' * (-len(encoded) % 8)
(root/'sfd_face.safetensors').write_bytes(len(encoded).to_bytes(8, 'little')+encoded+b''.join(chunks))
safe = analyze('scene.mkv', 'safetensors-report.json', '--face-model', 'sfd_face.safetensors')['tracks']
assert safe == tracked
print('Unpinned FP32 S3FD safetensors gives identical results')

# A stream timestamp delay must survive the two decoder invocations.
subprocess.run([ffmpeg, '-y', '-v', 'error', '-i', str(root/'example.avi'), '-itsoffset', '.16',
                '-i', str(root/'example.avi'), '-map', '0:v:0', '-map', '1:a:0', '-t', '4',
                '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(root/'timestamp-delay.mkv')], check=True)
timestamp = analyze('timestamp-delay.mkv', 'timestamp-report.json', '--cropped')['tracks'][0]
assert timestamp['offset_frames'] == base['offset_frames']-4
print('160 ms stream timestamp delay is preserved')
