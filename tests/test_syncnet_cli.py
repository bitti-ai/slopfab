"""Fail closed on model bytes before running media tools or writing a report."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='slopfab-syncnet-') as tmp:
    root = Path(tmp)
    def run(*args, code=2):
        p = subprocess.run([exe, '--agent', '--config', str(root/'config.json'), 'video', 'syncnet', *args],
                           cwd=root, capture_output=True, text=True, timeout=30)
        assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
        return json.loads(p.stdout.splitlines()[-1])
    assert '961e8696' in run('--help', code=0)['options']
    (root/'clip.mp4').write_bytes(b'not video')
    (root/'wrong.model').write_bytes(b'not pickle')
    args = ['clip.mp4', '--model', 'wrong.model', '--cropped', '--ffmpeg', 'must-not-run', '--out', 'result.json']
    assert run(*args, code=3)['error']['code'] == 'INVALID_MODEL'
    # Correct length is insufficient; a hash mismatch is still rejected.
    with (root/'wrong.model').open('wb') as f: f.truncate(54573114)
    e = run(*args, code=3)['error']
    assert e['code'] == 'INVALID_MODEL' and '961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442' in e['message']
    assert not (root/'result.json').exists()
    assert run(*args, '--face-model', 'sfd_face.pth')['error']['code'] == 'INVALID_REQUEST'
    for flag, value in [('--duration', 'nan'), ('--vshift', '-1'), ('--detect-every', '0'), ('--detect-size', '99999'), ('--backend', 'cuda')]:
        assert run(*args, flag, value)['error']['code'] == 'INVALID_REQUEST'
    assert run('clip.mp4', '--model', 'wrong.model')['error']['code'] == 'INVALID_REQUEST'
print('SyncNet CLI validation tests passed')
