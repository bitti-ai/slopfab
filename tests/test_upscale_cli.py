"""Exercise the shared selector through both public CLI commands, without weights."""
import pathlib
import subprocess
import sys
import tempfile

exe = str(pathlib.Path(sys.argv[1]).resolve())


def run(args, expected=0):
    result = subprocess.run([exe, *args], capture_output=True, text=True)
    assert (result.returncode == 0) == (expected == 0), result.stdout + result.stderr
    return result.stdout + result.stderr


with tempfile.TemporaryDirectory() as directory:
    source = pathlib.Path(directory) / 'input with spaces.mp4'
    source.touch()  # dry-run must not decode media or load models
    standalone = ['upscale', '--input', str(source), '--out', str(source.with_name('out.mp4')),
                  '--inference-backend', 'cuda', '--dry-run']
    generate = ['generate', '--prompt', 'test', '--resolution', '32x32', '--frames', '22',
                '--inference-backend', 'cuda', '--dry-run']
    for command in [standalone, generate]:
        for method in ['realesrgan', 'seedvr2']:
            assert method in run(command + ['--upscale-method', method])
        run(command + ['--upscale-method', 'unknown'], expected=1)
        run(command + ['--upscale-method', 'seedvr2', '--upscale-segment-frames', '6'], expected=1)
        run(command + ['--upscale-method', 'seedvr2', '--upscale-tile', '64'], expected=1)
        run(command + ['--upscale-method', 'seedvr2', '--inference-backend', 'vulkan'], expected=1)
    custom = ['--upscale-method', 'seedvr2', '--upscale-resolution', '320x192', '--upscale-tile', '0']
    assert '320x192' in run(generate + custom)
    assert '128x128' in run(generate + ['--upscale-method', 'realesrgan'])
    assert run(['upscale', '--help']).count('usage:') == 1
