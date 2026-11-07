"""Media group routing, help, opaque option values, and legacy compatibility."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='slopfab-media-commands-') as tmp:
    root = Path(tmp)
    def run(*args, code=0, input=None):
        p = subprocess.run([exe, '--agent', '--config', str(root/'config.json'), *args],
                           cwd=root, capture_output=True, text=True, input=input, timeout=30)
        assert p.returncode == code, (args, p.returncode, p.stdout, p.stderr)
        return json.loads(p.stdout.splitlines()[-1])
    groups = {'video': ['generate', 'frames', 'syncnet', 'upscale', 'upscale-latents', 'decode', 'compare'],
              'audio': ['generate', 'transcribe', 'separate'], 'image': ['generate', 'upscale']}
    listed = {c['command'] for c in run('help')['commands']}
    assert set(groups) <= listed
    assert 'motions' in listed
    assert not {'transcribe', 'separate', 'decode', 'upscale', 'compare-y4m'} & listed
    assert {'generate', 'validate', 'inspect', 'compare', 'encode-refmod', 'bundle-refmods'} <= listed
    for group, commands in groups.items():
        assert run(group)['command'] == group
        for command in commands:
            for args in [('help', group, command), (group, command, '--help'), (group, 'help', command)]:
                h = run(*args)
                assert h['command'] == group + ' ' + command, h
                assert h['usage'].startswith('slopfab ' + group + ' ' + command), h
        assert run(group, 'does-not-exist', code=2)['error']['code'] == 'INVALID_REQUEST'
    for group, subcommand in [('image', 'frame'), ('video', 'render')]:
        assert run(group, 'motions', subcommand, code=2)['error']['code'] == 'INVALID_REQUEST'
    for subcommand in ['render', 'frame']:
        assert run('motions', subcommand, '--help')['command'] == 'motions'
    caps = run('capabilities')['data']
    assert caps['motions']['command'].startswith('slopfab motions ')
    assert all('motions' not in commands for commands in caps['media_commands'].values())
    request = {'schema_version': 1, 'task': 'image', 'model': 'krea2', 'prompt': 'apple', 'seed': '42',
               'resolution': {'width': 512, 'height': 512}, 'output': {'path': 'apple.png'},
               'options': {'vulkan-arithmetic': 'portable'}}
    a = run('image', 'generate', '--request', '-', '--dry-run', input=json.dumps(request))
    b = run('generate', '--request', '-', '--dry-run', input=json.dumps(request))
    assert a['plans'] == b['plans']
    assert run('audio', 'generate', '--request', '-', '--dry-run', input=json.dumps(request), code=2)['error']['field'] == 'task'
    for group, task in [('video', 'video'), ('image', 'image'), ('audio', 'music')]:
        flags = ['--prompt', '--help', '--seed', '42', '--dry-run', '--vulkan-arithmetic', 'portable']
        if task == 'image': flags += ['--resolution', '512x512', '--out', 'x.png']
        elif task == 'video': flags += ['--resolution', '256x256', '--frames', '22', '--out', 'x.y4m']
        else: flags += ['--model', 'minimax_music3', '--duration', '2', '--out', 'x.wav']
        a = run(group, 'generate', *flags)
        b = run('generate', task, *flags)
        assert a['type'] != 'help' and a['plans'] == b['plans']
    for command in ['transcribe', 'separate', 'motions', 'upscale', 'upscale-latents', 'decode', 'compare-y4m']:
        assert run(command, '--help')['type'] == 'help'
    # Boolean flags must not swallow global --agent or subcommand --help.
    assert run('audio', 'transcribe', '--word-timestamps', '--help')['command'] == 'audio transcribe'
    assert run('video', 'syncnet', '--cropped', '--help')['command'] == 'video syncnet'
print('Media command routing tests passed')
