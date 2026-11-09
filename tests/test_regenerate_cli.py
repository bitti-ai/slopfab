"""Regenerate CLI/JSON discovery, schedules and incompatible request rejection."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='slopfab-regenerate-') as temp:
    root = Path(temp)
    config = root/'config.json'
    config.write_text(json.dumps({'schema_version': 1, 'weight_paths': [str(root)]}))
    def weights(name, shapes, metadata=None):
        header, offset = {'__metadata__': metadata or {}}, 0
        for key, shape in shapes.items():
            count = 1
            for size in shape: count *= size
            header[key] = {'dtype': 'F32', 'shape': shape, 'data_offsets': [offset, offset+count*4]}
            offset += count*4
        data = json.dumps(header).encode()
        path = root/name
        path.write_bytes(struct.pack('<Q', len(data))+data+bytes(offset))
        return path
    weights('ref2va.safetensors', {'adaln_t_table': [1], 'blocks.0.adaln_proj.linear.weight': [1]})
    shapes = {'transformer.token_refiner.refiner_blocks.0.attn.to_k.lora_A.weight': [384, 5376]}
    shapes.update({f'fixture.{i}': [1] for i in range(623)})
    weights('open_h3_regenerate_2k_ic_lora.safetensors', shapes,
            {'target': 'MiniMaxAI/MiniMax-H3 transformer_ref (Ref2VA transformer)', 'training_step': '6000'})
    hf = weights('minimax_h3_hyperflow_8step_v1.0.safetensors', {'unused': [1]},
        {'hyperflow': 'true', 'hyperflow_version': '1.0', 'base_model': 'MiniMaxAI/MiniMax-H3',
         'base_model_revision': 'fixture', 'hyperflow_gate': '0.25', 'lora_rank': '1', 'lora_alpha': '2',
         'hyperflow_video_shift': '12', 'hyperflow_audio_shift': '3',
         'hyperflow_sigmas': '[1,0.931506,0.839236,0.703462,0.5,0.296538,0.160764,0.068494,0]'})
    weights('silence_audio_124f.safetensors', {'audio_latents': [2, 32, 207]})
    movie = root/'draft.y4m'
    movie.write_bytes(b'YUV4MPEG2 W32 H32 F24:1 Ip A1:1 C420jpeg\n' +
                     (b'FRAME\n' + bytes([128])*1536)*124)
    def run(*args, ok=True, request=None):
        result = subprocess.run([exe, '--agent', '--config', str(config), *map(str,args)], cwd=root,
            input=json.dumps(request) if request else None, capture_output=True, text=True, timeout=30)
        assert (result.returncode == 0) == ok, (result.stdout, result.stderr)
        return json.loads(result.stdout.splitlines()[-1])
    base = ['video', 'generate', '--regenerate-2k', 'hf8', '--reference-video', movie,
            '--resolution', '64x64', '--dry-run']
    plan = run(*base)['plans'][0]
    assert plan['steps'] == 8 and plan['regenerate_mode'] == 'hf8'
    assert plan['conditioning'] == {'pin_target_audio': True, 'include_reference_audio': False}
    # Explicit HyperFlow must not be added a second time by discovery.
    assert run(*base, '--lora', hf)['plans'][0]['steps'] == 8
    for extras, message in [(['--regenerate-2k', 'bad'], 'up3'),
                            (['--regenerate-2k', 'up3'], 'upsampler'),
                            (['--steps', '3'], 'step count'),
                            (['--frames', '90'], '124 frames'),
                            (['--synthetic-latents'], 'initializes'),
                            (['--resolution', '4096x2304'], '2560x1440')]:
        event = run(*base, *extras, ok=False)
        assert message in event['error']['message'], event
    request = {'schema_version': 1, 'task': 'video', 'resolution': {'width': 64, 'height': 64},
               'references': [{'type': 'video', 'path': str(movie)}],
               'options': {'regenerate-2k': 'hf8'}}
    assert run('requests', 'validate', '--request', '-', request=request)['plans'][0]['steps'] == 8
    assert run('system', 'capabilities')['data']['regenerate_2k']['frames'] == 124
    text = subprocess.run([exe, 'help', 'video', 'generate'], capture_output=True, text=True).stdout
    assert '--regenerate-2k' in text
print('PASS: Regenerate CLI/JSON, weight discovery and recipe validation')
