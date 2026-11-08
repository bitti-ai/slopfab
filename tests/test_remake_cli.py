"""Native Remake CLI/JSON plans, mode validation and merged checkpoint guards."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix='slopfab-remake-') as temp:
    root = Path(temp)
    config = root/'config.json'
    config.write_text(json.dumps({'schema_version': 1, 'weight_paths': [str(root)]}))
    movie = root/'source.y4m'
    movie.write_bytes(b'YUV4MPEG2 W32 H32 F24:1 Ip A1:1 C420jpeg\n' +
                     (b'FRAME\n' + bytes([128])*1536)*124)
    model = root/'viggle_remake_swap_pruned_int8_convrot.safetensors'
    header = json.dumps({n: {'dtype':'F32', 'shape':[1], 'data_offsets':[i*4,i*4+4]}
                        for i,n in enumerate(['adaln_t_table','blocks.0.adaln_proj.linear.weight'])}).encode()
    model.write_bytes(struct.pack('<Q',len(header))+header+bytes(8))
    prompt = 'subject_definitions: robot\nsummary: dance\nretention_analysis: same motion\ndetailed_description: dancing\noverall_soundscape: source\nnon_diegetic_music: source'
    def run(*args, ok=True, request=None):
        result = subprocess.run([exe,'--agent','--config',str(config),*map(str,args)],cwd=root,
            input=json.dumps(request) if request else None,capture_output=True,text=True,timeout=30)
        assert (result.returncode == 0) == ok,(result.stdout,result.stderr)
        return json.loads(result.stdout.splitlines()[-1])
    base = ['video','generate','--remake','swap','--prompt',prompt,'--reference-video',movie,
            '--reference-image','painted.png','--frames','124','--resolution','32x32','--dry-run']
    plan = run(*base)['plans'][0]
    assert plan['steps'] == 6 and plan['remake_mode'] == 'swap'
    assert plan['conditioning'] == {'pin_target_audio':True,'include_reference_audio':False}
    assert 'fixed 362-token' not in plan['description']
    assert 'shift 12)' in plan['description']
    for extras, message in [(['--remake','edit','--transformer',model],'does not match'),
                            (['--lora','adapter.safetensors'],'already contain'),
                            (['--steps','4'],'six-step'),
                            (['--frames','125'],'17*n+5'),
                            (['--remake','bad'],'swap or edit'),
                            (['--reference-audio','absent.wav'],'')]:
        event = run(*base,*extras,ok=False)
        assert message in event['error']['message'],event
    run(*base,'--reference-image','second.png','--reference-image','third.png','--reference-image','fourth.png')
    run(*base,*sum([['--reference-image',str(i)+'.png'] for i in range(4)],[]),ok=False)
    request = {'schema_version':1,'task':'video','prompt':prompt,'resolution':{'width':32,'height':32},
               'references':[{'type':'video','path':str(movie)},{'type':'image','path':'painted.png'}],
               'options':{'remake':'swap','frames':124}}
    assert run('requests','validate','--request','-',request=request)['plans'][0]['steps'] == 6
    caps = run('system','capabilities')['data']['viggle_remake']
    assert caps['steps'] == 6 and not caps['merged_model_needs_lora']
    help_text = subprocess.run([exe,'help','video','generate'],capture_output=True,text=True).stdout
    assert '--remake' in help_text and '--animate' not in help_text
print('PASS: Remake CLI/JSON, discovery, six-step plans, merged guards and silent sources')
