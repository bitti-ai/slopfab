"""Paired long H3 continuation experiment using the existing DLL API.

Each window completes sampling, output LCS, joining and atomic archive saving.
The progress callback then cancels at the VIDEO_DECODE boundary to defer decoding.
Only that specific cancellation, with a validated archive and exact retained
prefixes, counts as a completed latent window. This changes no sampling settings.
Decode saved milestones/final archives with the normal `slopfab decode` command.
Requires numpy (archive validation), and the local lcs_drift helper.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from lcs_drift import read


class Progress(C.Structure):
    _fields_ = [('stage', C.c_int32), ('step', C.c_int32),
                ('total_steps', C.c_int32), ('elapsed_seconds', C.c_double)]


def save_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def validate_extension(previous, path, added_frames):
    old_meta, old = read(previous)
    meta, current = read(path)
    if int(meta['frames']) != int(old_meta['frames']) + added_frames:
        raise ValueError('Unexpected joined frame count')
    for key in ('width', 'height', 'fps', 'video_vae', 'audio_vae', 'transformer'):
        # Paths may be absolute in DLL requests and relative in earlier CLI runs.
        if key.endswith('vae') or key == 'transformer':
            equal = Path(meta[key]).resolve() == Path(old_meta[key]).resolve()
        else:
            equal = meta[key] == old_meta[key]
        if not equal:
            raise ValueError(f'Archive metadata changed: {key}')
    for value in current.values():
        if not np.isfinite(value).all():
            raise ValueError('Non-finite latent output')
    if not np.array_equal(old['video_rows'],current['video_rows'][:len(old['video_rows'])]):
        raise ValueError('Retained video prefix changed')
    old_audio, new_audio = old['audio_rows'].reshape(2,-1,32), current['audio_rows'].reshape(2,-1,32)
    if not np.array_equal(old_audio,new_audio[:,:old_audio.shape[1]]):
        raise ValueError('Retained audio prefix changed')
    return int(meta['frames'])


def verify_first_correction(source, baseline, corrected):
    """Same-source/seed output-only LCS probe, before the chains diverge."""
    meta, old = read(source)
    _, plain = read(baseline)
    _, fixed = read(corrected)
    if not np.array_equal(plain['audio_rows'],fixed['audio_rows']):
        raise ValueError('Output-only LCS changed same-seed audio')
    prefix = len(old['video_rows'])
    if not np.array_equal(plain['video_rows'][:prefix],fixed['video_rows'][:prefix]):
        raise ValueError('Same-source first pair has different retained prefixes')
    plane = (int(meta['width'])//32)*(int(meta['height'])//32)
    delta = (fixed['video_rows'][prefix:] - plain['video_rows'][prefix:]).reshape(-1,plane,96)
    maxima = np.max(np.abs(delta),axis=(1,2))
    if not len(maxima) or not np.isfinite(maxima).all() or not (maxima > 0).all():
        raise ValueError('First-pair probe did not observe a finite correction in every new latent slice')
    return dict(new_latent_slices=len(maxima),all_new_slices_changed=True,
                per_slice_max_abs_change=maxima.tolist(),audio_exact=True,retained_prefix_exact=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('transformer','embedding','baseline-source','corrected-source','calibration','settings','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--dll',type=Path,default=Path('build/Release/slopfab.dll'))
    p.add_argument('--vae',type=Path,default=Path('weights/vae/minimax_h3_video_vae_fp16.safetensors'))
    p.add_argument('--audio-vae',type=Path,default=Path('weights/vae/minimax_h3_audio_vae_fp32.safetensors'))
    p.add_argument('--pairs',type=int,default=32)
    p.add_argument('--seed',type=int,default=424253)
    p.add_argument('--steps',type=int,default=25)
    p.add_argument('--new-frames',type=int,default=68)
    p.add_argument('--decode-every',type=int,default=0,help='Decode both saved chains every N pairs, and the final pair')
    p.add_argument('--keystone-source',type=Path,help='Fixed original-clip anchor for both arms; enables front Keystone and rear Capstone')
    p.add_argument('--capstone-lock-after',type=int,default=4,help='Video updates before rear Capstone context is appended')
    p.add_argument('--lock-overlap',action='store_true',help='Explicitly enable locked continuation overlap in both arms')
    p.add_argument('--verify-first-lcs',action='store_true',help='Require identical starting sources and verify actual output correction across every new slice of the first pair')
    p.add_argument('--exe',type=Path,default=Path('build/Release/slopfab.exe'))
    a = p.parse_args()
    if a.pairs < 1 or a.steps < 1 or a.new_frames < 17 or a.new_frames % 17:
        p.error('positive pairs, steps >=1 and new frames a positive multiple of 17 required')
    if a.decode_every < 0:
        p.error('decode-every must be nonnegative')
    if a.capstone_lock_after < 0 or (a.keystone_source and a.capstone_lock_after >= a.steps):
        p.error('capstone-lock-after must be nonnegative and precede the final denoising update')
    if a.output.exists() and any(a.output.iterdir()):
        p.error('Use a fresh output directory; completed archives can be resumed as new sources')
    a.output.mkdir(parents=True,exist_ok=True)
    settings = a.settings.read_text(encoding='utf-8-sig')
    parsed_settings = json.loads(settings)
    if a.verify_first_lcs:
        if a.baseline_source.resolve() != a.corrected_source.resolve():
            p.error('verify-first-lcs requires the same starting source for both arms')
        if set(parsed_settings) - {'color_stability','edge_stability','color_max_shift'}:
            p.error('verify-first-lcs requires output-only stability settings')
    manifest = {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items()}
    manifest.update(settings_json=json.loads(settings),attention='flash2',backend='cuda',overlap_frames=22,
                    keystone_enabled=bool(a.keystone_source),capstone_enabled=bool(a.keystone_source),
                    arms={'baseline':'continuation without LCS','corrected':'same continuation with LCS'},
                    output_policy='cancel at video decode only, after completed latent archive save')
    watched=[a.transformer,a.vae,a.audio_vae,a.embedding,a.calibration,a.settings]
    if a.keystone_source:
        watched.append(a.keystone_source)
    signatures={str(path.resolve()):(path.stat().st_size,path.stat().st_mtime_ns) for path in watched}
    manifest['input_file_signatures']=signatures
    save_json(a.output/'manifest.json',manifest)
    lib = C.CDLL(str(a.dll.resolve()))
    pointer, integer, string = C.c_void_p,C.c_int32,C.c_char_p
    def bind(name,result,*args):
        fn=getattr(lib,'slopfab_'+name); fn.restype=result; fn.argtypes=args; return fn
    error=bind('last_error',string)
    def check(status):
        if status:
            raise RuntimeError(f'{status}: {error().decode()}')
    create=bind('request_create',pointer)
    destroy=bind('request_destroy',None,pointer)
    set_path=bind('request_set_model_path',integer,pointer,integer,string)
    set_embedding=bind('request_set_prompt_embedding_path',integer,pointer,string)
    set_frames=bind('request_set_frames',integer,pointer,integer)
    set_steps=bind('request_set_steps',integer,pointer,integer)
    set_seed=bind('request_set_seed',integer,pointer,C.c_uint64)
    set_attention=bind('request_set_attention',integer,pointer,string)
    set_backend=bind('request_set_inference_backend',integer,pointer,integer)
    set_verbose=bind('request_set_verbose',integer,pointer,integer)
    set_save=bind('request_set_save_latents',integer,pointer,string)
    set_source=bind('request_set_continuation_file',integer,pointer,string,integer)
    set_lock=bind('request_set_continuation_lock_overlap',integer,pointer,integer)
    set_keystone=bind('request_set_keystone_file',integer,pointer,string,integer,integer)
    set_lcs=bind('request_set_lcs',integer,pointer,string,string)
    callback_type=C.CFUNCTYPE(None,C.POINTER(Progress),pointer)
    start=bind('generation_start',integer,pointer,callback_type,pointer,C.POINTER(pointer))
    wait=bind('generation_wait',integer,pointer,integer)
    cancel=bind('generation_cancel',None,pointer)
    gen_error=bind('generation_error',string,pointer)
    gen_destroy=bind('generation_destroy',None,pointer)
    sources={'baseline':a.baseline_source,'corrected':a.corrected_source}
    records=[]; decoded=[]
    for index in range(a.pairs):
        for label in ('baseline','corrected'):
            for path in watched:
                if (path.stat().st_size,path.stat().st_mtime_ns) != signatures[str(path.resolve())]:
                    raise RuntimeError(f'Experiment input changed during the run: {path}')
            request=create(); generation=pointer(); reached_decode=[]; completed_steps=[]; callback_errors=[]
            path=a.output/f'{label}-{index:02d}.safetensors'
            @callback_type
            def progress(info,userdata):
                try:
                    s=info.contents
                    if s.stage==4:
                        completed_steps.append((s.step,s.total_steps))
                    if s.stage==5:
                        reached_decode.append(True)
                        cancel(generation)
                except BaseException as exc:
                    callback_errors.append(repr(exc)); cancel(generation)
            began=time.monotonic()
            try:
                for kind,file in [(0,a.transformer),(3,a.vae),(4,a.audio_vae)]:
                    check(set_path(request,kind,str(file.resolve()).encode()))
                check(set_embedding(request,str(a.embedding.resolve()).encode()))
                check(set_frames(request,a.new_frames)); check(set_steps(request,a.steps))
                check(set_seed(request,a.seed+index)); check(set_attention(request,b'flash2'))
                check(set_backend(request,0)); check(set_verbose(request,0))
                check(set_source(request,str(sources[label].resolve()).encode(),22))
                check(set_lock(request,int(a.lock_overlap)))
                if a.keystone_source:
                    check(set_keystone(request,str(a.keystone_source.resolve()).encode(),1,a.capstone_lock_after))
                check(set_save(request,str(path.resolve()).encode()))
                if label=='corrected':
                    check(set_lcs(request,str(a.calibration.resolve()).encode(),settings.encode()))
                print(f'{label} {index+1}/{a.pairs}: seed {a.seed+index}',flush=True)
                check(start(request,progress,None,C.byref(generation)))
                while True:
                    status=wait(generation,1000)
                    if status != -7:
                        break
                if status != -8 or len(reached_decode)!=1 or callback_errors or not path.exists():
                    raise RuntimeError(f'Unexpected completion {status}: {gen_error(generation).decode()}; {callback_errors}')
                # H3 reports zero-based evaluation indices, plus an initial -1.
                if {s for s,n in completed_steps if n==a.steps and s>=0} != set(range(a.steps)):
                    raise RuntimeError(f'Incomplete sampling progress: {completed_steps}')
                frames=validate_extension(sources[label],path,a.new_frames)
                elapsed=time.monotonic()-began
                records.append(dict(label=label,index=index,seed=a.seed+index,frames=frames,
                    source_archive=str(sources[label].resolve()),lcs_enabled=label=='corrected',
                    seconds=elapsed,archive=str(path),video_prefix_exact=True,audio_prefix_exact=True,
                    sampled_updates=a.steps,decode_deferred=True,status=status))
                save_json(a.output/'progress.json',records)
                print(f'  saved {frames} frames ({frames/24:.2f}s), {elapsed:.1f}s; prefixes exact',flush=True)
                sources[label]=path
            finally:
                if generation:
                    gen_destroy(generation)
                destroy(request)
        if index == 0 and a.verify_first_lcs:
            verification=verify_first_correction(a.baseline_source,sources['baseline'],sources['corrected'])
            save_json(a.output/'first-correction-check.json',verification)
            print(f"First LCS pair: all {verification['new_latent_slices']} new latent slices corrected; audio and prefix exact",flush=True)
        if a.decode_every and ((index+1) % a.decode_every == 0 or index+1 == a.pairs):
            for label in ('baseline','corrected'):
                archive=sources[label]
                output=archive.with_suffix('.y4m')
                command=[str(a.exe.resolve()),'decode','--vae',str(a.vae.resolve()),
                         '--latent',str(archive.resolve()),'--out',str(output.resolve())]
                print(f'Decoding {label} milestone {index+1}/{a.pairs}',flush=True)
                with archive.with_suffix('.decode.log').open('w') as log:
                    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                decoded.append(dict(label=label,index=index,video=str(output),command=command))
                save_json(a.output/'decoded.json',decoded)


if __name__=='__main__':
    main()
