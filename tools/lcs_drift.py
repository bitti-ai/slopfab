"""Opt-in H3 continuation experiment. Requires numpy, Pillow and FFmpeg.

Generate saves every cumulative latent archive and video, plus exact commands.
Analyze reports frame-by-frame appearance proxies, not a perceptual gloss score.
Fit learns a color-orthogonal direction from early/late clean latent distributions.
Use a separate continuation/seed to evaluate the fitted correction.
"""
import argparse
import csv
import ctypes as C
import json
from pathlib import Path
import shutil
import struct
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    with Path(path).open('rb') as f:
        length, = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(length))
        start = f.tell()
        arrays = {}
        for name, info in header.items():
            if name == '__metadata__':
                continue
            dtype = {'F32': '<f4', 'I32': '<i4'}[info['dtype']]
            lo, hi = info['data_offsets']
            f.seek(start + lo)
            arrays[name] = np.frombuffer(f.read(hi-lo), dtype=dtype).reshape(info['shape']).copy()
    return header.get('__metadata__', {}), arrays


def write(path, metadata, arrays):
    header, payload, offset = {'__metadata__': metadata}, [], 0
    for name, array in arrays.items():
        array = np.asarray(array, dtype='<f4')
        raw = array.tobytes()
        header[name] = dict(dtype='F32', shape=list(array.shape), data_offsets=[offset, offset+len(raw)])
        payload.append(raw)
        offset += len(raw)
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    temporary = Path(str(path) + '.tmp')
    temporary.write_bytes(struct.pack('<Q', len(raw)) + raw + b''.join(payload))
    temporary.replace(path)


def run(command, log):
    print(' '.join(map(str, command)), flush=True)
    with Path(log).open('w') as f:
        subprocess.run(list(map(str, command)), stdout=f, stderr=subprocess.STDOUT, check=True)


def generate(a):
    a.output.mkdir(parents=True, exist_ok=True)
    settings = a.lcs_settings_file.read_text() if a.lcs_settings_file else a.lcs_settings
    json.loads(settings)  # Fail before expensive generation on malformed settings.
    manifest = a.output / f'{a.name}-commands.json'
    if manifest.exists() or any(a.output.glob(f'{a.name}-[0-9][0-9].*')):
        raise FileExistsError('Experiment name already exists; choose a fresh name/output')
    previous = a.source
    commands = []
    for i in range(a.windows):
        stem = a.output / f'{a.name}-{i:02d}'
        command = [a.exe, 'generate', '--transformer', a.transformer, '--vae', a.vae,
                   '--audio-vae', a.audio_vae, '--prompt-embedding', a.embedding,
                   '--resolution', a.resolution, '--frames', str(a.new_frames if previous else a.first_frames),
                   '--steps', str(a.steps), '--attention', 'flash2', '--seed', str(a.seed+i),
                   '--out', stem.with_suffix('.mp4'), '--save-latents', stem.with_suffix('.safetensors')]
        if previous:
            command += ['--continue-from', previous, '--overlap-frames', '22']
        if a.lcs:
            command += ['--lcs', a.lcs, '--lcs-settings', settings]
        commands.append(list(map(str, command)))
        manifest.write_text(json.dumps(commands, indent=2))
        if stem.with_suffix('.safetensors').exists() or stem.with_suffix('.mp4').exists():
            raise FileExistsError(f'{stem}: choose a fresh name/output or use --source to continue')
        run(command, stem.with_suffix('.log'))
        # Some native builds have FFmpeg disabled and emit Y4M/WAV instead.
        if not stem.with_suffix('.mp4').exists() and stem.with_suffix('.y4m').exists():
            mux = [a.ffmpeg, '-v', 'error', '-n', '-i', stem.with_suffix('.y4m')]
            if stem.with_suffix('.wav').exists():
                mux += ['-i', stem.with_suffix('.wav')]
            mux += ['-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                    stem.with_suffix('.mp4')]
            run(mux, stem.with_suffix('.mux.log'))
        previous = stem.with_suffix('.safetensors')


def metrics(rgb):
    y = rgb @ np.array([.2126, .7152, .0722])
    # Fixed center ROI is deliberately explicit: inspect contact sheets for motion.
    h, w = y.shape
    y = y[h//4:3*h//4, w//4:3*w//4]
    lap = 4*y[1:-1, 1:-1]-y[:-2, 1:-1]-y[2:, 1:-1]-y[1:-1, :-2]-y[1:-1, 2:]
    q50, q90, q99 = np.quantile(y, [.5, .9, .99])
    crop = rgb[h//4:3*h//4, w//4:3*w//4]
    # Broad nonneutral midtone selection, without a skin-hue requirement. Report
    # coverage so hair/background or changing pose cannot silently look like success.
    selected = (y > .15) & (y < .8) & (crop.max(-1)-crop.min(-1) > .04)
    linear = np.where(crop <= .04045, crop/12.92, ((crop+.055)/1.055)**2.4)
    xyz = linear @ np.array([[.4124564,.2126729,.0193339],
                            [.3575761,.7151522,.1191920], [.1804375,.0721750,.9503041]])
    xyz /= np.array([.95047, 1., 1.08883])
    f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz/(3*(6/29)**2)+4/29)
    lab_a, lab_b = 500*(f[...,0]-f[...,1]), 200*(f[...,1]-f[...,2])
    # Positive local contrast at a five-pixel radius: includes bright halos and
    # legitimate boundaries, so this is explicitly an edge-growth proxy.
    pad = np.pad(y, 5, mode='reflect')
    integral = np.pad(pad, ((1,0),(1,0))).cumsum(0).cumsum(1)
    blurred = (integral[11:,11:]-integral[:-11,11:]-integral[11:,:-11]+integral[:-11,:-11])/121
    edge = np.maximum(0, y-blurred)
    return dict(mean=float(y.mean()), p50=q50, p90=q90, p99=q99,
                highlight_fraction=float((y > .8).mean()), highlight_tail=q99-q50,
                texture_rms=float(np.sqrt(np.mean(lap**2))),
                skin_a=float(np.median(lab_a[selected])) if selected.any() else 0.,
                skin_b=float(np.median(lab_b[selected])) if selected.any() else 0.,
                skin_coverage=float(selected.mean()), edge_p99=float(np.quantile(edge,.99)),
                edge_fraction=float((edge > .08).mean()))


def analyze(a):
    probe = json.loads(subprocess.check_output([a.ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', str(a.video)]))['streams'][0]
    w, h = probe['width'], probe['height']
    command = [a.ffmpeg, '-v', 'error', '-i', str(a.video)]
    if a.max_frames:
        command += ['-frames:v', str(a.max_frames)]
    command += ['-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    rows, samples = [], []
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        while True:
            raw = process.stdout.read(w*h*3)
            if not raw:
                break
            if len(raw) != w*h*3:
                raise RuntimeError('Truncated FFmpeg frame')
            rgb = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            index = len(rows)
            rows.append(dict(frame=index, **metrics(rgb.astype(np.float32)/255)))
            if index % a.stride == 0:
                samples.append((index, Image.fromarray(rgb).resize((192, 192))))
        if process.wait():
            raise RuntimeError('FFmpeg failed')
    if not rows:
        raise ValueError('No video frames')
    reference = np.mean([[r['skin_a'],r['skin_b']] for r in rows[:22]], axis=0)
    for row in rows:
        row['skin_chroma_delta'] = float(np.linalg.norm(np.array([row['skin_a'],row['skin_b']])-reference))
    with a.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    sheet = Image.new('RGB', (192*6, 214*((len(samples)+5)//6)), '#222222')
    draw = ImageDraw.Draw(sheet)
    for j, (index, frame) in enumerate(samples):
        x, y = (j % 6)*192, (j//6)*214
        sheet.paste(frame, (x, y))
        draw.text((x+4, y+194), f'frame {index}', fill='white')
    sheet.save(a.output.with_suffix('.jpg'))
    summary = {part: {k: float(np.mean([r[k] for r in subset])) for k in rows[0] if k != 'frame'}
               for part, subset in [('first_22', rows[:22]), ('last_22', rows[-22:]),
                                    ('last_120', rows[-120:])]}
    summary['frames'] = len(rows)
    a.output.with_suffix('.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def fit(a):
    metadata, calibration = read(a.calibration)
    early_meta, _ = read(a.early)
    late_meta, _ = read(a.late)
    for key in ('width', 'height', 'fps', 'video_vae', 'transformer'):
        if early_meta.get(key) != late_meta.get(key):
            raise ValueError(f'Calibration clips disagree on {key}')
    def samples(path, tail):
        meta, tensors = read(path)
        if meta.get('slopfab_latents') != 'h3-av-v1':
            raise ValueError('Expected an H3 saved-latent archive')
        w, h = int(meta['width'])//32, int(meta['height'])//32
        rows = tensors['video_rows'].reshape(-1, h, w, 96)
        if len(rows) < 7:
            raise ValueError('Calibration clips need at least seven latent slices')
        if not np.isfinite(rows).all():
            raise ValueError('Non-finite latent sample')
        # Seven latent slices = 22 decoded frames; discard first boundary slice.
        rows = rows[-6:] if tail else rows[1:7]
        rows = rows.reshape(len(rows), -1, 96).astype(np.float64)
        return rows*np.repeat(calibration['latent_std'], 4) + np.repeat(calibration['latent_mean'], 4)
    early, late = samples(a.early, False), samples(a.late, True)
    direction = late.mean((0, 1))-early.mean((0, 1))
    basis = calibration['basis'].astype(np.float64)
    direction -= basis @ (basis.T @ direction)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError('No measurable color-orthogonal drift')
    direction /= norm
    projections = early.mean(1) @ direction
    reference = float(projections.mean())
    tolerance = max(float(2*projections.std()), .01)
    calibration['drift_direction'] = direction.astype(np.float32)
    calibration['drift_reference'] = np.array([reference, tolerance], np.float32)
    metadata['drift_early'] = str(a.early.resolve())
    metadata['drift_late'] = str(a.late.resolve())
    metadata['drift_method'] = 'color-orthogonal mean shift; whole-frame statistics; experimental'
    axis = calibration['anchor_lcs'][7] - calibration['anchor_lcs'][6]
    light = ((early-calibration['mean']) @ basis - calibration['anchor_lcs'][6]) @ axis / (axis @ axis)
    knee = float(np.quantile(light, .9))
    if not np.isfinite(knee) or not 0 <= knee <= 2:
        raise ValueError('Reference highlight knee lies outside supported [0,2] range')
    settings = dict(highlight_strength=.5, highlight_knee=knee, start_step=16)
    a.output.with_suffix('.settings.json').write_text(json.dumps(settings, indent=2))
    write(a.output, metadata, calibration)
    report = dict(raw_shift_norm=float(norm), reference=reference, tolerance=tolerance, highlight_knee=knee,
                  early_projection=projections.tolist(), late_projection=(late.mean(1) @ direction).tolist())
    a.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def fit_stability(a):
    metadata, c = read(a.calibration)
    meta, tensors = read(a.reference)
    if meta.get('slopfab_latents') != 'h3-av-v1':
        raise ValueError('Expected an H3 latent archive')
    h, w = int(meta['height'])//32, int(meta['width'])//32
    rows = tensors['video_rows'].reshape(-1,h,w,96).astype(np.float64)
    if len(rows) < 12 or min(h,w) < 3 or not np.isfinite(rows).all():
        raise ValueError('Reference needs >=12 latent slices, >=3x3 patches, and finite values')
    raw = rows*np.repeat(c['latent_std'],4) + np.repeat(c['latent_mean'],4)
    colors = (raw-c['mean']) @ c['basis'] - c['anchor_lcs'][6]
    axis = c['anchor_lcs'][7] - c['anchor_lcs'][6]
    scale = np.linalg.norm(axis)
    n = axis/scale
    arb = np.array([0.,1.,0.]) if abs(n[0]) > .9 else np.array([1.,0.,0.])
    u = arb - n*(arb@n)
    u /= np.linalg.norm(u)
    v = np.cross(n,u)
    coords = colors @ np.stack([n,u,v],axis=1) / scale
    color_ref, edge_ref = [], []
    for phase in range(5):
        # Exclude the first temporal group; keep phase identity from the archive.
        selected = coords[np.arange(len(coords)) >= 5][(np.arange(len(coords))[5:] % 5) == phase]
        color_ref.append(np.quantile(selected[...,1:],[.1,.5,.9],axis=(0,1,2)).T)
        light = selected[...,0]
        edge = light[:,1:-1,1:-1] - (light[:,:-2,1:-1]+light[:,2:,1:-1]+
                    light[:,1:-1,:-2]+light[:,1:-1,2:])/4
        edge_ref.append(max(0.,float(np.quantile(edge,.9))))
    c['color_reference'] = np.asarray(color_ref,np.float32)
    c['edge_reference'] = np.asarray(edge_ref,np.float32)
    metadata['stability_reference'] = str(a.reference.resolve())
    metadata['stability_method'] = 'phase-wise chroma quantiles and positive spatial lightness excess'
    write(a.output,metadata,c)
    settings = dict(color_stability=.75, edge_stability=.75, color_max_shift=.1)
    a.output.with_suffix('.settings.json').write_text(json.dumps(settings,indent=2))
    report = dict(color_reference=c['color_reference'].tolist(),edge_reference=edge_ref,
                  reference=str(a.reference),settings=settings)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


def compare(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def rows(path):
        with path.open(newline='') as f:
            return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    baseline, candidate = rows(a.baseline), rows(a.candidate)
    if not baseline or len(baseline) != len(candidate):
        raise ValueError('Paired comparison needs equal, nonempty frame sequences')
    paired = []
    for left, right in zip(baseline, candidate):
        if left.keys() != right.keys() or left['frame'] != right['frame']:
            raise ValueError('Mismatched metric columns or frame indices')
        if left['frame'] < a.start_frame:
            continue
        row = {'frame': int(left['frame'])}
        for key in left:
            if key != 'frame':
                row.update({f'baseline_{key}': left[key], f'candidate_{key}': right[key],
                            f'delta_{key}': right[key]-left[key]})
        paired.append(row)
    if not paired:
        raise ValueError('No frames in selected comparison range')
    with a.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=paired[0])
        writer.writeheader()
        writer.writerows(paired)
    keys = ['highlight_fraction', 'highlight_tail', 'mean', 'texture_rms']
    keys += [k for k in ['skin_chroma_delta','skin_coverage','edge_p99','edge_fraction'] if k in baseline[0]]
    fig, axes = plt.subplots(len(keys), 1, figsize=(10, 2*len(keys)), sharex=True)
    for ax, key in zip(axes, keys):
        for label in ['baseline', 'candidate']:
            ax.plot([r['frame'] for r in paired], [r[f'{label}_{key}'] for r in paired], label=label)
        ax.set_ylabel(key.replace('_', ' '))
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel('Decoded frame (24 fps fixture)')
    fig.suptitle('Paired continuation: appearance proxies, fixed center crop')
    fig.tight_layout()
    fig.savefig(a.output.with_suffix('.svg'))
    plt.close(fig)


def calibrate(a):
    dll = C.CDLL(str(a.dll.resolve()))
    dll.slopfab_request_create.restype = C.c_void_p
    dll.slopfab_request_set_model_path.argtypes = [C.c_void_p, C.c_int, C.c_char_p]
    dll.slopfab_lcs_calibrate.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
    dll.slopfab_request_destroy.argtypes = [C.c_void_p]
    dll.slopfab_last_error.restype = C.c_char_p
    request = dll.slopfab_request_create()
    try:
        for status in [dll.slopfab_request_set_model_path(request, 3, str(a.vae.resolve()).encode())]:
            if status:
                raise RuntimeError(dll.slopfab_last_error().decode())
        status = dll.slopfab_lcs_calibrate(request, str(a.output.resolve()).encode(), 64, 256, None, None)
        if status:
            raise RuntimeError(dll.slopfab_last_error().decode())
    finally:
        dll.slopfab_request_destroy(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('generate')
    p.add_argument('--exe', type=Path, default=Path('build/Release/slopfab.exe'))
    p.add_argument('--transformer', type=Path, required=True)
    p.add_argument('--vae', type=Path, default=Path('weights/vae/minimax_h3_video_vae_fp16.safetensors'))
    p.add_argument('--audio-vae', type=Path, default=Path('weights/vae/minimax_h3_audio_vae_fp32.safetensors'))
    p.add_argument('--embedding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--name', default='baseline')
    p.add_argument('--source', type=Path)
    p.add_argument('--windows', type=int, default=5)
    p.add_argument('--first-frames', type=int, default=73)
    p.add_argument('--new-frames', type=int, default=68)
    p.add_argument('--resolution', default='384x384')
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--seed', type=int, default=424242)
    p.add_argument('--lcs', type=Path)
    p.add_argument('--lcs-settings', default='{"drift_strength":0.5,"drift_max":1,"start_step":16}')
    p.add_argument('--lcs-settings-file', type=Path, help='JSON file; avoids shell quoting differences')
    p.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    p.set_defaults(func=generate)
    p = subs.add_parser('analyze')
    p.add_argument('video', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stride', type=int, default=24)
    p.add_argument('--max-frames', type=int, default=0, help='Analyze only this many decoded frames; 0 means all')
    p.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    p.add_argument('--ffprobe', default=shutil.which('ffprobe') or 'ffprobe')
    p.set_defaults(func=analyze)
    p = subs.add_parser('compare', help='Paired frame CSV and SVG curves; requires matplotlib')
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=0)
    p.set_defaults(func=compare)
    p = subs.add_parser('fit')
    for key in ('calibration', 'early', 'late', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.set_defaults(func=fit)
    p = subs.add_parser('fit-stability', help='Fit persistent appearance statistics using only the original clip')
    for key in ('calibration','reference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.set_defaults(func=fit_stability)
    p = subs.add_parser('calibrate')
    p.add_argument('--dll', type=Path, default=Path('build/Release/slopfab.dll'))
    p.add_argument('--vae', type=Path, default=Path('weights/vae/minimax_h3_video_vae_fp16.safetensors'))
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=calibrate)
    a = parser.parse_args()
    if a.command == 'generate' and (a.windows < 1 or a.steps < 2):
        parser.error('windows must be positive and steps at least two')
    if a.command == 'generate' and (not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.name)):
        parser.error('name may contain only letters, numbers, underscores and hyphens')
    if a.command == 'analyze' and a.stride < 1:
        parser.error('stride must be positive')
    if a.command == 'analyze' and a.max_frames < 0:
        parser.error('max-frames must be nonnegative')
    a.func(a)


if __name__ == '__main__':
    main()
