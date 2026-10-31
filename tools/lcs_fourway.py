"""Export four aligned drift-test videos with labels, preserving each full frame."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess


def probe(path, ffprobe):
    return json.loads(subprocess.check_output([
        ffprobe, '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=width,height,r_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(path)]))['streams'][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'lcs', 'anchored', 'anchored-lcs', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frames', type=int, default=2880)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--ffprobe', default='ffprobe')
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error('frames must be positive')
    if args.output.exists():
        parser.error('output already exists')
    sources = [args.baseline, args.lcs, args.anchored, args.anchored_lcs]
    info = [probe(path, args.ffprobe) for path in sources]
    width, height = info[0]['width'], info[0]['height']
    fps = Fraction(info[0]['r_frame_rate'])
    for path, stream in zip(sources, info):
        if (stream['width'], stream['height'], Fraction(stream['r_frame_rate'])) != (width, height, fps):
            raise ValueError(f'Mismatched video geometry or frame rate: {path}')
        if int(stream['nb_read_frames']) < args.frames:
            raise ValueError(f'Input too short: {path}')
    font = Path('C:/Windows/Fonts/arial.ttf')
    font_arg = "fontfile='C\\:/Windows/Fonts/arial.ttf':" if font.exists() else ''
    labels = ['Baseline', 'LCS (color + edge)', 'Keystone + Capstone + Lock', 'Anchors + LCS (color + edge)']
    filters = [f"[{i}:v]setpts=PTS-STARTPTS,pad=iw:ih+32:0:32:black,"
               f"drawtext={font_arg}text='{label}':x=10:y=7:fontsize=18:fontcolor=white[v{i}]"
               for i, label in enumerate(labels)]
    filters += ['[v0][v1]hstack=inputs=2[top]', '[v2][v3]hstack=inputs=2[bottom]',
                '[top][bottom]vstack=inputs=2[v]']
    command = [args.ffmpeg, '-v', 'error', '-n']
    for path in sources:
        command += ['-i', str(path)]
    command += ['-filter_complex', ';'.join(filters), '-map', '[v]', '-frames:v', str(args.frames),
                '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(sources=[str(path.resolve()) for path in sources], labels=labels,
                    input_probes=info, frames=args.frames, fps=str(fps), command=command)
    args.output.with_suffix('.command.json').write_text(json.dumps(manifest, indent=2))
    subprocess.run(command, check=True)
    result = probe(args.output, args.ffprobe)
    if (result['width'],result['height'],int(result['nb_read_frames']),Fraction(result['r_frame_rate'])) != (2*width,2*(height+32),args.frames,fps):
        raise ValueError('Export geometry/frame validation failed')
    if abs(float(result['duration']) - args.frames/float(fps)) > 1e-5:
        raise ValueError('Export duration validation failed')
    args.output.with_suffix('.probe.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
