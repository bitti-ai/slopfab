"""Frame selection, RGB buffers, downscaled grids and compressed-video extraction."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

exe = str(Path(sys.argv[1]).resolve())
repo = Path(__file__).resolve().parents[1]
ffmpeg = shutil.which('ffmpeg') or str(repo/'external/ffmpeg/bin/ffmpeg.exe')
ffprobe = shutil.which('ffprobe') or str(repo/'external/ffmpeg/bin/ffprobe.exe')

def png(file):
    data = file.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    w, h, depth, color = struct.unpack('>IIBB', data[16:26])
    assert (depth, color) == (8, 2)
    chunks, offset = [], 8
    while offset < len(data):
        size = int.from_bytes(data[offset:offset+4], 'big')
        if data[offset+4:offset+8] == b'IDAT':
            chunks.append(data[offset+8:offset+8+size])
        offset += size + 12
    raw = zlib.decompress(b''.join(chunks))
    assert all(raw[y*(w*3+1)] == 0 for y in range(h))
    return w, h, b''.join(raw[y*(w*3+1)+1:(y+1)*(w*3+1)] for y in range(h))

with tempfile.TemporaryDirectory(prefix='slopfab-frames-') as temp:
    root = Path(temp)
    def run(*args, code=0):
        p = subprocess.run([exe, '--agent', '--config', str(root/'config.json'), *map(str,args)],
                           cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=40)
        assert p.returncode == code, (args,p.returncode,p.stdout[-3000:],p.stderr[-3000:])
        events = [json.loads(line) for line in p.stdout.splitlines()]
        assert len(events) == 1, events
        return events[0]
    def capture(*args, **kw): return run('video', 'frames', 'input.y4m', '--out', 'out.png', *args, **kw)
    # Distinct grayscale frames; rational FPS, frame headers with tags, unicode file paths.
    with (root/'input.y4m').open('wb') as file:
        file.write(b'YUV4MPEG2 W8 H4 F4:1 Ip A1:1 C420jpeg\n')
        for y in (16, 71, 126, 180, 235):
            file.write(b'FRAME Xsample=1\n' + bytes([y])*32 + bytes([128])*16)
    assert run('system', 'capabilities')['data']['video_frames']['generation_ram_capture']
    assert run('video', 'frames', '--help')['type'] == 'help'
    assert run('video', 'frames', code=2)['type'] == 'error', 'old root command is still exposed'
    result = capture('--ffmpeg', 'missing-tool')['data']
    assert result['cells'] == [{'frame':0,'time':0,'x':0,'y':0}]
    assert png(root/'out.png') == (8,4,bytes(8*4*3))
    original = (root/'out.png').read_bytes()
    capture('--frame',4,code=6)
    assert (root/'out.png').read_bytes() == original
    capture('--frame',4,'--overwrite')
    assert png(root/'out.png')[2] == bytes([255])*(8*4*3)
    capture('--time',.6,'--width',4,'--overwrite')
    w,h,pixels = png(root/'out.png')
    assert (w,h)==(4,2) and 127 <= pixels[0] <= 129
    data = capture('--grid','3x2','--width',4,'--overwrite')['data']
    assert (data['width'],data['height'])==(12,4)
    assert [c['frame'] for c in data['cells']]==[0,1,2,2,3,4]
    w,h,pixels = png(root/'out.png')
    assert pixels[0] == 0 and pixels[((h-1)*w+w-1)*3] == 255
    data = capture('--grid','2x1','--start',.25,'--end',.75,'--overwrite')['data']
    assert [c['frame'] for c in data['cells']]==[1,3]
    capture('--frame',0,'--width',100,'--overwrite')
    assert png(root/'out.png')[:2]==(8,4), 'upscaled a small input'
    for args in [('--frame',5),('--frame',-1),('--frame',1.5),('--time',2),('--time',-1),
                 ('--grid','0x3'),('--grid','32x32'),('--grid','1x1','--frame',0),
                 ('--frame',0,'--time',0),('--start',0),('--width',0),('--unknown',1)]:
        capture(*args,'--overwrite',code=2)
    (root/'empty.y4m').write_bytes(b'YUV4MPEG2 W8 H4 F4:1 Ip C420jpeg\n')
    run('video','frames','empty.y4m','--out','empty.png',code=2)
    (root/'truncated.y4m').write_bytes((root/'input.y4m').read_bytes()[:-3])
    run('video','frames','truncated.y4m','--out','truncated.png',code=2)
    run('video','frames','missing.y4m','--out','missing.png',code=3)
    run('video','frames','input.y4m','--out','input.y4m','--overwrite',code=2)
    # One-frame grids duplicate the available frame, with valid zero timestamps.
    (root/'one.y4m').write_bytes(b'YUV4MPEG2 W2 H2 F30000:1001 Ip C420\nFRAME\n'+bytes([235])*4+bytes([128])*2)
    data = run('video','frames','one.y4m','--grid','2x2','--out','one.png')['data']
    assert [c['frame'] for c in data['cells']]==[0]*4
    (root/'réview').mkdir()
    data = run('video','frames','input.y4m','--frame',1,'--out','réview/frame.png')['data']
    assert Path(data['path']).is_file()
    # Typed-array subviews must honor offsets; downscaling must average, not discard pixels.
    data = run('run','--eval',"""import {frames} from 'slopfab'; export default ()=>{
      const storage=new Uint8Array(8+2*2*3*3+5); const rgb=storage.subarray(8,storage.length-5);
      for(let frame=0;frame<3;frame++) for(let p=0;p<4;p++) rgb[frame*12+p*3+frame]=255;
      return frames.fromRGB(rgb,{sourceWidth:2,sourceHeight:2,count:3,fps:2,
        columns:3,rows:1,width:1,output:'rgb.png'});
    };""")['data']
    assert data['source']=='memory:rgb24' and [c['time'] for c in data['cells']]==[0,.5,1]
    assert png(root/'rgb.png')==(3,1,bytes([255,0,0,0,255,0,0,0,255]))
    run('run','--eval',"""export default ()=>slopfab.frames.fromRGB(
      new Uint8Array([0,0,0,255,255,255,0,0,0,255,255,255]),
      {sourceWidth:2,sourceHeight:2,output:'average.png',width:1});""")
    assert png(root/'average.png')==(1,1,bytes([128]*3))
    run('run','--eval',"""export default ()=>slopfab.frames.extract({input:'input.y4m',
      output:'script.png',frame:3,width:2});""")
    assert png(root/'script.png')[:2]==(2,1)
    run('run','--eval',"""slopfab.frames.fromRGB(new Uint8Array(2),
      {sourceWidth:2,sourceHeight:2,output:'bad.png'});""",code=6)
    run('run','--eval',"""slopfab.frames.fromRGB(new Float32Array(12),
      {sourceWidth:2,sourceHeight:2,output:'bad.png'});""",code=6)
    # Invalid review options must fail before generation/model loading.
    run('video','frames','generate','--out','bad.jpg','--','--prompt','test',code=2)
    run('video','frames','generate','--out','bad.png','--grid','0x0','--','--prompt','test',code=2)
    # Exercise the scripting wrapper and its argument delimiter without loading weights.
    data = run('run','--eval',"""export default ()=>slopfab.frames.generate(
      ['--prompt','--agent'],{output:'bad.jpg'},{check:false});""")['data']
    assert data['exitCode']==2 and data['result']['type']=='error'
    assert not list(root.rglob('*.slopfab-reserved')) and not list(root.rglob('*.tmp'))

    if Path(ffmpeg).exists() and Path(ffprobe).exists():
        def ff(*args):
            p=subprocess.run([ffmpeg,'-v','error',*map(str,args)],cwd=root,capture_output=True,timeout=30)
            assert p.returncode==0,p.stderr
        ff('-i','input.y4m','-c:v','ffv1','lossless.mkv')
        tools=['--ffmpeg',ffmpeg,'--ffprobe',ffprobe]
        data=run('video','frames','lossless.mkv','--out','compressed.png','--grid','3x1','--width',4,*tools)['data']
        assert [c['frame'] for c in data['cells']]==[0,2,4]
        assert png(root/'compressed.png')[:2]==(12,2)
        # Nonuniform timestamps, tested by time and index independently.
        ff('-i','input.y4m','-vf',r'select=eq(n\,0)+eq(n\,1)+eq(n\,4)','-vsync','0','-c:v','ffv1','vfr.mkv')
        data=run('video','frames','vfr.mkv','--out','vfr.png','--time',.7,*tools)['data']
        assert data['cells'][0]['frame']==2 and abs(data['cells'][0]['time']-1)<.001
        assert png(root/'vfr.png')[2][0]>=254
        data=run('video','frames','vfr.mkv','--out','index.png','--frame',1,*tools)['data']
        assert abs(data['cells'][0]['time']-.25)<.001
        assert 62<=png(root/'index.png')[2][0]<=65
        # Decoder launch failure leaves neither an output nor reservations behind.
        run('video','frames','lossless.mkv','--out','failure.png','--ffprobe',ffprobe,
            '--ffmpeg','slopfab-no-such-ffmpeg',code=6)
        assert not (root/'failure.png').exists()
        (root/'failure.png').write_bytes(b'original output')
        run('video','frames','lossless.mkv','--out','failure.png','--overwrite','--ffprobe',ffprobe,
            '--ffmpeg','slopfab-no-such-ffmpeg',code=6)
        assert (root/'failure.png').read_bytes()==b'original output'
        assert not list(root.rglob('*.slopfab-reserved')) and not list(root.rglob('*.tmp'))
    else:
        print('FFmpeg unavailable; compressed-video cases skipped')
print('video frames CLI tests passed')
