"""Real ThorVG pixels, JavaScript timelines and optional FFmpeg composition; no GPU/weights."""
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
ffmpeg = repo/'external'/'ffmpeg'/'bin'/'ffmpeg.exe'
ffmpeg = str(ffmpeg) if os.name=='nt' and ffmpeg.exists() else shutil.which('ffmpeg')
ffprobe = str(Path(ffmpeg).with_name('ffprobe.exe' if os.name=='nt' else 'ffprobe')) if ffmpeg else None

def png(path):
    data=path.read_bytes(); assert data[:8]==b'\x89PNG\r\n\x1a\n'
    pos=8; compressed=b''
    while pos<len(data):
        size=struct.unpack('>I',data[pos:pos+4])[0]; kind=data[pos+4:pos+8]; body=data[pos+8:pos+8+size]; pos+=size+12
        if kind==b'IHDR':
            w,h,depth,color=struct.unpack('>IIBB',body[:10]); assert depth==8 and color==6
        if kind==b'IDAT': compressed+=body
    rows=zlib.decompress(compressed); pixels=bytearray()
    for y in range(h):
        row=rows[y*(w*4+1):(y+1)*(w*4+1)]; assert row[0]==0; pixels.extend(row[1:])
    return w,h,pixels

with tempfile.TemporaryDirectory(prefix='slopfab-motions-') as temp:
    root=Path(temp); (root/'config.json').write_text('{"schema_version":1,"weight_paths":[]}',encoding='utf-8')
    def run(*args, code=0):
        p=subprocess.run([exe,'--agent','--config',str(root/'config.json'),*map(str,args)],cwd=root,
            capture_output=True,text=True,encoding='utf-8',timeout=40)
        assert p.returncode==code,(args,p.returncode,p.stdout[-3000:],p.stderr[-3000:])
        events=[json.loads(line) for line in p.stdout.splitlines()]
        assert len(events)==1,events
        return events[0]
    def motion(command, output, *args, code=0):
        return run('motions',command,'motion.js','--out',output,*args,code=code)
    (root/'motion.js').write_text("""
      import {motions as m} from 'slopfab';
      export default {width:64,height:48,fps:4,duration:1,
        frame:({width,height,progress})=>m.svg(width,height,
          m.element('rect',{x:8+Math.round(progress*24),y:8,width:16,height:16,fill:'#ff0000','fill-opacity':0.5}))};
    """,encoding='utf-8')
    assert run('capabilities')['data']['motions']['cli_only']
    assert run('motions','--help')['type']=='help'
    (root/'ünicode').mkdir()
    result=motion('frame','ünicode/frame.png','--ffmpeg','missing-ffmpeg')['data']
    assert result['renderer']=='ThorVG 1.1.3' and result['frames']==1
    w,h,pixels=png(root/'ünicode/frame.png'); assert (w,h)==(64,48)
    def pixel(x,y): return list(pixels[(y*w+x)*4:(y*w+x+1)*4])
    assert pixel(0,0)==[0,0,0,0] and pixel(12,12)[:3]==[255,0,0] and 126<=pixel(12,12)[3]<=129
    motion('frame','frame.svg','--time','0.5')
    assert 'x="20"' in (root/'frame.svg').read_text(encoding='utf-8')
    previous=(root/'frame.svg').read_bytes()
    motion('frame','frame.svg',code=6)
    assert (root/'frame.svg').read_bytes()==previous
    motion('frame','frame.svg','--overwrite')
    assert 'x="8"' in (root/'frame.svg').read_text(encoding='utf-8')
    motion('frame','small.png','--width','32','--height','24')
    assert png(root/'small.png')[:2]==(32,24)
    motion('frame','bad.mp4',code=2)
    motion('render','bad.png',code=2)
    motion('frame','bad.png','--width','0',code=6)
    motion('frame','bad.png','--width','999999',code=6)
    motion('frame','bad.png','--time','-1',code=6)
    run('motions','frame','missing.js','--out','missing.png',code=3)
    (root/'broken.js').write_text("export default {frame:()=>'<not-svg/>'};",encoding='utf-8')
    run('motions','frame','broken.js','--out','broken.png',code=6)
    assert not (root/'broken.png').exists()
    font=Path('C:/Windows/Fonts/arial.ttf') if os.name=='nt' else Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if font.exists():
        (root/'text.js').write_text("""import {motions as m} from 'slopfab'; export default args=>{
          m.font('TestFont',args[0]); return {width:128,height:48, frame:()=>m.svg(128,48,
          m.text('Motion & text',{x:2,y:30,'font-family':'TestFont','font-size':22,fill:'#ffffff'}))};};""",encoding='utf-8')
        run('motions','frame','text.js','--out','text.png','--',font)
        assert sum(png(root/'text.png')[2][3::4])>10000
    # The same API works from ordinary workflow scripts, with awaited frame callbacks.
    run('run','--eval',"""import {motions as m} from 'slopfab'; export default async()=> {
      const a=await m.render({width:16,height:16,output:'api.png'},async()=>m.svg(16,16,m.element('circle',{cx:8,cy:8,r:5,fill:'blue'})));
      return a; };""")
    assert png(root/'api.png')[:2]==(16,16)
    motion('render','missing-encoder.mp4','--ffmpeg','slopfab-nonexistent-ffmpeg',code=6)
    assert not (root/'missing-encoder.mp4').exists() and not list(root.glob('.missing-encoder.*'))

    if ffmpeg and ffprobe and Path(ffprobe).exists():
        def ff(*args):
            p=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error',*map(str,args)],cwd=root,capture_output=True,timeout=40)
            assert p.returncode==0,p.stderr.decode(errors='replace'); return p.stdout
        def probe(file):
            p=subprocess.run([ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(root/file)],capture_output=True,timeout=20)
            assert p.returncode==0,p.stderr; return json.loads(p.stdout)
        motion('render','motion.mp4','--ffmpeg',ffmpeg)
        video=probe('motion.mp4')['streams'][0]
        assert (video['width'],video['height'],int(video['nb_frames']))==(64,48,4)
        rgb=ff('-i','motion.mp4','-f','rawvideo','-pix_fmt','rgb24','pipe:1')
        first=rgb[(12*64+12)*3:(12*64+12)*3+3]
        last=rgb[3*64*48*3+(12*64+36)*3:3*64*48*3+(12*64+36)*3+3]
        assert first[0]>100 and first[1]<20 and last[0]>100
        motion('render','alpha.mov','--ffmpeg',ffmpeg)
        rgba=ff('-i','alpha.mov','-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1')
        assert rgba[3]==0 and 120<=rgba[(12*64+12)*4+3]<=135
        motion('render','alpha.webm','--ffmpeg',ffmpeg)
        rgba=ff('-c:v','libvpx-vp9','-i','alpha.webm','-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1')
        assert rgba[3]<5 and 120<=rgba[(12*64+12)*4+3]<=135
        ff('-f','lavfi','-i','color=c=blue:s=64x48:r=4:d=2','-f','lavfi','-i','sine=frequency=440:duration=2',
           '-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','base.mp4')
        motion('render','overlay.mp4','--over','base.mp4','--ffmpeg',ffmpeg)
        info=probe('overlay.mp4')
        assert any(s['codec_type']=='audio' for s in info['streams'])
        assert 1.9<float(info['format']['duration'])<2.2
        motion('render','long-motion.mp4','--duration','3','--over','base.mp4','--ffmpeg',ffmpeg)
        assert 1.9<float(probe('long-motion.mp4')['format']['duration'])<2.2
        rgb=ff('-i','overlay.mp4','-f','rawvideo','-pix_fmt','rgb24','pipe:1')
        p=rgb[(12*64+12)*3:(12*64+12)*3+3]
        assert p[0]>100 and p[2]>100 and p[1]<20,p
        p=rgb[6*64*48*3+(12*64+36)*3:6*64*48*3+(12*64+36)*3+3]
        assert p[0]<20 and p[2]>220,p # Motion ends; the full base video continues.
        ff('-i','base.mp4','-an','-pix_fmt','yuv420p','base.y4m')
        ff('-i','base.mp4','-vn','audio.wav')
        motion('render','native-overlay.mp4','--over','base.y4m','--audio','audio.wav','--ffmpeg',ffmpeg)
        assert any(s['codec_type']=='audio' for s in probe('native-overlay.mp4')['streams'])
        ff('-f','lavfi','-i','sine=frequency=440:duration=0.3','short.wav')
        motion('render','padded-audio.mp4','--over','base.y4m','--audio','short.wav','--ffmpeg',ffmpeg)
        assert 1.9<float(probe('padded-audio.mp4')['format']['duration'])<2.2
        motion('render','base.mp4','--over','base.mp4','--overwrite','--ffmpeg',ffmpeg,code=6)
        (root/'fail.js').write_text("""export default {width:64,height:48,fps:4,duration:1,frame:({frame})=>{
          if(frame===2)throw Error('fixture'); return '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="48"><rect width="64" height="48" fill="red"/></svg>';}};""",encoding='utf-8')
        run('motions','render','fail.js','--out','failed.mp4','--ffmpeg',ffmpeg,code=6)
        assert not (root/'failed.mp4').exists() and not list(root.glob('.failed.*'))
        (root/'loop.js').write_text('export default {frame:()=>{while(true){}}};',encoding='utf-8')
        run('motions','render','loop.js','--out','loop.mp4','--timeout-ms','100','--ffmpeg',ffmpeg,code=6)
        assert not (root/'loop.mp4').exists() and not list(root.glob('.loop.*'))
    else:
        print('SKIP: video export/composition (FFmpeg and ffprobe not available)')
print('PASS: ThorVG pixels, alpha, fonts, timeline, exports, audio overlays and cleanup')
