"""Embedded workflow integration tests. Python is a test harness, never a runtime dependency."""
import concurrent.futures
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

exe = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="slopfab-js-") as directory:
    root = Path(directory)
    config = root / "config.json"
    config.write_text('{"schema_version":1,"weight_paths":[]}', encoding="utf-8")

    def run(*args, code=0, stdin=None):
        result = subprocess.run([exe, "--agent", "--config", str(config), "--jobs-dir", str(root / "jobs"), *map(str,args)],
                                cwd=root, input=stdin, capture_output=True, text=True,
                                encoding="utf-8", timeout=20)
        assert result.returncode == code, (args,result.returncode,result.stdout[-6000:],result.stderr[-3000:])
        events = [json.loads(line) for line in result.stdout.splitlines()]
        assert len(events) == 1 and events[0].get("ok", True) == (code == 0), events
        return events[0]

    def js(source, *args, code=0):
        return run("run", "--eval", source, *args, code=code)

    assert js("export default args => ({args, n: 2n.toString()});", "--", "--agent", "--help", "--cuda-version", "13")["data"] == {
        "args":["--agent","--help","--cuda-version","13"], "n":"2"}
    assert run("run", "-", stdin="slopfab.result({stdin: true});")["data"] == {"stdin":True}
    assert js("console.log('a log'); export default () => 42;")["data"] == 42
    (root/"helper.js").write_text("export const answer = 41;",encoding="utf-8")
    (root/"workflow.js").write_text("""
      import sf, {fs, path} from 'slopfab';
      import {answer} from './helper.js';
      export default async args => {
        const imported = await import('./helper.js');
        await new Promise(resolve => setTimeout(resolve, 2));
        fs.mkdir('files ü');
        const file = path.join('files ü', 'value #.json');
        fs.writeJson(file, {answer: answer+1, unicode:'日本語🎬', args});
        const bytes = new Uint8Array([0,1,2,255,4]);
        fs.writeBytes('data.bin', bytes.subarray(1,4));
        const read = fs.readBytes('data.bin');
        fs.writeBytes(path.join('files ü', '日本.bin'), read);
        if (sf.sha256(path.join('files ü', '日本.bin')) !== sf.sha256('data.bin')) throw Error('Unicode SHA-256 path');
        fs.remove(path.join('files ü', '日本.bin'));
        const utf = new TextEncoder().encode('å日本🎬');
        if (new TextDecoder().decode(utf) !== 'å日本🎬') throw Error('UTF-8 roundtrip');
        fs.copy('data.bin', 'copied.bin'); fs.rename('copied.bin','renamed.bin'); fs.remove('renamed.bin');
        return {data:fs.readJson(file), bytes:[...read], digest:sf.sha256('data.bin'),
          url:path.fileUrl(file), stat:fs.stat('data.bin'), list:fs.list('files ü').length,
          main:import.meta.main, imported:imported.answer, moduleUrl:import.meta.url};
      };
    """,encoding="utf-8")
    data=run("run","workflow.js","--","x y")["data"]
    assert data["data"] == {"answer":42,"unicode":"日本語🎬","args":["x y"]}
    assert data["bytes"] == [1,2,255] and data["digest"] == hashlib.sha256(bytes([1,2,255])).hexdigest()
    assert "%20" in data["url"] and "%23" in data["url"] and data["url"].startswith("file://")
    assert data["stat"]["size"] == 3 and data["stat"]["isFile"] and data["main"] and data["imported"] == 41
    assert data["list"] == 1 and data["moduleUrl"].endswith("workflow.js")
    for source in ("throw Error('expected failure')", "export default () => Promise.reject(Error('rejected'))",
                   "Promise.reject(Error('unhandled'));", "await new Promise(() => {});", "export !!!"):
        assert js(source,code=6)["error"]["code"] == "SCRIPT_ERROR"
    assert js("while(true) {}", "--timeout-ms","30",code=6)["error"]["code"] == "SCRIPT_ERROR"
    assert "pending jobs" in js("await new Promise(() => {});",code=6)["error"]["message"]
    js("export default async () => { const p=Promise.reject(Error('handled')); try {await p} catch{} return true; };")
    assert js("export default () => new Promise(resolve => setTimeout(() => resolve(42), 1));")["data"] == 42
    js("const x=[]; while(true) x.push(new Uint8Array(1024*1024));", "--memory-mb","16",code=6)
    run("run","--memory-mb","1","--eval","0",code=2)

    # Two large streams are drained without pipe deadlock; no shell interprets metacharacters.
    (root/"child.js").write_text("""
      console.error('e'.repeat(150000));
      export default args => ({args, env:slopfab.env.get('SLOPFAB_JS_TEST'), cwd:slopfab.cwd(), large:'o'.repeat(150000)});
    """,encoding="utf-8")
    data=js("""
      export default () => {
        const sf=slopfab;
        const opts={env:{SLOPFAB_JS_TEST:'value ü'}, maxOutputBytes:1000000};
        const args=['semi;colon','dollar$(echo nope)','a "quote"','back\\\\slash',''];
        const a=sf.spawn([sf.executable,'--config',sf.config,'run','child.js','--',...args],opts);
        const b=sf.spawn([sf.executable,'--config',sf.config,'run','child.js','--','second'],opts);
        const first=a.wait(), second=b.wait();
        const cli=sf.cli(['system', 'version']);
        return {first:JSON.parse(first.stdout).data, stderr:first.stderr.length,
          second:JSON.parse(second.stdout).data.args, version:cli.result.version, done:a.poll().done};
      };
    """)["data"]
    assert data["first"]["args"] == ['semi;colon','dollar$(echo nope)','a "quote"','back\\slash','']
    assert data["first"]["env"] == "value ü" and Path(data["first"]["cwd"]) == root
    assert data["stderr"] >= 150000 and data["second"] == ["second"] and data["done"] and data["version"]
    assert js("export default () => slopfab.exec(['slopfab-nonexistent-executable']);",code=6)["error"]["code"] == "SCRIPT_ERROR"
    assert js("""export default () => slopfab.exec([slopfab.executable,'run','-'],
      {stdin:'export default () => 123;'}).stdout;""")["data"].find('123') >= 0
    assert js("""export default () => slopfab.exec([slopfab.executable,'run','--eval',"throw Error('child')"],
      {check:false}).exitCode;""")["data"] == 6
    js("slopfab.exec([slopfab.executable,'run','--eval','slopfab.sleep(5000)'],{timeoutMs:30});",code=6)
    js("slopfab.exec([slopfab.executable,'run','child.js'],{maxOutputBytes:100});",code=6)
    js("slopfab.exec([slopfab.executable,'run','child.js'],{stdoutFile:'out.txt',stderrFile:'err.txt',maxOutputBytes:1});")
    assert (root/"out.txt").stat().st_size > 150000 and (root/"err.txt").stat().st_size >= 150000
    # A queued worker survives the script that submitted it.
    submitted=js("""export default () => slopfab.cli(['jobs', 'submit','--request','-','--dry-run'], {
      stdin:JSON.stringify({schema_version:1,task:'image',model:'h3',prompt:'test',output:{path:'queued.png'}})
    }).result.job.job_id;""")["data"]
    try:
        assert run('jobs', 'wait',submitted,'--timeout','10')['job']['state']=='succeeded'
    finally:
        run('jobs', 'worker','--stop')

    payload=bytes(range(256))*1024
    class Server(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/redirect':
                self.send_response(302); self.send_header('Location','/binary'); self.end_headers(); return
            status=404 if self.path=='/missing' else 200
            body=payload if self.path=='/binary' else json.dumps({'ok':True,'header':self.headers.get('X-Test')}).encode()
            self.send_response(status); self.send_header('Content-Length',str(len(body))); self.end_headers()
            self.wfile.write(body)
        def log_message(self,*args): pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Server)
    thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        data=js(f"""export default () => {{
          const base={json.dumps(url)};
          const json=slopfab.http.json(base+'/json',{{headers:{{'X-Test':'works'}}}});
          const file=slopfab.http.download(base+'/redirect','download.bin');
          let failed=false; try {{slopfab.http.download(base+'/missing','download.bin',{{overwrite:true}})}} catch {{failed=true}}
          return {{json,bytes:file.bytes,digest:slopfab.sha256('download.bin'),failed,status:slopfab.http.request(base+'/missing').status}};
        }};""")["data"]
        assert data["json"] == {"ok":True,"header":"works"} and data["failed"] and data["status"] == 404
        assert data["bytes"] == len(payload) and data["digest"] == hashlib.sha256(payload).hexdigest()
        assert (root/"download.bin").read_bytes() == payload and not list(root.glob('*.part'))
        js(f"slopfab.http.download({json.dumps(url+'/binary')},'download.bin');",code=6)
        js(f"slopfab.http.request({json.dumps(url+'/binary')},{{maxBytes:10}});",code=6)
        # Exercise the shipped command with a local payload instead of large model weights.
        example = Path(__file__).resolve().parents[1]/'examples'/'mods'/'download'
        installed = root/'mods'/'download'
        shutil.copytree(example, installed)
        (installed/'catalog.json').write_text(json.dumps({'fixture':{'files':[
            {'id':'small','url':url+'/binary'}]}}),encoding='utf-8')
        downloaded = run('download','fixture','small','models ü/weights.bin')['data']
        assert downloaded['bytes']==len(payload) and downloaded['sha256']==hashlib.sha256(payload).hexdigest()
        shutil.rmtree(installed)
    finally:
        server.shutdown(); server.server_close(); thread.join()

    # Script commands are discoverable without running code; files are live.
    def mod(name,command,enabled=True,script="main.js"):
        folder=root/'mods'/name; folder.mkdir(parents=True,exist_ok=True)
        (folder/'MOD.md').write_text(f'---\nname: {name}\ndescription: Test workflow\ncommand: {command}\nscript: {script}\nenabled: {str(enabled).lower()}\n---\nUse this workflow.\n',encoding='utf-8')
        (folder/'main.js').write_text("slopfab.fs.writeText('ran.txt','yes'); export default args => ({args, dir:slopfab.scriptDirectory});",encoding='utf-8')
        return folder
    download=mod('model-download','download')
    assert run('mods','list')['mods'][0]['command']=='download'
    assert any(c['command']=='download' for c in run('help')['commands'])
    assert any(c['command']=='download' for c in run('system', 'capabilities')['data']['mod_commands'])
    run('help','download'); run('download','--help'); run('mods','load','model-download')
    assert not (root/'ran.txt').exists()
    assert run('download','minimaxh3')['data']['args']==['minimaxh3']
    assert Path(run('download','minimaxh3')['data']['dir'])==download
    mod('second-download','download')
    assert run('download','minimaxh3',code=3)['error']['code']=='MOD_COMMAND_AMBIGUOUS'
    assert any(e['code']=='MOD_COMMAND_AMBIGUOUS' for e in run('mods','list')['errors'])
    shutil.rmtree(root/'mods'/'second-download')
    mod('model-download','download',False)
    assert run('download',code=4)['error']['code']=='MOD_DISABLED'
    mod('model-download','version')
    assert any(e['code']=='INVALID_MOD' for e in run('mods','list')['errors'])
    mod('model-download','download',script='../../workflow.js')
    assert run('mods','load','model-download',code=2)['error']['code']=='INVALID_MOD'

    # Install and invoke the shipped example with no model download/network access.
    shutil.rmtree(root/'mods'/'model-download')
    example = Path(__file__).resolve().parents[1]/'examples'/'mods'/'download'
    shutil.copytree(example, root/'mods'/'download')
    catalog = run('download','minimaxh3')['data']
    assert catalog['model']=='minimaxh3' and len(catalog['files']) >= 4
    assert all(item['url'].startswith('https://huggingface.co/') for item in catalog['files'])
    assert run('download','--','--help',code=6)['error']['code']=='SCRIPT_ERROR'

    # The license mod prints full readable texts and keeps structured stdout.
    license_mod = example.parent / 'licenses'
    installed = root / 'mods' / 'licenses'
    shutil.copytree(license_mod, installed)
    repo = example.parents[2]
    licenses = [{'source': name, 'text': (repo / name).read_text(encoding='utf-8')}
                for name in ('LICENSE', 'third_party/whisper/LICENSE')]
    (installed / 'licenses.json').write_text(json.dumps(licenses), encoding='utf-8')
    for agent_args in ([], ['--agent']):
        result = subprocess.run([exe, '--config', str(config), *agent_args, 'licenses'],
                                cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=20)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['data']['licenses'] == [entry['source'] for entry in licenses]
        for entry in licenses:
            assert f"=== {entry['source']} ===\n\n{entry['text']}" in result.stderr
    assert run('licenses', 'unexpected', code=6)['error']['code'] == 'SCRIPT_ERROR'

    # Use a real FFmpeg when it is installed; no media/model fixture required.
    ffmpeg=Path(exe).parents[2]/'external'/'ffmpeg'/'bin'/'ffmpeg.exe'
    ffmpeg=str(ffmpeg) if ffmpeg.exists() else shutil.which('ffmpeg')
    if ffmpeg:
        data=js(f"export default () => slopfab.exec([{json.dumps(ffmpeg)},'-hide_banner','-f','lavfi','-i','color=c=red:s=16x16:d=0.1','-frames:v','1','-y','frame.png']);")["data"]
        assert data['exitCode']==0 and (root/'frame.png').stat().st_size > 0
    else:
        print('SKIP: FFmpeg executable not installed')
print('PASS: embedded modules, async jobs, binary files, processes, HTTP, CLI, FFmpeg and mod commands')
