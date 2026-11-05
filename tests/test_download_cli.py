"""Native multipart/resume integration tests against a fault-injecting HTTP server."""
import http.server
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

exe = str(Path(sys.argv[1]).resolve())
payload = bytes(range(256)) * 16384
guard = threading.Lock()
requests = []
active = peak = 0
faults = set()
mode = "normal"
etag = '"fixture-v1"'


class Server(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global active, peak
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/file')
            self.end_headers()
            return
        with guard:
            current_mode, current_etag = mode, etag
            range_header = self.headers.get('Range')
            requests.append((range_header, self.headers.get('If-Range')))
        match = re.fullmatch(r'bytes=(\d+)-(\d+)', range_header or '')
        ranged = bool(match) and current_mode not in ('no-range', 'empty')
        first, last = map(int, match.groups()) if ranged else (0, len(payload) - 1)
        probe = ranged and first == last == 0
        if current_mode == 'ignore-parts' and not probe:
            ranged, first, last = False, 0, len(payload) - 1
        body = payload[first:last + 1] if current_mode != 'empty' else b''
        self.send_response(206 if ranged else 200)
        self.send_header('Content-Length', str(len(body)))
        if current_mode != 'no-etag':
            self.send_header('ETag', '"wrong-version"' if current_mode == 'bad-etag' and not probe
                             else ('W/' if current_mode == 'weak-etag' else '') + current_etag)
        if ranged:
            reported_first = first + 1 if current_mode == 'bad-range' and not probe else first
            self.send_header('Content-Range', f'bytes {reported_first}-{last}/{len(payload)}')
        self.end_headers()
        with guard:
            active += 1
            peak = max(peak, active)
            fail = current_mode == 'drop' and not probe
            if current_mode == 'drop-once' and not probe and first not in faults:
                faults.add(first)
                fail = True
        try:
            if fail:
                self.wfile.write(body[:65536])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            # Give other connections time to overlap and make cancellation observable.
            for offset in range(0, len(body), 32768):
                self.wfile.write(body[offset:offset + 32768])
                self.wfile.flush()
                if not probe:
                    time.sleep(0.02 if current_mode == 'slow' else 0.001)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with guard:
                active -= 1

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Server)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = f'http://127.0.0.1:{server.server_port}'
try:
    with tempfile.TemporaryDirectory(prefix='slopfab-download-') as temporary:
        root = Path(temporary)
        config = root / 'config.json'
        config.write_text('{"schema_version":1,"weight_paths":[]}')

        def command(name, *, endpoint='/file', options=None, script_options=()):
            opts = {'resume': True, 'connections': 4, 'retries': 0, 'timeoutMs': 10000}
            opts.update(options or {})
            source = f"export default () => slopfab.http.download({json.dumps(url + endpoint)}, {json.dumps(name)}, {json.dumps(opts)});"
            return [exe, '--config', str(config), '--agent', 'run', '--eval', source, *script_options]

        def run(name, *, code=0, **kwargs):
            result = subprocess.run(command(name, **kwargs), cwd=root, capture_output=True,
                                    text=True, encoding='utf-8', timeout=30)
            assert result.returncode == code, (result.returncode, result.stdout, result.stderr)
            event = json.loads(result.stdout)
            return event['data'] if code == 0 else event['error']

        def valid(name):
            assert (root / name).read_bytes() == payload
            assert not (root / (name + '.download')).exists()

        # Real concurrent connections, redirects, byte-exact output and clean state.
        result = run('parallel ü.bin', endpoint='/redirect')
        valid('parallel ü.bin')
        assert result['connections'] == 4 and result['resumedBytes'] == 0
        assert result['bytes'] == len(payload) and peak >= 2, (result, peak)
        assert all(tag == etag for rng, tag in requests if rng not in (None, 'bytes=0-0'))
        assert 'exists' in run('parallel ü.bin', code=6)['message']

        # Keep partial bytes across a failed process; request only each missing suffix.
        mode = 'drop'
        run('resume.bin', code=6)
        assert not (root / 'resume.bin').exists()
        state = root / 'resume.bin.download'
        sizes = {int(p.stem): p.stat().st_size for p in state.glob('*.part')}
        assert sizes and sum(sizes.values()) > 0
        with guard:
            requests.clear()
        mode = 'normal'
        result = run('resume.bin')
        valid('resume.bin')
        assert result['resumedBytes'] == sum(sizes.values()), (result, sizes)
        for i, have in sizes.items():
            assert (f'bytes={i * (len(payload) // 4) + have}-{(i + 1) * (len(payload) // 4) - 1}', etag) in requests

        # A new representation invalidates every old part, even at the same size.
        mode = 'drop'
        run('changed.bin', code=6)
        payload = bytes(reversed(range(256))) * 16384
        etag = '"fixture-v2"'
        mode = 'normal'
        result = run('changed.bin')
        valid('changed.bin')
        assert result['resumedBytes'] == 0

        # Short reads are retried from the committed partial byte count.
        mode = 'drop-once'
        # Fail each original part once, but permit retries starting at its suffix.
        faults.update(i * (len(payload) // 4) + 65536 for i in range(4))
        run('retry.bin', options={'retries': 1})
        valid('retry.bin')

        # A misleading Content-Range must never publish or append unvalidated bytes.
        for mode in ('bad-range', 'bad-etag'):
            name = mode + '.bin'
            run(name, code=6)
            assert not (root / name).exists()
            assert not any(p.stat().st_size for p in (root / (name + '.download')).glob('*.part'))
            mode = 'normal'
            run(name)
            valid(name)

        # No strong validator, ignored ranges (initial or mid-transfer), and empty files.
        for mode in ('no-range', 'no-etag', 'weak-etag', 'ignore-parts'):
            result = run(mode + '.bin')
            valid(mode + '.bin')
            assert result['connections'] == 1 and result['resumedBytes'] == 0
        mode = 'empty'
        result = run('empty.bin')
        assert result['bytes'] == 0 and (root / 'empty.bin').read_bytes() == b''

        # Runtime cancellation retains resumable data and preserves an old destination.
        mode = 'slow'
        (root / 'overwrite.bin').write_bytes(b'old file')
        run('overwrite.bin', code=6, options={'overwrite': True}, script_options=('--timeout-ms', '180'))
        assert (root / 'overwrite.bin').read_bytes() == b'old file'
        assert sum(p.stat().st_size for p in (root / 'overwrite.bin.download').glob('*.part')) > 0
        mode = 'normal'
        result = run('overwrite.bin', options={'overwrite': True})
        valid('overwrite.bin')
        assert result['resumedBytes'] > 0

        # Abrupt process termination releases the lock and resumes from on-disk bytes.
        mode = 'slow'
        killed = subprocess.Popen(command('killed.bin'), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while sum(p.stat().st_size for p in (root / 'killed.bin.download').glob('*.part')) < 131072:
                assert killed.poll() is None and time.monotonic() < deadline
                time.sleep(0.01)
        finally:
            killed.kill()
            killed.communicate(timeout=10)
        assert not (root / 'killed.bin').exists()
        mode = 'normal'
        result = run('killed.bin')
        valid('killed.bin')
        assert result['resumedBytes'] > 0

        # Explicitly disabling resume removes partial state on failure.
        mode = 'drop'
        run('discard.bin', code=6, options={'resume': False})
        assert not (root / 'discard.bin.download').exists()
        for options in ({'connections': 0}, {'connections': 17}, {'retries': 11},
                        {'headers': {'Range': 'bytes=0-1'}}, {'method': 'POST'}):
            run('invalid.bin', code=6, options=options)

        # Two processes cannot write the same destination/state concurrently.
        mode = 'slow'
        first = subprocess.Popen(command('locked.bin'), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            while not (root / 'locked.bin.download/metadata.json').exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert 'already in use' in run('locked.bin', code=6)['message']
            out, err = first.communicate(timeout=20)
            assert first.returncode == 0, (out, err)
            valid('locked.bin')
        finally:
            if first.poll() is None:
                first.kill()
                first.wait()

        # Download mod opts in to the native multipart implementation.
        mode = 'normal'
        example = Path(__file__).resolve().parents[1] / 'examples/mods/download'
        installed = root / 'mods/download'
        shutil.copytree(example, installed)
        (installed / 'catalog.json').write_text(json.dumps({'fixture': {'files': [{'id': 'file', 'url': url + '/file'}]}}))
        result = subprocess.run([exe, '--config', str(config), '--agent', 'download', 'fixture', 'file', 'mod.bin'],
                                cwd=root, capture_output=True, text=True, encoding='utf-8', timeout=30)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert json.loads(result.stdout)['data']['connections'] == 4
        valid('mod.bin')
finally:
    server.shutdown()
    server.server_close()
    thread.join()

print('PASS: concurrent ranges, redirects, durable resume, retries, validators, fallback, cancellation, locking and download mod')
