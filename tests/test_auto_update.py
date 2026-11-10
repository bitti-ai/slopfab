"""Daily update scheduling and detached installation, without network access."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

fixture, cli = (str(Path(arg).resolve()) for arg in sys.argv[1:3])
enabled = os.environ.copy()
enabled.pop('SLOPFAB_NO_AUTO_UPDATE', None)
disabled = dict(enabled, SLOPFAB_NO_AUTO_UPDATE='1')


def run(exe, *args, env=disabled, ok=True, cwd=None):
    result = subprocess.run([exe, *map(str, args)], cwd=cwd, env=env,
                            capture_output=True, text=True, encoding='utf-8', timeout=5)
    assert (result.returncode == 0) == ok, (args, result.stdout, result.stderr)
    return result


def wait_for(predicate):
    deadline = time.monotonic() + 10
    while not predicate():
        assert time.monotonic() < deadline, 'background update did not finish'
        time.sleep(.02)


with tempfile.TemporaryDirectory(prefix='slopfab-auto-update-') as directory:
    root = Path(directory)
    config = root / 'settings with spaces' / 'config.json'
    config.parent.mkdir()
    original = {'schema_version': 1, 'weight_paths': [], 'presets': {'keep': {'task': 'image'}}}
    config.write_text(json.dumps(original))

    # Claim once across processes, preserving other independently written settings.
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: run(fixture, 'claim', config, '2026-11-09').stdout, range(12)))
    assert claims.count('claimed') == 1 and claims.count('skipped') == 11, claims
    assert json.loads(config.read_text())['presets'] == original['presets']
    assert run(fixture, 'claim', config, '2026-11-10').stdout == 'claimed'
    assert run(fixture, 'claim', config, '2026-11-10').stdout == 'skipped'
    # Clock corrections should not suppress updates indefinitely.
    assert run(fixture, 'claim', config, '2026-11-08').stdout == 'claimed'
    for invalid in ('2026-02-29', '1900-02-29', '2026-13-01', '2026-11-31', '2026-1-1', ''):
        before = config.read_bytes()
        run(fixture, 'claim', config, invalid, ok=False)
        assert config.read_bytes() == before
    assert run(fixture, 'claim', config, '2024-02-29').stdout == 'claimed'

    def settings(*args, ok=True):
        return json.loads(run(cli, '--config', config, '--agent', 'settings', *args,
                              cwd=root, ok=ok).stdout)

    settings('set', 'auto_update', 'false')
    before = config.read_bytes()
    assert run(fixture, 'claim', config, '2026-11-11').stdout == 'skipped'
    assert config.read_bytes() == before
    # Manual checks still record their date with automatic updates disabled.
    run(fixture, 'record', config, '2026-11-11')
    assert settings('get', 'last_update_check')['value'] == '2026-11-11'
    for key, value in [('auto_update', '"yes"'), ('auto_update', '1'),
                       ('auto_update.enabled', 'true'), ('last_update_check', '"2026-11-11"')]:
        before = config.read_bytes()
        settings('set', key, value, ok=False)
        assert config.read_bytes() == before
    settings('remove', 'auto_update')
    assert run(fixture, 'claim', config, '2026-11-12').stdout == 'claimed'

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(settings, 'set', f'presets.writer{i}', '{"task":"image"}') for i in range(5)]
        futures.append(pool.submit(run, fixture, 'record', config, '2026-11-13'))
        for future in futures:
            future.result()
    state = settings('get')['settings']
    assert len(state['presets']) == 6 and state['last_update_check'] == '2026-11-13'

    # Exercise the real detached worker, substituting only the network installer.
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    for fail in (False, True):
        folder = root / ('offline worker' if fail else 'successful worker')
        folder.mkdir()
        selected = folder / 'config.json'
        selected.write_text(json.dumps(original))
        if fail:
            (folder / 'fail').touch()
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:
                foreground = list(pool.map(lambda _: run(fixture, 'launch', selected, env=enabled), range(4)))
            assert all(p.stdout == 'foreground result\n' and not p.stderr for p in foreground)
            wait_for(lambda: (folder / 'attempts').exists())
            assert not (folder / 'finished').exists(), 'foreground waited for background work'
            (folder / 'release').touch()
            wait_for(lambda: (folder / 'finished').exists())
            if fail:
                wait_for(lambda: 'fixture offline' in (folder / 'auto-update.log').read_text())
            else:
                assert (folder / 'installed').read_text() == 'new release'
            assert json.loads(selected.read_text())['last_update_check'] == today
            run(fixture, 'launch', selected, env=enabled)
            assert (folder / 'attempts').read_text().splitlines() == ['attempt']
        finally:
            (folder / 'release').touch()
        time.sleep(.1)  # Let the detached process close its log before Windows cleanup.

    # The real CLI must keep one JSON response and avoid a worker when today's
    # date is recorded or the global setting is disabled.
    run(fixture, 'record', config, today)
    response = run(cli, '--config', config, '--agent', 'system', 'version', env=enabled, cwd=root)
    assert json.loads(response.stdout)['ok'] and not response.stderr
    settings('set', 'auto_update', 'false')
    run(fixture, 'record', config, '2000-01-01')
    response = run(cli, '--config', config, '--agent', 'system', 'version', env=enabled, cwd=root)
    assert json.loads(response.stdout)['ok']
    assert settings('get', 'last_update_check')['value'] == '2000-01-01'
    assert not (config.parent / 'auto-update.log').exists()

print('PASS: daily update claims, concurrency, settings, detached installation and offline failures')
