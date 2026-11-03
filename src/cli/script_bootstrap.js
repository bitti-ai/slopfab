(() => {
  const host = (op, ...args) => __slopfabHost(op, JSON.stringify(args));
  const checked = (result, options) => {
    if (options.check !== false && result.exitCode !== 0) {
      const error = new Error(`Process exited with code ${result.exitCode}: ${(result.stderr || result.stdout).slice(-2000)}`);
      error.result = result;
      throw error;
    }
    return result;
  };
  const spawn = (argv, options = {}) => {
    const {id, pid} = host('spawn', argv, options);
    return Object.freeze({id, pid,
      poll: () => host('poll', id),
      wait: () => checked(host('wait', id), options),
      kill: () => host('kill', id)
    });
  };
  const exec = (argv, options = {}) => spawn(argv, options).wait();
  const fs = {
    readText: (path, options = {}) => host('readText', path, options),
    writeText: (path, text, options = {}) => host('writeText', path, String(text), options),
    readBytes: (path, options = {}) => __slopfabReadBytes(path, options.maxBytes ?? 67108864),
    writeBytes: (path, data) => __slopfabWriteBytes(path, data),
    exists: path => host('exists', path),
    stat: path => host('stat', path),
    list: path => host('list', path),
    mkdir: path => host('mkdir', path),
    copy: (from, to, options = {}) => host('copy', from, to, options),
    rename: (from, to) => host('rename', from, to),
    remove: (path, options = {}) => host('remove', path, options),
    readJson: path => JSON.parse(fs.readText(path)),
    writeJson: (path, value) => fs.writeText(path, JSON.stringify(value, null, 2) + '\n')
  };
  const path = Object.fromEntries(['join','resolve','dirname','basename','extname','fileUrl']
    .map(name => [name, (...parts) => host(`path.${name}`, ...parts)]));
  const cli = (args, options = {}) => {
    const result = exec([__slopfabInfo.executable, '--agent', '--config', __slopfabInfo.config,
      '--jobs-dir', __slopfabInfo.jobs, ...args], options);
    const events = result.stdout.split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line));
    return {...result, events, result: events.at(-1) ?? null};
  };
  const http = {
    request: (url, options = {}) => host('http', url, options, ''),
    download: (url, destination, options = {}) => host('http', url, options, destination),
    json: (url, options = {}) => {
      const response = http.request(url, options);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return JSON.parse(response.body);
    }
  };
  const escapeXML = value => String(value).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
  const element = (name, attributes = {}, children = '') => {
    if (!/^[A-Za-z][\w:-]*$/.test(name)) throw new TypeError('Invalid SVG element name');
    const attrs = Object.entries(attributes).map(([key, value]) => {
      if (!/^[A-Za-z_][\w:.-]*$/.test(key)) throw new TypeError('Invalid SVG attribute name');
      return ` ${key}="${escapeXML(value)}"`;
    }).join('');
    return `<${name}${attrs}>${Array.isArray(children) ? children.join('') : children}</${name}>`;
  };
  const motions = Object.freeze({
    escape: escapeXML, element,
    svg: (width, height, body) => element('svg', {xmlns:'http://www.w3.org/2000/svg',
      width, height, viewBox:`0 0 ${width} ${height}`}, body),
    text: (text, attributes = {}) => element('text', attributes, escapeXML(text)),
    font: (name, file) => host('motions.font', name, file),
    render: async (options, frame = options.frame) => {
      if (typeof frame !== 'function') throw new TypeError('Motion needs frame(context)');
      const still = /\.(png|svg)$/i.test(options.output || '');
      const at = options.time ?? 0;
      if (!Number.isFinite(at) || at < 0) throw new TypeError('Motion time must be nonnegative');
      const session = host('motions.start', options);
      try {
        for (let i = 0; i < session.frames; ++i) {
          const time = still ? at : i / session.fps;
          const context = {...session, frame:still ? Math.round(time*session.fps) : i, time,
            progress:still ? Math.min(time/session.duration,1) : session.frames > 1 ? i/(session.frames-1) : 0};
          const svg = await frame(context);
          if (typeof svg !== 'string') throw new TypeError('Motion frame must return SVG markup');
          host('motions.frame', session.id, svg);
        }
        return host('motions.finish', session.id);
      } catch (error) {
        try { host('motions.cancel', session.id); } catch {}
        throw error;
      }
    }
  });
  globalThis.slopfab = Object.freeze({...__slopfabInfo, fs, path, http, exec, spawn, cli, motions,
    cwd: () => host('cwd'),
    chdir: dir => host('chdir', dir),
    env: Object.freeze({get: key => host('env.get', key), set: (key, value) => host('env.set', key, value)}),
    sleep: ms => host('sleep', ms),
    sha256: file => host('sha256', file),
    tool: name => host('tool', name),
    result: value => host('result', value),
    log: (...args) => host('log', args.map(v => typeof v === 'string' ? v : JSON.stringify(v)).join(' '))
  });
  globalThis.console = Object.freeze({log: slopfab.log, info: slopfab.log, warn: slopfab.log, error: slopfab.log,
    debug: slopfab.log, assert: (condition, ...args) => { if (!condition) throw new Error(args.join(' ') || 'Assertion failed'); }});
  globalThis.print = slopfab.log;
  globalThis.TextEncoder = class { encode(text = '') { return __slopfabEncode(String(text)); } };
  globalThis.TextDecoder = class {
    constructor(encoding = 'utf-8') { if (!['utf-8','utf8'].includes(encoding.toLowerCase())) throw new Error('Only UTF-8 is supported'); }
    decode(bytes = new Uint8Array()) { return __slopfabDecode(bytes); }
  };
  let timerId = 0;
  const timers = new Map();
  function timer(fn, ms, args, repeat) {
    if (typeof fn !== 'function' || !Number.isFinite(ms) || ms < 0) throw new TypeError('timer needs a function and nonnegative delay');
    const id = ++timerId;
    timers.set(id, {fn, at: Date.now() + ms, ms, args, repeat});
    return id;
  }
  globalThis.setTimeout = (fn, ms = 0, ...args) => timer(fn, ms, args, false);
  globalThis.setInterval = (fn, ms = 0, ...args) => timer(fn, ms, args, true);
  globalThis.clearTimeout = globalThis.clearInterval = id => timers.delete(id);
  globalThis.__slopfabTick = () => {
    const now = Date.now();
    for (const [id, t] of [...timers]) {
      if (!timers.has(id) || t.at > now) continue;
      if (t.repeat) t.at = now + Math.max(t.ms, 1); else timers.delete(id);
      t.fn(...t.args);
    }
    return timers.size ? Math.max(0, Math.min(...[...timers.values()].map(t => t.at)) - Date.now()) : -1;
  };
})();
