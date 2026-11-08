# Embedded JavaScript workflows

`slopfab run` executes JavaScript modules with embedded QuickJS-NG. File handling,
HTTP downloads, subprocesses, generation, jobs and project operations can all be
orchestrated in one script without a Python, Node.js or shell-script runtime.
QuickJS is statically linked **only into the CLI executable**. Neither
`slopfab.dll` nor Linux `libslopfab.so` links it or exposes its API. FFmpeg remains
an external executable that a workflow can invoke directly.

```sh
slopfab run workflow.js -- input.y4m output.mp4
slopfab --agent run workflow.js --timeout-ms 60000 --memory-mb 256 -- input.y4m output.mp4
slopfab run --eval 'export default () => ({answer: 6 * 7});'
slopfab run -
```

`-` reads a module from stdin. Put script arguments after `--` so that global CLI
options and help flags remain literal arguments. Shell quoting differs between
platforms; use a file or stdin for longer scripts. The working directory remains
the caller's directory. Imports resolve relative to the importing module; access
companion assets through `slopfab.scriptDirectory`.

```js
import sf from 'slopfab';

export default ([input, output]) => {
  if (!input || !output) throw Error('Expected INPUT.y4m OUTPUT.mp4');
  sf.exec([sf.tool('ffmpeg'), '-nostdin', '-n', '-i', input,
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output], {timeoutMs: 120000});
  return {output: sf.path.resolve(output), url: sf.path.fileUrl(output)};
};
```

The optional default-exported function receives the argument array. Its return
value becomes the final response's `data`; promises are awaited. Alternatively,
call `slopfab.result(value)` explicitly. Results must be JSON-serializable.
`console.log`, `console.error`, `print` and `slopfab.log` write to stderr; stdout
contains one structured result or error. Uncaught errors include the JavaScript
stack and use `SCRIPT_ERROR` (exit 6). Catch host errors with ordinary `try/catch`.

## API version 1

The global `slopfab` and the default export of `import sf from 'slopfab'` are the
same object. Named exports include `fs`, `path`, `http`, `exec`, `spawn`, `cli`,
`env`, `sleep`, `sha256`, `tool`, `args`, `motions` and `frames`. The [Motions API](motions.md)
renders SVG artwork and JavaScript animations using the CLI's embedded ThorVG.
The [frames API](video_frames.md) extracts stills and contact sheets from video files,
packed RGB buffers, or live generation samples in RAM.
`slopfab system capabilities` advertises the
engine version, API version, supported facilities and installed mod commands.

| Member | Behavior |
| --- | --- |
| `args`, `scriptPath`, `scriptDirectory` | Argument array and absolute entry-module paths. |
| `executable`, `config`, `jobs` | Current executable, settings file and jobs directory. |
| `version`, `apiVersion`, `platform` | Slopfab version, host API version, `win32` or `linux`. |
| `cwd()`, `chdir(directory)` | Get/change the workflow's working directory. |
| `env.get(name)`, `env.set(name, value)` | Read/write environment strings; `null` removes a variable. |
| `sleep(ms)` | Blocking, interruptible delay. |
| `sha256(file)` | Stream a file through SHA-256; return lowercase hex. |
| `tool(name)` | Locate a media tool beside Slopfab or on PATH, including `ffmpeg`. |
| `result(value)`, `log(...values)` | Set the final data or write diagnostics. |

### Files and paths

`fs.readText(path, {maxBytes})` reads UTF-8 text (64 MiB default limit).
`fs.writeText(path, text, {append: false})` creates or replaces a file.
`fs.readJson(path)` and `fs.writeJson(path, value)` handle JSON.
`fs.readBytes(path, {maxBytes})` returns a `Uint8Array`; `fs.writeBytes(path, data)`
accepts an `ArrayBuffer` or typed-array view, respecting its offset and length.
Basic UTF-8 `TextEncoder` and `TextDecoder` are available.

`fs.exists(path)`, `fs.stat(path)` and `fs.list(directory)` inspect files. Stats
include `isFile`, `isDirectory`, `isSymlink` and `size`; sorted directory entries
include `name`, absolute `path`, `isDirectory` and `isSymlink`.
`fs.mkdir(path)` creates missing parents. `fs.copy(from, to, {recursive: false,
overwrite: false})`, `fs.rename(from, to)` and `fs.remove(path, {recursive: false})`
perform filesystem operations. Writes do not automatically create parent folders.

`path.join(...parts)`, `path.resolve(...parts)`, `path.dirname(file)`,
`path.basename(file)` and `path.extname(file)` use native filesystem paths.
`path.fileUrl(file)` produces an absolute, percent-encoded `file://` URL suitable
for sharing the project's `NAME.html` in a terminal.

### Processes and all CLI commands

`exec([executable, ...args], options)` waits for a process and returns
`{done: true, exitCode, stdout, stderr}`. Arguments are passed directly; no shell
expands quotes, pipes, globs, `$` or redirection. Use absolute executable paths
when overriding PATH for a child.

| Option | Default / meaning |
| --- | --- |
| `cwd` | Current directory. |
| `env` | Object overriding inherited variables; `null` deletes a variable. |
| `stdin` | Text supplied on stdin; otherwise empty/EOF. |
| `stdinFile` | Stream an existing file to stdin, relative to child `cwd`; mutually exclusive with `stdin`. |
| `stdoutFile`, `stderrFile` | Redirect to files relative to child `cwd`, replacing them; the corresponding returned string is empty. |
| `maxOutputBytes` | 16 MiB per captured stream; redirect large/binary output to files. |
| `timeoutMs` | 0 (no child timeout); enforced when polling/waiting. |
| `check` | `true`: a nonzero exit throws with captured output in `error.result`. |

`spawn(argv, options)` returns `{id, pid, poll(), wait(), kill()}`. Start several
processes before waiting to run them concurrently. `poll()` returns `{done:false}`
while running, or the full result when finished. `wait()` checks the exit code;
`poll()` lets the caller inspect it. `kill()` terminates the process group/job.
Normal child processes are scoped to the workflow and terminated when it ends.
Captured output is spooled to temporary files, avoiding pipe-buffer deadlocks.

`cli(args, options)` invokes the current Slopfab executable with `--agent` and the
same settings/jobs paths, returning `{...processResult, events, result}` where
`result` is the final parsed NDJSON event. This exposes every built-in and mod
command without duplicating generation logic in the JavaScript API. The normal
agent instructions creation/upgrade rules also apply to these child invocations.

```js
import sf from 'slopfab';
export default () => {
  const request = {schema_version: 1, task: 'image', model: 'h3',
    prompt: 'A red apple', output: {path: 'apple.png'}};
  const submitted = sf.cli(['submit', '--request', '-'], {
    stdin: JSON.stringify(request)
  }).result;
  // Persistent Slopfab workers survive the submitting workflow.
  return sf.cli(['wait', submitted.job.job_id]).result;
};
```

### HTTP and downloads

`http.request(url, options)` returns `{status, ok, headers, body, bytes, path}`.
Response header names are lowercase. `http.json(url, options)` requires a 2xx
response and parses its body. `http.download(url, destination, options)` streams
binary data to disk and returns the same metadata with an empty body and an
absolute path. It requires a 2xx response and publishes the destination atomically.
By default failures remove the temporary `.part` file and preserve any old destination.

Options: `method` (GET), `headers` (object), `body` (text), `timeoutMs` (300000;
0 disables the total HTTP timeout), `maxBytes` (16 MiB for in-memory responses),
and `overwrite` (false; downloads only). Downloads are not buffered in JavaScript
and have no default size cap. HTTP/HTTPS, redirects and standard TLS certificate
validation are supported. Transport failures throw; `request` leaves HTTP status
handling to the script. On Windows, blocking network calls can delay cancellation
until the current I/O timeout; with `timeoutMs:0`, the per-operation timeout is
60 seconds. Authentication can be supplied explicitly in request headers.

For resumable downloads with native parallel byte-range transfers:

```js
const file = slopfab.http.download(url, 'weights/model.safetensors', {
  resume: true, connections: 4, retries: 2, timeoutMs: 7200000
});
```

`resume: true` enables persistent partial files and defaults to four connections.
`connections` accepts 1–16; a value greater than one also enables this downloader
and defaults `resume` to true. Set `connections: 1, resume: true` for a single
resumable range. `resume: false` with multiple connections discards previous
progress and removes parts on failure. `retries` accepts 0–10 (default 2), in
addition to the first attempt. The total `timeoutMs` covers probing, retries,
transfers and assembly; runtime deadlines and cancellation still apply.

The downloader probes the original URL with `Range: bytes=0-0`, follows redirects,
and requires a strong ETag before splitting or reusing bytes. Each range request
uses `If-Range` and validates the returned status, ETag, Content-Range, encoding,
and byte count before publishing. A changed URL, ETag, size or connection count
invalidates saved progress. Servers that ignore ranges or lack a strong ETag fall
back to a single streamed request, restarting that request on retry. These modes
require GET without a body and manage `Range`, `If-Range` and `Accept-Encoding`
headers internally. See the [HTTP range semantics](https://www.rfc-editor.org/rfc/rfc9110.html#section-14).

State lives in `DESTINATION.download/` beside the destination, with one file per
range and a JSON metadata record. Repeat the same download to resume after a
failure or process restart. Successful assembly removes this directory. Assembly
requires space for the parts plus a complete final file, while memory use stays
bounded. A persistent `DESTINATION.download.lock` file prevents concurrent writers;
the OS releases its lock when the process exits. The final destination is never a
partial file, and `overwrite: true` preserves the old file until publication.

Results also contain `connections` (actual range count, or 1 for fallback) and
`resumedBytes` (bytes reused from previous attempts). On Windows, resumable-mode
blocking network calls have at most a 15-second per-operation timeout; cancellation
can wait for an in-flight call. JavaScript callbacks run only on the caller's
thread; the native transfer workers do not enter the JavaScript runtime.

## Runtime behavior and extension

Local ES modules, dynamic import, promises, top-level `await`, `setTimeout`,
`setInterval` and their clear functions are supported. The event loop drains
pending jobs and timers before completing. Unhandled rejections and unresolved
promises with no remaining work fail instead of silently succeeding. Host file,
HTTP and `exec` operations are synchronous, so timers run between host calls.
Use `spawn` or persistent Slopfab jobs for concurrency.

This is an embedded JavaScript host, not Node.js or a browser: npm packages that
depend on Node APIs, DOM, `fetch`, sockets or native Node addons are not provided.
Use the documented host APIs and direct native executables for those workflows.
Scripts have the invoking user's filesystem, network and process access; run
trusted scripts/mods. The JavaScript heap defaults to 256 MiB (`--memory-mb`
accepts 16..16384); this does not limit child processes or all native allocations.
`--timeout-ms` defaults to unlimited. Ctrl+C interrupts JavaScript, sleeps and
process waits; native filesystem/network operations may finish before observing it.

To add a CLI command without rebuilding, package a module in a
[mod with `command` and `script` metadata](mods.md#executable-commands).
See [the download mod](../examples/mods/download/MOD.md) and
[FFmpeg preview workflow](../examples/workflows/preview.js). For new native host
facilities, extend the CLI's `script.cpp`/`script_bootstrap.js`; keep them out of
the core and C API targets. The vendored engine's version and license are recorded
in [third_party/quickjs/README.slopfab.md](../third_party/quickjs/README.slopfab.md).
