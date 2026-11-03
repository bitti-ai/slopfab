# Agent CLI

Run `slopfab` with no arguments for a short introduction. Startup first suggests
installing the NVIDIA CUDA Toolkit when NVIDIA hardware is present without a
compatible installation, then installing FFmpeg if it is absent from PATH and
the executable directory. It shows only the first missing prerequisite; rerun
after installation to get the next instruction. If no weight paths are configured,
startup asks the agent to request a path from the user and save it with
`slopfab settings set weight_paths PATH`. Normal instructions resume once configured. Explicit help and commands remain
accessible. FFmpeg remains an external conversion tool, not a linked dependency. Every command accepts
`--help`; add `--agent` for machine-readable help and responses. The DLL's C ABI
is unchanged. CLI requests, jobs, output policy, and configuration are separate
from DLL generation.

The first `--agent` invocation in a working directory creates `AGENTS.md` if
that file does not already exist. Its compact instructions cover Slopfab for
image/video/music requests, projects, filename rules, previews and edits. The
same text appears once in the first JSON response as
`agent_context: {path, instructions}`; the requested command still runs normally.
Subsequent launches do not rewrite or repeat it. Existing files (even empty
ones) and symlinks are preserved. This is per working directory, independent
of the global config, and does not run for ordinary non-agent launches or
background workers. The embedded source is [templates/AGENTS.md](../templates/AGENTS.md).

For generation requests, start with `slopfab projects create NAME` and pass
`--project NAME` to generation. Projects provide an offline review page with
copyable agent edit prompts, saved recipes, clip relationships and explicit
filename rules. See [projects](projects.md) or `slopfab help projects`.

```sh
slopfab capabilities
slopfab models list
slopfab doctor
slopfab schema
slopfab generate image --prompt "A red apple" --seed 42 --out apple.png
slopfab generate video --prompt "A cat playing piano" --duration 5 --out cat.y4m
slopfab generate music --prompt "Gentle piano" --duration 10 --out piano.wav
```

Image defaults to Krea 2, video to H3, and music to YuE2. `--model` selects a
family or configured alias. H3 also supports `generate image --model h3`.
Models still require compatible local weights and a supported GPU. No command
downloads weights. `capabilities` reports compiled support; `doctor` probes
available devices. `models list` reports files, which does not prove checkpoint
compatibility or that inference fits in memory.

## Requests and validation

```json
{
  "schema_version": 1,
  "task": "video",
  "model": "h3",
  "prompt": "A cat playing piano in warm lamplight",
  "duration_seconds": 5,
  "resolution": {"width": 512, "height": 512},
  "seed": "42",
  "output": {"path": "artifacts/cat.y4m", "format": "y4m", "overwrite": false},
  "deadline_seconds": 600,
  "options": {"steps": 20}
}
```

```sh
slopfab validate --request request.json
slopfab validate --request request.json --check-ready
slopfab generate --request request.json
```

`--request -` reads one UTF-8 JSON object from stdin (maximum 16 MiB). Prompts
and lyrics may contain newlines and Unicode. Seeds are decimal strings covering
the full unsigned 64-bit range. Unknown fields, duplicate JSON keys, and NUL
characters are rejected. Request-file paths and reference/output paths are
resolved against the invocation's working directory. Jobs retain that directory.
For project generation, a relative output path is instead relative to the
project folder. Use top-level `project`, `project_item` and `output_directory`
fields in JSON requests.

A request cannot be combined with generation flags. `--dry-run` and
`--check-ready` are allowed alongside it. Global `--agent`, `--config`, and
`--jobs-dir` also remain available. Ordinary flags retain last-value-wins
behavior; incompatible controls such as duration and frame count are rejected.

`options` holds advanced generate flags without the `--` prefix. Use booleans
for switches, numbers/strings for values, arrays for repeatable options, and
objects for inline JSON settings. Top-level fields cannot be overridden through
`options`. `generate --help` lists the supported advanced controls. For ordered
reference/adapter settings, prefer these explicit lists:

```json
{
  "references": [{"type": "image", "path": "character.png"}],
  "loras": [{"path": "adapter.safetensors", "strength": 0.8}]
}
```

Reference types are `image`, `video`, `audio`, and `refmod`. They remain subject
to the selected model's compatibility rules. Use `lyrics` for music lyrics.

Validation resolves dimensions, frame alignment, duration, seed, output path,
and sampling settings without loading GPU weights. It may read checkpoint
headers and reference/latent files. `--check-ready` additionally checks required
checkpoint files. It does not promise sufficient VRAM or successful execution.
Music duration is a ceiling; generated audio can end earlier. Video duration is
aligned to the model's supported frame grid. Results report actual media length.

## Agent protocol

`--agent` reserves stdout for UTF-8 newline-delimited JSON. Diagnostics go to
stderr, including messages from the inference libraries. Flush and parse each
line independently; do not parse diagnostic text. Each generation or ordinary command emits exactly
one terminal `result`, `error`, or `help` event. Generation may first emit
`started`, `progress`, and `artifact` events. Every event has `schema_version: 1`.
Older inspection/conversion commands wrap their textual result in `text`.
A foreground `worker --agent` streams multiple jobs; each job has its own
terminal event and `job_id`. Worker diagnostics are saved in `worker.log`.

Progress contains a stage, completed/total counts when known, and elapsed
seconds. Counts are stage-local, not an overall percentage. Events are rate
limited; long GPU operations can leave gaps. A completed sample is reported
with absolute artifact paths, MIME types, byte lengths, media dimensions or
sample rate, resolved plan/seed, and elapsed time. Batch errors retain earlier
results. Treat the final result as authoritative; a progress stage named
`finished` alone is not a success acknowledgement.

Errors contain `code`, `message`, `field` when known, and `retryable`.

| Exit | Meaning |
| --- | --- |
| 0 | Command succeeded; inspect job state for `status`/`wait` |
| 1 | Generation, worker, or other runtime failure |
| 2 | Invalid request or idempotency conflict |
| 3 | Missing/ambiguous model or missing input |
| 4 | Unsupported capability |
| 5 | Out of memory |
| 6 | Output failure/collision |
| 7 | Busy GPU/store or full queue; retry later |
| 8 | `wait` timed out; the job continues |
| 124 | Generation deadline exceeded |
| 130 | Cancelled |

Ctrl+C and `--deadline SECONDS` request cooperative cancellation. Cancellation
is checked at runner callbacks; it cannot interrupt an executing GPU kernel.
It may also be delayed during model loading or media writing. `cancel JOB_ID`
does not kill a process. Poll status until terminal. Retries do not automatically
reduce quality, change backends, or resume partial diffusion.

## Native media

Neither executable nor DLL requires FFmpeg. Native outputs are PNG (RGB8),
optional PPM, Y4M video, and WAV audio. PNG uses lossless stored DEFLATE blocks;
files are larger than compressed PNGs. Video and its soundtrack are separate
files with the same stem. MP4/MP3 output paths are rejected before inference.

Output directories are created as needed. Default paths use a random run
directory. Existing files are refused unless `--overwrite` is selected. CLI
generation reserves output names and writes media to temporary sibling files
before publishing each completed artifact. Multi-file delivery is not one
filesystem transaction: an error can leave already published artifacts, which
are included in the partial result. Stale `.slopfab-reserved` directories after
a forcibly killed process require inspection and removal before reusing a path.

Use a separately installed FFmpeg yourself, for example:

```sh
ffmpeg -i cat.y4m -i cat.wav -c:v libx264 -pix_fmt yuv420p -c:a aac cat.mp4
```

Reference video input is progressive, limited-range BT.709, 8-bit 4:2:0 Y4M;
reference audio is mono/stereo PCM16/24/32 or float32 WAV. Convert compressed
inputs externally. Y4M has no soundtrack; pass audio separately when supported.
Native reference durations retain the existing 2–15 second limit. Decoded video
input is bounded at 2 GiB; WAV chunks at 256 MiB. Windows uses WIC for still
images; Linux uses libpng/libjpeg. `upscale` accepts native images/Y4M and writes
PNG/PPM/Y4M; its native video reader currently buffers the clip within that bound.

## Configuration

Use `--config FILE`, `SLOPFAB_CONFIG`, or the path shown by `doctor`.
Defaults are `%LOCALAPPDATA%/slopfab/config.json` on Windows and
`$XDG_CONFIG_HOME/slopfab/config.json` (otherwise `~/.config/slopfab/config.json`)
on Linux. The CLI creates this file automatically with `schema_version: 1` and
an empty `weight_paths` list. The DLL does not read or create it.

```sh
slopfab settings get
slopfab settings set weight_paths "D:/models/slopfab" "E:/other-weights"
slopfab settings get weight_paths
slopfab settings remove weight_paths "E:/other-weights"
slopfab settings remove weight_paths
slopfab settings set models.local '{"family":"h3","transformer":"D:/models/h3.safetensors"}'
slopfab settings get models.local
slopfab settings remove models.local
```

`set weight_paths` replaces the list; it also accepts a JSON array as one argument.
Paths may be directories or individual checkpoint files. Paths supplied to this
command are resolved against the current working directory and saved as absolute
paths. Relative paths in manually edited configuration resolve against the config
file. Duplicate paths collapse. `remove weight_paths` without paths clears the
list. `models` and `presets` support dotted keys and JSON values; `schema_version`
is read-only. Writes are atomic and concurrent updates are serialized.
Existing `weights_directory` configuration migrates to `weight_paths` automatically.

```json
{
  "schema_version": 1,
  "weight_paths": ["D:/models/slopfab"],
  "models": {
    "image-default": {
      "family": "krea2",
      "transformer": "D:/models/krea2.safetensors",
      "text_encoder": "D:/models/qwen3vl.safetensors",
      "vae": "D:/models/qwen_image_vae.safetensors"
    }
  },
  "presets": {
    "product-image-v1": {
      "task": "image",
      "model": "image-default",
      "resolution": {"width": 1024, "height": 1024},
      "options": {"steps": 8}
    }
  }
}
```

Select presets with a JSON request's `preset` field. Explicit request fields
replace preset fields; nested `options` is replaced as a whole. Names can carry
versions so callers can pin a recipe. Discovery searches each configured weight
path and its conventional `transformer/`, `text_encoder/`, and `vae/` subdirectories.
An empty list disables implicit discovery; explicit checkpoint arguments and
model aliases still work. Multiple matching checkpoints are an error; use aliases
or explicit checkpoint paths to disambiguate.

For lower-VRAM H3 text conditioning, model aliases also accept
`clip_projection`, or use `options.clip-projection` in a JSON request. Pair it
with a matching Qwen3-VL-4B/8B `text_encoder`. See [ClipProj setup and supported
scope](clipproj.md). The default 32B path is unchanged.

## Mods

Use `slopfab mods list` to retrieve active workflow names and descriptions, then
`slopfab mods load NAME` to read a selected prompt into agent context. Mods are
`mods/NAME/MOD.md` files with YAML front matter and a Markdown prompt; they can
live beside the settings file, beside the executable, or in the working directory.
`--agent` returns structured metadata and prompt text. See the [mod format and
sharing guide](mods.md) for setup, activation and error handling.

## Persistent jobs

```sh
slopfab submit --request request.json --idempotency-key scene-17
slopfab status JOB_ID
slopfab wait JOB_ID --timeout 60
slopfab cancel JOB_ID
slopfab worker --idle-seconds 60
slopfab worker --stop
```

Submit starts a hidden local worker. One worker owns each store, executes jobs
serially, and reuses a `GenerationSession`. This retains eligible conditioning
and model caches; weights still obey the existing per-stage memory policy.
CLI generation processes also share a per-user GPU lock. The DLL retains its
existing process-local concurrency rules and does not participate in that lock.

`--jobs-dir` selects the store; otherwise it is `jobs/` beside the config.
Capacity is 32 unfinished jobs. Job manifests and `events.ndjson` persist after
the caller exits. Requests are materialized before queuing, including stdin
requests. Reusing an idempotency key with the same arguments, config path and
working directory returns the original job. A different request is rejected.
Keep referenced files and configuration stable until execution completes.

States are `queued`, `running`, `succeeded`, `failed`, `cancelled`, and
`interrupted`. Restarting a worker marks abandoned running jobs interrupted;
it does not rerun them, since some output may already exist. Queued jobs survive
restarts. `worker --stop` exits after the current job; a new submit starts service
again. The worker normally exits after 60 idle seconds. This is a local queue,
not a network service, and does not expose an HTTP endpoint.
