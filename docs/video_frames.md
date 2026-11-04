# Video frames and contact sheets

`slopfab video frames` saves a specific video frame or a downscaled grid as PNG.
It is available in the executable and embedded JavaScript; it adds no dependency
to `slopfab.dll` or `libslopfab.so`.

```sh
slopfab video frames output.y4m --frame 48 --out frame.png
slopfab video frames output.mp4 --time 2.5 --out frame.png
slopfab video frames output.y4m --grid 4x3 --width 320 --out review.png
slopfab video frames output.mp4 --grid 3x2 --start 1 --end 4 --out detail.png
slopfab video frames --help
```

Frame indices start at zero. Without a selector, frame zero is selected. `--time`
selects the nearest frame timestamp, with ties choosing the earlier frame.
Times are seconds relative to the first frame, and must be between zero and the
last frame's timestamp. Compressed video uses the actual presentation timestamps,
including for variable frame rates.

A grid samples evenly in time, includes both ends of the range, and fills cells
from left to right, then top to bottom. Short inputs may repeat frames. A one-cell
grid selects the start. `--width` is the maximum width of **each cell**, preserving
aspect ratio and never enlarging the source. Grids default to 320 pixels per cell;
single frames default to native resolution. Downscaling averages source pixels.
Grids support up to 256 cells, 32 per side. Output images are limited to 64 million
pixels and 16384 pixels per side.

Native Y4M extraction needs no external tools. It supports the generated BT.709
limited-range, progressive, even-sized 8-bit 4:2:0 format. It indexes frame offsets,
then reads only selected frames rather than loading the entire video into RAM.
Other video formats require FFmpeg and ffprobe, found beside the executable,
under `external/ffmpeg/bin` in the current directory, or on PATH. Override these
with `--ffmpeg PATH --ffprobe PATH`. Probing compressed video scans its timestamps;
probe output is limited to 64 MiB. Extraction uses the first video stream in its
stored pixel orientation, without applying rotation metadata.

Existing output files are protected unless `--overwrite` is specified. PNGs are
published atomically; failures remove temporary files and output reservations.
`--agent` returns metadata in the final result's `data`: absolute `path`, PNG
dimensions, `columns`, `rows`, `cell_width`, `cell_height`, `source_frames`, and
`cells` containing the exact `frame`, `time`, `x` and `y` of each selection.

## JavaScript

Use `slopfab.frames` or `import {frames} from 'slopfab'`. These calls are synchronous
and return the same metadata as the CLI. PNG output paths are required.

```js
import {frames} from 'slopfab';

export default () => {
  const still = frames.extract({
    input: 'output.y4m', output: 'frame.png', frame: 48
  });
  const grid = frames.extract({
    input: 'output.mp4', output: 'review.png',
    columns: 4, rows: 3, width: 256, overwrite: true
  });
  return {still, grid};
};
```

The options mirror CLI selectors: `frame` or `time`, or `columns` and `rows` with
optional `start` and `end`. A grid with just one dimension defaults the other to
four columns or three rows. `ffmpeg` and `ffprobe` override compressed-video tools.
Workflow deadlines and cancellation also stop native extraction and child decoders.

To use an existing RAM buffer, pass an `ArrayBuffer` or an 8-bit typed-array view:

```js
const review = frames.fromRGB(rgbBytes, {
  sourceWidth: 512, sourceHeight: 288, count: 121, fps: 24,
  output: 'ram-review.png', columns: 4, rows: 3, width: 256
});
```

The buffer contains tightly packed RGB24: three bytes per pixel, rows in order,
then frames in order. Its length must exactly equal `sourceWidth * sourceHeight *
count * 3`. Typed-array offsets are honored. `count` defaults to 1 and `fps` to 30.
The host reads the buffer directly and only allocates the selected image/grid,
not a second copy of the entire source. Native image allocations are separately
bounded by the image limits above, outside the JavaScript heap limit.

## Review immediately after generation

```sh
slopfab --agent video frames generate --out review.png --grid 4x3 --width 256 -- --request shot.json
slopfab --agent video frames generate --out ending.png --frame 120 -- --request shot.json
```

Everything after `--` is a normal `generate` argument list. Generation keeps its
normal video/audio output and adds the review PNG to its returned artifacts.
The capture callback reads borrowed planar RGB floats while generation samples
are still alive in RAM. It does not read back the saved video or duplicate the
full decoded buffer. The resulting image identifies its source as
`memory:generation`. This also works for generated still images. Music-only
generation has no video samples and cannot produce a frame review.

With `--count`, reviews are named `review.png`, `review-2.png`, `review-3.png`, etc.
Review `--overwrite` is independent of any generation `--overwrite` placed after
the delimiter. Choose review paths inside the project when generating project
assets; the review is recorded in the generation artifacts, not automatically
registered as a separate project library item.

The JavaScript wrapper runs generation in a managed child process so the model's
RAM stays in that process. Capture happens there before decoded samples are
released; the workflow receives PNG paths and metadata rather than transferring
gigabytes of pixels through JavaScript or JSON.

```js
import {frames} from 'slopfab';

export default () => {
  const generation = frames.generate(['--request', 'shot.json'], {
    output: 'review.png', columns: 4, rows: 3, width: 256
  }, {timeoutMs: 3600000});
  return generation.result;
};
```

The return value follows `slopfab.cli`: `exitCode`, captured output, parsed `events`
and final `result`. The final result's `results[].artifacts` includes the PNG and
its frame/cell metadata. Process options such as `timeoutMs` and `check` are
supported in the third argument. A review error makes generation fail; previously
published outputs are retained. `--dry-run` validates the generation request but
does not decode samples or create a review image.
