# Motions

Motions are JavaScript-driven vector graphics and animations. A Motion module
returns SVG artwork for each frame; the CLI's embedded ThorVG CPU renderer turns
that artwork into pixels. Static graphics work without a GPU, browser or external
script interpreter. Video encoding and composition use external FFmpeg.

**ThorVG and QuickJS are statically linked only into `slopfab.exe` (or the Linux
CLI). `slopfab.dll` / `libslopfab.so` and the C ABI have no ThorVG dependency.**
The source is pinned to [ThorVG v1.1.3](https://github.com/thorvg/thorvg/releases/tag/v1.1.3)
and built offline with its CPU, SVG, PNG and TTF/OTF support. Lottie, GPU rendering
and browser/DOM APIs are not enabled in this integration.

```sh
slopfab motions frame examples/motions/orbit.js --out still.png --time 1
slopfab motions frame examples/motions/orbit.js --out artwork.svg --time 1
slopfab motions render examples/motions/orbit.js --out motion.mp4
slopfab motions render examples/motions/orbit.js --out transparent.mov
slopfab motions render examples/motions/orbit.js --out composed.mp4 --over generated.mp4
slopfab motions render examples/motions/orbit.js --out composed.mp4 --over generated.y4m --audio generated.wav
```

Add `--agent` for the normal structured response. Paths are relative to the
invocation directory. Parent output directories must exist. Existing files are
preserved unless `--overwrite` is supplied; even then the output is published
only after successful rendering/encoding. Failures remove partial output and
stop the encoder. The output must differ from the overlay/audio inputs.

## Authoring a Motion

```js
import {motions as m} from 'slopfab';

export default {
  width: 1280, height: 720, fps: 30, duration: 3,
  frame({width, height, time, progress}) {
    return m.svg(width, height, [
      m.element('rect', {x: 60 + progress * 800, y: 250,
        width: 240, height: 140, rx: 8, fill: '#d4ef85',
        transform: `rotate(${time * 10} 640 360)`})
    ]);
  }
};
```

The default export can also be a factory `(args) => motion`, including an async
factory. Arguments after `--` are passed unchanged. Relative imports resolve
against the module; `slopfab.scriptDirectory` locates companion assets. Use
`slopfab.fs.readText` to load an existing SVG, or construct markup directly.

The `frame(context)` callback returns a complete SVG string and may be async.
Callbacks run sequentially at fixed timestamps, not wall-clock animation speed.

| Context field | Meaning |
| --- | --- |
| `width`, `height` | Effective canvas dimensions after CLI overrides. |
| `fps`, `duration` | Effective Motion timing. |
| `frames` | Number of rendered frames: `ceil(fps * duration)` for video, 1 for a still. |
| `frame` | Zero-based animation frame; for stills, `round(time * fps)`. |
| `time` | Seconds: `frame / fps` for video, `--time` (default 0) for stills. |
| `progress` | 0..1 across rendered video frames; for stills, `min(time / duration, 1)`. |

The final video timestamp is one frame interval before its encoded end. Prefer
`time` for physically timed motion, `progress` for transitions spanning the
whole export, and deterministic math/seeds for repeatable renders.

`--width`, `--height`, `--fps` and `--duration` override the module's values.
Absent both CLI and module values, defaults are 1280×720, 30 fps and 5 seconds.
`--time` applies to stills. The rendered canvas supports SVG paths, shapes,
gradients, strokes, groups, transforms, opacity, clipping, masks, and text within
ThorVG's SVG support. SVG's declarative animation and JavaScript inside SVG are
not the timeline; animate attributes in the JavaScript callback instead.

### JavaScript helpers

The global `slopfab.motions` and named `motions` import expose:

| Function | Behavior |
| --- | --- |
| `svg(width, height, body)` | Wrap markup (or an array of markup strings) in an SVG with namespace and viewBox. |
| `element(tag, attributes, children)` | Build SVG markup; attribute values are XML-escaped, children are markup strings/arrays. |
| `text(value, attributes)` | Build an SVG text element with escaped text. |
| `escape(value)` | XML-escape plain text for hand-built SVG. |
| `font(name, path)` | Load a TTF/OTF file under an explicit family name. |
| `render(options, frame)` | Await a full render from any `slopfab run` workflow; `options.frame` is also accepted. |

`render` options use camel-free names matching the Motion definition and CLI:
`width`, `height`, `fps`, `duration`, `output`, `time`, `over`, `audio`, `x`, `y`,
`overwrite` and `ffmpeg`. It returns output path, renderer, canvas dimensions,
rendered frame count, fps, Motion duration and whether composition was enabled.
For overlays the output canvas/duration come from the base video; the returned
width/height and `motion_duration` describe the Motion layer.

```js
import sf, {motions as m} from 'slopfab';
import motion from './orbit.js';

export default async () => {
  const result = await m.render({...motion, output: 'orbit.mp4'}, motion.frame);
  sf.cli(['projects', 'add', 'demo', '--id', 'orbit', '--file', result.output,
    '--title', 'Orbit Motion']);
  return result;
};
```

Register rendered Motion videos/stills using existing project video/image items.
For an assembled project video, register the composed replacement with
`projects update` on the same item ID so previous versions remain available.

### Text and fonts

Load fonts explicitly; no font is bundled or downloaded automatically:

```js
import {motions as m} from 'slopfab';
m.font('Title', 'fonts/MyFont.ttf');
const label = m.text('A title & subtitle', {
  x: 40, y: 80, 'font-family': 'Title', 'font-size': 48, fill: '#ffffff'
});
```

See [lower-third.js](../examples/motions/lower-third.js) for an animated title
factory. Pass a font path and optional title after `--`. Font selection is
explicit to make rasterization reproducible. Exported SVG retains text/family
references; viewers need the matching font unless your SVG embeds fonts or uses
outlined paths. Raster/video output already contains the rendered text.

## Formats and composition

| Output | Behavior |
| --- | --- |
| `.svg` | Vector markup for one frame, validated/rendered by ThorVG before publishing. |
| `.png` | Straight RGBA8 still with transparency; native PNG writer, no FFmpeg. |
| `.mp4` | H.264/yuv420p with AAC audio when present; standalone transparency is composited onto black. |
| `.mov` | ProRes 4444 with alpha, suitable for editing/compositing; PCM audio when present. |
| `.webm` | VP9 with alpha; Opus audio when present. Alpha playback depends on the decoder/player. |

Add a full-canvas rectangle to the SVG when you want an opaque background of a
different color. Transparent MOV/WebM files can be composited later. FFmpeg must
provide the codecs used for the selected format. Use `--ffmpeg PATH` to choose an
installation; otherwise Slopfab searches its usual media-tool locations/PATH.

`--over VIDEO` places the Motion at the top-left of the base video. `--x`/`--y`
offset it in pixels; the layer is not silently resized. Set matching canvas
dimensions for a full-frame overlay. Base video timestamps are normalized to
zero. Its full video duration and frame rate are retained, and the Motion
disappears when its own timeline ends. A shorter base limits the composite's
visible duration. Existing base audio is retained and encoded for the output
format. `--audio FILE` replaces it or attaches a separate generated soundtrack;
short audio is padded with silence and long audio is trimmed to the video.

For standalone MP4 output, dimensions must be even. Video codecs may impose
further constraints. Canvas limits are 1..8192 on each axis and 16 megapixels in
total; fps is 1..240 and duration is up to one hour. SVG markup is limited to
16 MiB per frame and fonts to 64 MiB. The renderer keeps a single frame buffer
and streams frames to FFmpeg; it does not accumulate an entire uncompressed
animation or create a folder of temporary frame images.

Use `--timeout-ms` and `--memory-mb` as with embedded workflows. The JavaScript
heap limit does not cap ThorVG or FFmpeg memory. Cancellation interrupts the
JavaScript timeline and stops blocked encoder writes; a synchronous ThorVG draw
can finish before observing cancellation. `motions` and `run` return errors
through the usual `SCRIPT_ERROR` envelope when a render fails.
