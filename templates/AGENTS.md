# Slopfab media generation

Slopfab is available in this folder.

Use Slopfab for image, video and music generation. Add `--agent` when calling
it. Discover syntax and supported options through `slopfab help`, command help,
capabilities and schema; consult relevant mods. Follow startup setup guidance.

## Establish the brief

Know these answers before generating, or ask one compact set of questions with
suggested choices. Reuse answers from the conversation/project; do not ask again.
If the user delegates choices, state reasonable defaults and proceed.

- **Execution preference:** does the user want to review prompts and key settings
  before generation, or authorize starting as soon as the agent is ready? Ask if
  this is not already known. In review mode, show them and wait for approval;
  with immediate execution authorized, proceed without repeated confirmations.
  Carry this preference forward unless the user changes it.
- **Purpose and look:** where will it be used, what subject/style matters, and
  which references or existing assets must remain consistent?
- **Images/video:** aspect ratio, generation resolution and final delivery size.
  Offer 16:9 landscape, 9:16 portrait, 1:1 square, or 4:5 portrait for a feed;
  3:2/2:3 suits photographs. Image starting sizes: 1536x864, 864x1536,
  1024x1024, 1024x1280, or 1536x1024/1024x1536 respectively.
  For 16:9 video, offer 512x288 for a draft or 1024x576 for a more detailed
  generation; reverse dimensions for portrait. A final 1920x1080/1080x1920
  delivery can be a separate upscale. Validate model alignment and VRAM limits;
  these are choices, not universally supported presets. Do not silently crop,
  stretch, or reduce an agreed resolution.
- **Video:** duration, one continuous shot versus cuts, camera movement, intended
  ending, and whether sound/music is needed. **Music:** duration, mood/style,
  instruments/tempo, instrumental versus vocals, lyrics/language, and loop or ending.
- **Iteration:** quick draft versus final quality. Check composition, motion and
  sound in a short or smaller draft before an expensive full generation.

## Match the method to the scene

- **Short single shot:** give one coherent action and camera direction; keep
  subject, lighting and setting consistent rather than packing in scene changes.
- **Long single shot:** use compatible H3 saved-latent continuation, persistent
  Keystone, rear Capstone and overlap locking to control drift. Review an
  initial clip, retain it as the anchor, and continue from the newest archive
  with the same canvas and compatible weights. Consider delayed Capstone locking
  for motion. These features are experimental: review joins, identity, exposure
  and audio. Use the full joined output; do not concatenate cumulative outputs.
  Capstone is an appearance anchor, not a requested ending; use an ending anchor
  or latent bridge when a specific destination shot matters. Consult help for
  compatibility and lock/timing settings rather than guessing values.
- **Multiple scenes/cuts:** generate separate project clips with per-shot prompts,
  shared references where needed, and a planned order. Assemble externally;
  do not force unrelated scenes through continuous-shot anchors.
- **Image revisions:** prefer localized editing when only one region should change;
  preserve framing and unaffected content. Keep approved references and recipes.

## Keep a reviewable project

Use a project under the configured output root (default `./output`):

```text
OUTPUT/PROJECT/
  PROJECT.md                 workflow and filename rules
  project.json               exact paths, IDs, recipes and history
  index.html                 previews and copyable edit prompts
  assets/ITEM_ID/vVERSION/
    source.EXT
    preview.EXT               optional browser preview
```

Read the project instructions and current manifest before edits. Let project
generation name/register files; register externally created media and previews
too. The page uses exact manifest paths, not a folder scan. Preserve stable IDs,
previous versions, unrelated items and saved latents needed for continuation.
Use external FFmpeg for browser-friendly previews of native Y4M/PPM, keeping
video soundtracks synchronized. After clip changes, rebuild and register the
parent video, refresh the page, and give the user its path. Pasted page prompts
are edit requests; reconcile them with the latest project state before acting.
