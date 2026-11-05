# Agent-driven projects

Use a project when a user asks for generated media. A project is a folder under
the output directory, containing `project.json`, `PROJECT.md`, `index.html` and
its media. The CLI embeds the [HTML template](../templates/project.html), so the
page also works when the executable is installed by itself. No server, CDN,
account or browser-to-agent connection is required.

```sh
slopfab projects create launch-film --title "Launch film"
slopfab generate image --project launch-film --prompt "A quiet city at dusk"
slopfab generate video --project launch-film --prompt "A slow camera move over the city"
slopfab generate music --project launch-film --prompt "Warm ambient synths" --duration 10
slopfab projects show launch-film
slopfab projects open launch-film
```

Use `slopfab projects open NAME` to open `index.html` in the default browser,
or open the returned `index_path` yourself. The command supports `--output-dir`
and the configured output directory. If the page is missing, rebuild it with
`slopfab projects render NAME`. The page shows images, video and
music with custom playback controls, filters, a clip sequence and generation
details. The Clips and Generation prompts sections span the page width.
Select an item and choose Edit, Regenerate or Remove. Edit starts with an empty
Prompt field; the other actions fill a short request using the project name
and displayed clip number, such as "Remove clip 24 from the launch-film video".
Edit the text as needed and click Copy prompt to copy exactly that field into
your agent. Refresh the page after the agent updates the project. Page actions
never execute commands or silently change the manifest.

The main player opens the latest active assembled video, ignoring a previous
clip URL fragment. The clip strip starts with **FULL VIDEO**; selecting a clip
plays it in the main view, and FULL VIDEO returns to its current parent version.
Clip cards decode a paused frame directly from the registered browser-playable
video, including its start trim. No thumbnail images need to be generated.
Unsupported/missing videos show a placeholder. Other library assets open in a
modal with independent playback, generation prompts and edit actions; closing it
preserves the main selection and playback position.

Media registration records `media_revision` so the latest video can be selected
even when an older item receives a replacement. Videos with registered clips
take priority over standalone library videos. Older manifests without this field
fall back to registration order; every item uses its current registered version.

The **Generation prompts** panel beneath the preview shows the selected item's
prompt, negative prompt when supplied, and music lyrics. Copy each input directly
or select an earlier version to inspect its saved text; the media preview and edit
actions continue to target the current item. New generations save resolved text
in `generation.prompts`, including prompt files and request/music settings. Older
items fall back to their recorded prompt and saved request, labeled accordingly.
Missing text is never reconstructed from a file that may have changed. Rebuild
existing pages with `slopfab projects render NAME` to use the updated view.

Use `--agent` for JSON responses and generation progress. Project management
does not require model weights or a GPU. The DLL and generation engine have no
project dependency.

## Output directory and filenames

For generation on another computer, use [SSH execution](remote.md) with
`generate --host HOST --local-project NAME --project NAME`.
The remote project records generation; the local project imports downloaded
results and rebuilds its review page. Each run creates new local items, without
copying the remote project manifest or merging its edits.

The default root is `./output`. Set a persistent root with
`slopfab settings set output_directory PATH`, or pass `--output-dir DIR` to
project commands and project generation. This setting also controls default
output locations for generation outside projects. Project names and item IDs
contain lowercase letters, digits, hyphens or underscores, start with a letter
or digit and have at most 80 characters. Reserved Windows filenames are rejected
on every platform.

**Read `PROJECT.md` or `projects show NAME` before writing files.** Their
`naming_rules` explain the contract:

- `project.json` is the source of truth. Each item's `file` and optional
  `preview` are exact paths relative to the project, using forward slashes.
  The page does not scan the output folder or guess a preview filename.
- Imported sources use `assets/ITEM_ID/vVERSION/source.EXT`. Imported browser
  previews use `assets/ITEM_ID/vVERSION/preview.EXT`. Versions start at 1;
  metadata edits also advance the version. Existing versions remain on disk.
- `generate --project NAME` chooses and registers its output path automatically.
  Batch outputs append `-001`, `-002`, etc. The response gives each item's ID
  and page/manifest paths. A custom `--out` is resolved relative to the project
  and must stay inside it. Project generation rejects `--overwrite`.
- Files created separately must be registered using `projects add` or
  `projects update`. Their `--file` and `--preview` arguments are relative to
  the command's working directory. The CLI copies them into the naming layout;
  files already at the exact canonical destination are registered in place.
- Registration refreshes `index.html` and `PROJECT.md`. Use `projects render`
  to recover/rebuild the page without modifying the manifest.

Every new item has a stable ID. To replace an image while preserving that ID:

```sh
slopfab generate image --project launch-film --project-item ITEM_ID --prompt "The same city at sunrise"
```

This writes a new version, retains the old files, and saves the generation
arguments, working directory, resolved plan (including seed), and artifact
paths. To reuse a recipe, resolve relative model/reference paths against its
recorded working directory; use its resolved plan for actual settings. An agent
can vary the seed or prompt as requested. `--project-item` requires count 1 and
a matching media type. A concurrent project edit rejects replacement registration
instead of losing the newer edit; the generated output remains available for
manual registration.

JSON requests and queued jobs also support projects:

```json
{
  "schema_version": 1,
  "task": "image",
  "project": "launch-film",
  "project_item": "cover",
  "output_directory": "/work/output",
  "prompt": "A city at sunrise"
}
```

`project_item` and `output_directory` are optional. `validate`/`--dry-run`
resolves the project output without creating media or modifying the manifest.

## Browser previews

PNG images and WAV audio can be reviewed directly. Video generation produces
Y4M, which browsers cannot play. The agent should encode a separate MP4/WebM
preview with its external FFmpeg installation, including the generated WAV
soundtrack when present, then register that preview:

```sh
ffmpeg -i source.y4m -i source.wav -c:v libx264 -pix_fmt yuv420p -c:a aac preview.mp4
slopfab projects update launch-film VIDEO_ID --preview preview.mp4
```

For a silent video, omit the audio input. PPM images similarly need a PNG
preview. The page explicitly offers a "Create browser preview" action for
unsupported sources and explains playback failures. Slopfab does not invoke or
link FFmpeg. Browser codec availability still depends on the browser/OS.

## Video clips and edits

Register an assembled video and its clips. Supply each clip's generation
request or prompt so the agent can regenerate it:

```sh
slopfab projects add launch-film --id film --file final.mp4 --title "Final film"
slopfab projects add launch-film --id opening --kind clip --parent film --position 0 --file opening.mp4 --request opening.json --title "Opening shot"
slopfab projects add launch-film --id closing --kind clip --parent film --position 1 --file closing.mp4 --prompt "City lights at night"
```

Clips are ordered by `position` (ties retain manifest order). Optional
`--start SECONDS --end SECONDS` trim the clip's **source**, not the assembled
video timeline. The preview player respects these bounds; the agent must also
apply them during assembly. Separate source clips or registered ranges are
necessary: Slopfab does not infer scene boundaries from a finished video.

```sh
slopfab projects remove launch-film opening
slopfab projects restore launch-film opening
slopfab projects update launch-film closing --position 0
slopfab projects update launch-film closing --file new-closing.mp4
```

Adding, changing, removing or restoring a clip marks its parent `needs_rebuild`.
The page warns that the assembled video is outdated. The generated agent
prompt requests rebuilding that video from its active clips in position order,
preserving the intended music/audio. The agent uses external editing tools,
then calls `projects update NAME VIDEO_ID --file NEW_VIDEO` to register the
assembly and clear the warning. Removing a whole video hides its clips from
the active page without deleting them. Removing/restoring items retains history
and media; there is no destructive project delete command.

Mutations are serialized with a project lock. Agents can pass
`--if-revision N` to add/update/remove/restore to reject edits from a stale
snapshot. `projects show` returns the current revision and removed items too.

## Verification

`projects_cli` tests folder creation, filenames, versions, clip changes,
revision conflicts, HTML escaping, path containment and generation planning
without a GPU. `cli_delivery` checks automatic registration of actual native
media and version replacement. The optional `tests/test_projects_page.cjs`
exercises video/audio/image playback, custom controls, prompt composition,
clipboard fallback and mobile layout in an installed Chrome with Playwright.
