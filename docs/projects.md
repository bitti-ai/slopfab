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
```

Open the returned `index_path` in a browser. The page shows images, video and
music with custom playback controls, filters, a clip sequence and generation
details. Select an item, choose an action, add direction and copy the resulting
prompt into the agent. The prompt identifies the project, item, version,
manifest, requested change and any parent video to rebuild. Refresh the page
after the agent updates the project. Page actions only compose prompts; they
never execute commands or silently change the manifest.

Use `--agent` for JSON responses and generation progress. Project management
does not require model weights or a GPU. The DLL and generation engine have no
project dependency.

## Output directory and filenames

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
