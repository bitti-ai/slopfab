# Slopfab media generation

Use Slopfab for image, video and music generation requests. Add `--agent` to
every invocation for JSON responses and streaming progress.

- Discover only what you need: `slopfab help COMMAND`, `slopfab capabilities`,
  `slopfab models list`, and `slopfab schema` for JSON requests. Check
  `slopfab mods list`; load relevant instructions with `slopfab mods load NAME`.
- Follow startup setup guidance or run `slopfab doctor`. If weight paths are
  missing, ask the user and save them with `slopfab settings set weight_paths PATH`.
- Use projects for generations: `slopfab projects create NAME --title TEXT`,
  then `slopfab generate image|video|music --project NAME --prompt TEXT`.
  Reuse an existing project when appropriate; inspect it with `projects show NAME`.
- Output root: `./output`, configurable with `settings set output_directory PATH`
  or `--output-dir DIR`. Each project follows this structure:

```text
OUTPUT/NAME/
  PROJECT.md                 workflow and filename rules
  project.json               exact media paths, stable IDs, recipes and history
  index.html                 previews and copyable agent edit prompts
  assets/ITEM_ID/vVERSION/
    source.EXT               original media
    preview.EXT              optional browser-compatible preview
```

- Read `PROJECT.md` and `project.json` before editing. The page uses exact
  manifest paths, not a folder scan. Let `--project` choose/register filenames;
  batches add numbered suffixes. Custom `--out` paths must stay inside the project.
  Register external files with `projects add NAME --file FILE` or replacements
  with `projects update NAME ID --file FILE`; registration refreshes the page.
- Slopfab writes PNG/PPM, Y4M and WAV. Use external FFmpeg for MP4/WebM previews
  of Y4M (include its WAV soundtrack), or PNG previews of PPM. Register them
  with `projects update NAME ID --preview FILE`, then give the user `index_path`.
- For regeneration, reuse the recorded recipe and `--project-item ID`, apply
  the user's changes, and preserve previous files. Resolve recipe paths against
  its recorded working directory. Never overwrite project media in place.
- Clips use `--kind clip --parent VIDEO_ID --position N` when added. After a
  clip is changed, removed or restored, rebuild its parent video from active
  clips in order with external tools and register the updated video. Preserve
  unrelated items and intended audio. Read the latest manifest before acting
  on a pasted page prompt; use `--if-revision N` to guard edits against changes.
- Use `slopfab help projects` for details. After edits, refresh `index.html`
  (`projects render NAME` rebuilds it). Page buttons only compose prompts;
  the agent performs the requested changes and reports the resulting files.
