# Mods

A mod is a shareable Markdown prompt that tells an agent how to carry out a
workflow. Install one by copying its folder into a `mods/` directory. Slopfab
indexes the metadata and returns the prompt on demand; the calling agent follows
the instructions using its own tools.

```sh
slopfab mods list
slopfab mods load product-shot
slopfab mods list --all
```

Add `--agent` when calling from an agent. `mods list` (also `mods index` or simply
`mods`) returns active mods with names, descriptions, source paths and load
commands. It omits prompt bodies to keep context small. `mods load NAME` returns
the selected mod's complete prompt, metadata, file path and directory in a JSON
`mod` object. Without `--agent`, loading prints the original Markdown document.
Read that output into the agent's context before following the workflow.
Loading a mod does not itself run shell commands or generate media.

## Location and sharing

Slopfab scans the immediate subfolders of these directories:

- `mods/` beside the settings file (by default `%LOCALAPPDATA%/slopfab/mods/`
  on Windows or `$XDG_CONFIG_HOME/slopfab/mods/`, falling back to
  `~/.config/slopfab/mods/`, on Linux).
- `mods/` beside the executable, for portable installations.
- `mods/` in the current working directory, for project-specific workflows.

`--config FILE` or `SLOPFAB_CONFIG` also relocates the first directory.
The index reports its absolute search roots, including directories that do not
exist yet. Create the desired folder and copy a mod there. Share the entire
mod folder, including any companion documents or assets. Relative references
inside the prompt should be resolved against the returned mod `directory`.

Each invocation reads the current files; no refresh command or persistent index
is needed. Overlapping roots are deduplicated. Different files with the same mod
name are reported as ambiguous and excluded from the index until one is renamed
or removed. There is no silent precedence between global and project mods.

## MOD.md format

Create `mods/product-shot/MOD.md`:

```markdown
---
name: product-shot
description: >-
  Create a product image with consistent lighting and framing.
enabled: true
version: "1.0"
author: Example studio
---
# Product shot

1. Ask the user which product to depict and where to save the image if unspecified.
2. Run `slopfab models list` and select an available image model.
3. Describe the product on a plain background with soft studio lighting.
4. Use `slopfab help generate` to check options, then generate a PNG image.
5. Return the output path and the prompt used.

When calling slopfab from an agent, add its global `--agent` parameter.
```

`MOD.md` must be UTF-8, at most 1 MiB, with YAML front matter between two `---`
lines followed by a nonempty Markdown prompt. UTF-8 BOM and CRLF are accepted;
the prompt body is returned without rewriting its content. The filename is
`MOD.md`, not `.txt` or `.yaml`. Other files in the folder can be companion assets.

The metadata schema is deliberately small:

| Field | Meaning |
| --- | --- |
| `name` | Required; matches the folder name. Starts with `a-z`, then uses `a-z`, `0-9`, `-` or `_`; maximum 64 characters. |
| `description` | Required, nonempty text explaining when to use the mod. |
| `enabled` | Optional unquoted `true` or `false`; defaults to `true`. |
| `version` | Optional version text. |
| `author` | Optional attribution text. |

Supported YAML syntax is a flat mapping of scalar values, comments, plain or
single/double-quoted strings, and literal (`|`) or folded (`>`) block strings
with optional `-`/`+` chomping. Double-quoted strings use JSON-compatible escapes.
Nested mappings, lists, anchors, tags, unknown fields and duplicate fields are
rejected. Quote descriptions containing `: ` or a literal ` #`.

Set `enabled: false` to deactivate a mod. Disabled mods appear only with `--all`
and cannot be loaded. Invalid files and duplicate names appear in the index's
`errors` array (stderr in ordinary text mode); valid mods remain usable.
Loading a missing mod returns `MOD_NOT_FOUND` (exit 3), an ambiguous mod
`MOD_AMBIGUOUS` (exit 3), an invalid mod `INVALID_MOD` (exit 2), and a disabled
mod `MOD_DISABLED` (exit 4). Indexing itself succeeds when it can return a partial
index; inspect `errors` to detect problems.

Mods and settings commands remain available even when generation dependencies
or weight paths have not been configured. They do not affect DLL generation.
