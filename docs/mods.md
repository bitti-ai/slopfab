# Mods

A mod is a shareable Markdown prompt, optionally paired with a JavaScript module
that registers a new CLI command. Install one by copying its folder into a `mods/`
directory. Slopfab indexes the metadata and returns the prompt on demand; the
calling agent can follow its instructions or invoke its executable command.

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
| `command` | Optional CLI command name, using the same syntax as `name`; requires `script`. Cannot replace a built-in command. |
| `script` | Entry module path relative to the mod folder; requires `command`. Must resolve to an existing file inside that folder. |

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

## Executable commands

Add `command` and `script` to the front matter to register a command immediately:

```markdown
---
name: download
description: List model URLs and download selected files.
command: download
script: main.js
---
# Downloads
Run `slopfab download minimaxh3` to list available files.
```

`main.js` uses the [embedded workflow API](scripting.md), with an optional default
export receiving arguments. No external JavaScript interpreter is needed.
`slopfab help`, `capabilities` and `mods list` expose the command.
`slopfab help download` and `slopfab download --help` show the prompt and entry
path without executing code. Indexing and `mods load` also never execute code.
Scripts run only when their command is invoked.

Copy [examples/mods/download](../examples/mods/download/MOD.md) into a discovered
`mods/download/` folder, including `main.js` and `catalog.json`. From a source
checkout, this installation can itself use embedded JavaScript:

```sh
slopfab run --eval 'slopfab.fs.mkdir("mods"); slopfab.fs.copy("examples/mods/download", "mods/download", {recursive:true});'
slopfab download
slopfab download minimaxh3
slopfab download minimaxh3 audio-vae weights/vae/minimax_h3_audio_vae_fp32.safetensors
```

The last command downloads a large model file. The catalog includes Slopus's
bundled generator variants, shared components, LoRAs and upscalers, plus
[Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
for transcription. Run `slopfab download whisper` to see its file ID and URL.
See the [download mod](../examples/mods/download/MOD.md) for groups and companion
requirements; edit the catalog to add models without recompiling Slopfab.
Release packages install the download and licenses mods in `mods/` beside both
executables. `slopfab licenses` prints the complete bundled license and notice
texts offline, labeled by their source paths. Its readable output goes to stderr;
stdout contains a structured list of the sources. Other examples remain in the
source repository and are not included in release packages.

The invocation's working directory remains unchanged; use
`slopfab.scriptDirectory` to locate companion assets. Use `--` before arguments
that resemble global Slopfab flags, for example `slopfab download -- --help`
passes a literal `--help` argument to the script. Executable mods are trusted
code with the same access as the invoking user. Their entry path cannot escape
the mod folder, but this is not a sandbox for the code itself.

Disabled mods cannot run. Multiple enabled mods claiming the same command report
`MOD_COMMAND_AMBIGUOUS` (exit 3 when invoked); no command wins by search order.
Disable or rename one to resolve the conflict. Invalid metadata is reported by
`mods list` and never registered. Prompt-only mods continue to work unchanged.
