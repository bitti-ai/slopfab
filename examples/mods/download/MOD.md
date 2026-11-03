---
name: download
description: List MiniMax H3 model URLs and download explicitly selected files.
version: "1.0"
command: download
script: main.js
---
# Model downloads

Run `slopfab download minimaxh3` to list the bundled catalog without network
access. Select checkpoints appropriate to the model, backend, available memory
and intended task; this catalog is a set of options, not a requirement to fetch
every file. Review the source repository's model license before use.

Run `slopfab download minimaxh3 FILE_ID DESTINATION` to download one selected file.
The destination must not exist. Model files are large; downloads allow up to two
hours and do not resume partial transfers. Register the containing weights root
with `slopfab settings set weight_paths PATH` after placing files in the desired
`transformer/`, `text_encoder/` and `vae/` folders.

Add `--agent` for the normal agent response. Edit `catalog.json` to add more
models or URLs. This example does not automatically alter Slopfab settings.
