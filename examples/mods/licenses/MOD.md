---
name: licenses
description: Print the Slopfab license and all bundled third-party licenses and notices.
version: "1.0"
command: licenses
script: main.js
---
# Licenses

Run `slopfab licenses` to print all bundled license and notice texts offline.
Each text is labeled with its original source path. Like other script output,
the readable text goes to stderr; stdout contains a structured list of sources.
Use `slopfab licenses 2> licenses.txt` to save the readable text.

The package builder supplies `licenses.json` from the repository's `LICENSE`
and all license and notice files under `third_party/`. Keep it beside `main.js`
when sharing this mod. Model weights are distributed separately.
