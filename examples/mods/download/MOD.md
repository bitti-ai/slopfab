---
name: download
description: List available model files and download selected weights.
version: "1.8"
command: download
script: main.js
---
# Model downloads

```sh
slopfab download                              # List model groups
slopfab download MODEL                        # List file IDs and URLs
slopfab download MODEL FILE_ID                # Download a selected file
slopfab download MODEL FILE_ID DESTINATION    # Override the destination
```

Listings are offline. Add `--agent` for structured responses.

LoRA URL entries include advisory `recommendedSampling`: denoising update count,
base model, strength, CFG, sources and native Slopfab `settings` JSON. Sigma
arrays are unshifted, include terminal zero, and have one more entry than updates;
video/audio shifts are applied separately. `requires` identifies recipe companions.
`basis: local-default` marks a fallback rather than an upstream recommendation.
Downloads do not apply these settings; legacy `stepOverride`/`samplingPreset`
fields are retained for older consumers and are not Slopfab CLI arguments.

Before selecting downloads, run `slopfab system devices` and check the current GPU's
available VRAM. Choose compatible model variants and companion files that fit,
leaving headroom for inference. Use catalog `minVramGb` values as per-file guidance,
not a guarantee that the full pipeline fits; missing values mean unknown.
Prefer supported lower-memory quantizations and smaller text encoders with their
matching ClipProj projections when needed. Do not download every variant.

Downloads default to `weights/` beside the running Slopfab executable, regardless
of the working directory. The filename comes from the URL or the catalog's
`filename` override. Listings report `weightsDirectory`. Explicit relative
destinations resolve against the working directory; existing files are never
overwritten. The mod does not change Slopfab settings.

Downloads use four connections, two retries per failed request and a two-hour
timeout. To resume an interrupted transfer, repeat the same command and retain
`DESTINATION.download/`. Resume requires server byte-range support and a matching
strong ETag; otherwise the transfer restarts. The final file appears only when
complete. Assembly temporarily needs space for both the parts and final file.
The persistent `DESTINATION.download.lock` file does not indicate an active transfer.

Edit `catalog.json` to add model groups, file IDs, URLs or filename overrides.
