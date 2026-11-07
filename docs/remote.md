# Remote commands over SSH

Use the regular command with `--host NAME` or its alias `--remote NAME`:

```sh
slopfab generate --host gpu video --prompt "A cat playing a piano" --frames 22
slopfab generate --remote gpu image --model krea2 --prompt "A red apple"
slopfab devices --host gpu
slopfab inspect weights/transformer/model.safetensors --host gpu
```

`slopfab remote` manages the connection and the remote installation. It does not
contain a second set of generation or media commands.

## Set up a host

Install an OpenSSH client locally and enable an SSH server on the other computer.
Connect once with `ssh gpu` to verify its host key and key/agent authentication.
The SSH alias can specify the username, port, identity and jump host. Slopfab uses
`BatchMode=yes`, so passwords and host-key confirmations must be handled before
running it. It preserves your SSH host-key checking settings.

Create or check out the project on both computers. Models, reference files and
other inputs must already be present remotely. Then configure the remote project:

```sh
slopfab remote setup gpu --host artist@gpu-box --directory /srv/slopfab
```

For Windows OpenSSH:

```powershell
slopfab remote setup gpu-win --host artist@windows-box --directory 'D:\Projects\slopfab' --platform windows
```

`--host` in setup defaults to the profile name. `--directory` must be an existing
absolute remote working directory. The platform is inferred from that path;
`--platform linux|windows` overrides it. WSL uses `linux` and an SSH server inside
the distribution. Windows supports cmd.exe and PowerShell as the SSH login shell.
`--ssh PATH` optionally chooses the local OpenSSH executable.

The package contains `windows/slopfab.exe` and `linux/slopfab`. Slopfab finds this
package root beside its executable or one directory above it. Development builds
can specify a package root explicitly:

```sh
slopfab remote setup gpu --directory /srv/slopfab --binary-dir /work/slopfab/dist
```

Saved profiles are stored beside the local settings file in
`CONFIG_PATH.remotes.json`. `--config FILE` selects the local settings/profile
store; it is not forwarded to the remote computer. Passwords and keys are never
stored in these profiles.

```sh
slopfab remote list
slopfab remote show gpu
slopfab remote update gpu
slopfab remote remove gpu
```

Removing a profile only forgets the connection; remote files remain.

## Automatic executable updates

Setup installs the bundled executable for the remote operating system. Every
remote command checks its SHA-256 checksum and uploads the bundled copy only if
it is missing, different or corrupted. No existing slopfab executable is needed
for this bootstrap: Linux uses a POSIX shell and `sha256sum`; Windows uses
PowerShell and .NET binary streams. Linux executable permissions are set during
installation. Neither Python nor SCP/SFTP/rsync is required remotely.

Verified executables live under `REMOTE_PROJECT/.slopfab/bin/SHA256/`. Uploads
are written to temporary files and verified before publication. A new package
selects a new executable directory, leaving older versions available for running
jobs, including Windows processes that lock their executable. The original
checkout's executable is not overwritten. `remote update NAME` performs this
check immediately without running another command.

Only the executable and command metadata are uploaded. Sources, models, caches,
project manifests and reference media are not synchronized. Install GPU drivers,
CUDA/Vulkan dependencies and Linux system libraries on the remote machine as
usual. Setup does not install those system dependencies or download weights.

## Run regular commands

Generation, validation, transcription, frame extraction, Motions, upscaling,
encoding, inspection, comparison, tokenization, discovery, settings, project
management, scripts, mods and job commands use the same arguments remotely.
Paths and model settings are evaluated on the remote computer, relative to the
configured working directory where applicable.
`--jobs-dir` also selects a remote path. `--config` selects the local settings
file and its adjacent host profiles; the remote uses its own settings.

```sh
slopfab generate --host gpu music --prompt "Warm ambient synths" --duration 10
slopfab generate --host gpu image --prompt "Connection check" --dry-run
slopfab generate --host gpu --request settings/request.json
slopfab audio transcribe interview.wav --model whisper-small.safetensors --out transcript.srt --host gpu
slopfab video frames output/scene.y4m --out contact.png --grid 4x3 --host gpu
slopfab settings set weight_paths /models/weights --host gpu
slopfab projects create film --host gpu
slopfab run workflows/render.js --host gpu -- workflow-arguments
slopfab submit image --prompt "A red apple" --host gpu
slopfab status JOB_ID --host gpu
slopfab wait JOB_ID --host gpu
slopfab cancel JOB_ID --host gpu
```

Place remote flags before a command's `--` argument separator. Arguments after
that separator remain opaque script/generation arguments. Prompt values equal
to `--host` or `--remote` remain literal values. Long prompts and shell
metacharacters travel as JSON over SSH stdin rather than in shell command text.
Stdin-based inputs such as `--request -` and `run -` are unavailable over this
transport; use an existing remote file or `run --eval` instead.

Host setup, download recovery and the internal worker entry points are local
control operations. `projects open` launches a local browser and cannot be
routed to a remote host; use `projects show --host NAME` to inspect its metadata.
The C API remains local.

## Result downloads

Generated artifacts are downloaded automatically. This includes batch outputs,
video/audio sidecars, requested latents, frame images, transcripts and other
reported command outputs. A script can return an artifact descriptor such as
`{artifacts: [{path: "output/clip.mp4", media_type: "video/mp4"}]}` or an
`output` filename. Metadata-only commands return their response without copying
project/model files. There is no directory scanning or recursive copying.
For a custom workflow that does not report artifacts, request exact output files:

```sh
slopfab run render.js --host gpu --download output/final.mp4 --download output/score.wav
```

Use `--download-dir DIR` to change the local root from `output/remote`. Each
invocation has a unique run directory with `remote.json` and downloaded files.
Files from the same remote directory stay together, preserving video/audio
sidecars. Different remote directories receive separate numbered subdirectories.
Existing local files are never overwritten. Downloads are checked by byte count
and SHA-256 before publication; the destination filesystem must support hard
links (for example NTFS or ext4). Remote outputs are retained.

A failed download can be retried without rerunning the command:

```sh
slopfab fetch output/remote/RUN_ID/remote.json
```

Verified files are checked and skipped; incomplete files restart from the
beginning. Changed local files cause an error. Keep remote outputs until all
downloads have succeeded. Recovery also retries an unfinished local project
import. The current bundled remote executable is verified before further
transfers.

With `--agent`, responses include remote progress events and a final result or
error. Successful results preserve the regular command's response fields, with
downloaded file paths pointing locally. `remote_result` retains the original
server response, and `artifacts` and `receipt` describe local delivery.
Diagnostics stream on stderr. A connection loss during a synchronous command
cannot guarantee that remote inference stopped; inspect its process and outputs
before rerunning. Use submit/status/wait/cancel for remote queued jobs.

## Review projects on both computers

Create a local project and a remote project, then use both project flags:

```sh
slopfab projects create film
slopfab projects create film --host gpu
slopfab generate --host gpu video --project film --local-project film --prompt "A city at sunrise"
slopfab projects open film
```

`--project` selects the remote project; `--local-project` imports downloaded
primary media into the existing local review project and regenerates its page.
Each run creates new local items and records the host, command, remote working
directory, resolved plan and downloaded files. Sidecars and latents remain in
the download directory and are referenced by the recipe. Re-fetching a receipt
does not duplicate imported items. Remote IDs, edits and histories are not
merged into the local manifest. `--project-item` replaces only the remote item.

## Packaging and checks

Release archives always contain both platform binaries. `package.cmd` packages
the current `build/Release` Windows and `build/linux-cli` Linux outputs without
rebuilding. It requires Python 3.10+ and can be run from any working directory.
`VERSION` defines the build and package version, currently `0.1.0` (separate
from the C ABI version). Run `package.cmd --help` for options:

```bat
package.cmd
package.cmd --windows-dir dist/windows --linux-dir dist/linux
package.cmd --windows-dir "D:\builds\windows" --linux-dir "D:\builds\linux" --output-dir dist
```

The archives are `dist/slopfab-0.1.0-windows-linux-x64.zip` and `.tar.gz`.
Each extracts directly into the chosen folder, with exactly these top-level entries:

```text
slopfab.exe
slopfab
mods/
  download/
  licenses/
```

Run `slopfab.exe download` or `./slopfab download` after extraction.
`slopfab licenses` prints the Slopfab license and all bundled third-party licenses
and notices. The license texts live inside `mods/licenses/`; no documentation,
libraries, headers, manifests or checksum files are added to the archive.
`--version MAJOR.MINOR.PATCH` overrides the archive version for one invocation.

`package.sh` builds Linux and requires a prebuilt Windows directory.
`python tools/package_bundle.py` is the same packaging entry point as `package.cmd`.
Packaging uses the supplied binaries as-is; run `build.cmd` first if fresh builds
are needed, then select its `dist/windows` and `dist/linux` outputs. Both binaries
must support `slopfab run` and executable mods to use the bundled commands;
packaging does not upgrade older binaries. Missing
either executable is a packaging error. Native `cmake --install` and CI smoke
test artifacts are development installations, not distributable releases.

`remote_cli` exercises real native workers through a local SSH stand-in on
Windows and Linux: bootstrap uploads, binary changes/corruption, normal command
routing, artifact-only downloads, retries, shell quoting, checksums and project
imports. `package_bundle` checks the exact archive layout, both binaries, both
mods, complete license texts and Linux executable permissions. These checks need no GPU; live SSH/GPU validation
is separate.
