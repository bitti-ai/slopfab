# Remote generation over SSH

`slopfab remote generate` runs generation in an existing project directory on
another computer and downloads its result artifacts. Both computers need
slopfab from a version that includes this command. The remote computer needs
the GPU backend, models, settings and reference inputs used by the request.
The local computer can use a CPU-only build.

Source code, model weights, input files, caches and whole project directories
are never synchronized. Only the JSON control request travels to the server;
progress, result metadata and generated artifacts travel back. No Python,
rsync, SCP, SFTP or separate server is required at runtime.

## SSH setup

Install an OpenSSH client locally and enable an SSH server on the generation
computer. Configure a host alias in your local SSH config, for example:

```sshconfig
Host gpu-linux
    HostName 192.168.1.50
    User artist
    IdentityFile ~/.ssh/id_ed25519
    Port 22
```

Connect once with `ssh gpu-linux` to verify the host key and authentication.
Slopfab uses `BatchMode=yes`: authenticate with a key or SSH agent, without
interactive password or host-key prompts. Ports, identities and jump hosts
come from your normal SSH configuration. See the
[OpenSSH configuration reference](https://man.openbsd.org/ssh_config.5).

Use `--platform posix` (the default) for a Linux or WSL SSH server. Use
`--platform windows` for Windows OpenSSH; its default shell can be cmd.exe or
PowerShell. Windows invokes the native worker through PowerShell with inherited
binary handles, preserving media bytes. See
[Windows OpenSSH setup](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh-server-configuration).
A WSL endpoint must connect to the SSH server inside the distribution.

`--directory` is an **absolute remote working directory**. All generation
paths, including `--out`, prompt files, JSON requests, models and references,
are evaluated there or according to the remote user's settings. Local settings
and local absolute paths are not translated or forwarded. Configure model
locations with `slopfab settings` on the remote computer.

`--executable` is the remote slopfab executable, resolved before changing into
the working directory. Use an absolute path, or omit it if `slopfab` is on the
remote SSH session's PATH. `--ssh` selects a local SSH client path.

## Generate video, images and music

Linux/WSL remote, from either a Windows or Linux client:

```sh
slopfab remote generate --host gpu-linux --directory /srv/slopfab --executable /srv/slopfab/build/slopfab -- video --prompt "A cat playing a piano" --frames 22 --seed 11
slopfab remote generate --host gpu-linux --directory /srv/slopfab --executable /srv/slopfab/build/slopfab -- image --model krea2 --prompt "A red apple"
slopfab remote generate --host gpu-linux --directory /srv/slopfab --executable /srv/slopfab/build/slopfab -- music --prompt "Warm ambient synths" --duration 10
```

Windows remote, with a `gpu-windows` SSH alias:

```powershell
slopfab remote generate --host gpu-windows --platform windows --directory 'D:\Projects\slopfab' --executable 'D:\Projects\slopfab\build\Release\slopfab.exe' -- video --prompt 'A cat playing a piano' --frames 22 --seed 11
```

The `--` separator is required. Options before it control SSH and local
delivery; arguments after it go to the remote `generate` command. Arguments
travel as JSON over SSH stdin, so quotes, Unicode, multiline prompts and shell
metacharacters remain literal generation arguments.

Use `--dry-run` after the separator to validate remotely without loading models:

```sh
slopfab remote generate --host gpu-linux --directory /srv/slopfab --executable /srv/slopfab/build/slopfab -- image --prompt "Connection check" --dry-run
```

`--request settings/request.json` reads that file **on the remote computer**.
Stdin requests (`--request -`) are unavailable because SSH stdin carries the
control protocol. Local `--config` selects local settings only.

## Downloads and projects on both computers

Each invocation gets `output/remote/RUN_ID/remote.json` locally. Choose another
download root with `--local-dir PATH`. Artifacts are grouped under numbered
subdirectories by remote parent directory, preserving video/audio sidecars
together and distinguishing duplicate basenames. Existing local files are
never overwritten. Byte counts and SHA-256 hashes are checked before publishing
downloads. Use a filesystem supporting hard links, such as NTFS or ext4, for
atomic publication without overwrites.

The downloaded files are exactly the generation result's artifacts, including
requested saved latents. No directory scanning or recursive copying occurs.
Native generation currently reports PNG/PPM, Y4M and WAV; this command does not
perform additional MP4/MP3 encoding. Remote outputs are retained.

To maintain review projects on both computers, create a project on each using
`slopfab projects create film`. They may have the same name and different
configured output roots. Then use both project options:

```sh
slopfab remote generate --host gpu-linux --directory /srv/slopfab --executable /srv/slopfab/build/slopfab --local-project film -- video --project film --prompt "A city at sunrise" --frames 22
slopfab projects open film
```

`--project film` after `--` registers generation remotely. `--local-project film`
before `--` imports downloaded primary media into the existing local project
and rebuilds its review page. The local recipe records the SSH host, remote
working directory, arguments, resolved plan and downloaded artifact paths.
Other artifacts, including a video's WAV soundtrack and saved latents, stay in
the download directory and are linked from that recipe. Importing copies the
primary media locally; it does not transfer it again over the network.

Each run creates new local items. Remote item IDs, histories and edits are not
merged into the local manifest; `--project-item` after `--` replaces only the
remote item. Existing local edits remain intact. Re-fetching the same receipt
does not create duplicate local items. See [projects](projects.md) for local
editing and browser previews.

## Retry and progress

If a download fails, keep its receipt and retry:

```sh
slopfab remote fetch output/remote/RUN_ID/remote.json
```

This downloads missing artifacts without rerunning generation. Verified files
are checked and skipped; incomplete files restart from the beginning. Changed
local files cause an error instead of being overwritten. Keep the original
remote outputs until all downloads succeed. The receipt also allows failed
local project registration to be retried after fixing the project issue.

With `--agent`, local output is NDJSON: `remote_started`, wrapped `remote_event`
progress/artifact events, `download` events, then one local `result` or `error`.
The successful result contains the receipt path, artifact paths relative to its
directory, original remote result and optional local project items.
Diagnostics stream on stderr.

Generation is synchronous. Ctrl+C stops the local SSH operation, but an SSH
disconnect cannot guarantee cancellation of inference remotely. If generation
fails or the connection drops before the artifact manifest is received, inspect
the remote process and outputs before starting another run. Remote job
submission, detached execution and automatic source/input sync are outside
this command's scope. The C API remains local.

## Development checks

`ctest --test-dir BUILD -C Release -R remote_cli --output-on-failure` uses a local
SSH stand-in. It exercises the actual remote shell command and native worker,
remote dry-run, every byte value in binary downloads, Unicode/quoted paths,
checksum failures, retries, traversal rejection, exit statuses and idempotent
local project imports. Synthetic generation events replace GPU inference;
testing against a real SSH server and model is separate.
