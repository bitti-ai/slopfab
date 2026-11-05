# slopfab

Generate videos, images and music through Claude Code, Codex, Hermes, Gemini CLI, OpenCode, GitHub Copilot CLI or your favourite agentic CLI. Supports local generation or remote generation over SSH. Runs natively on Windows and Linux with CUDA and Vulkan.

1. Extract the package.
2. Launch your favourite agent in that folder.
3. Tell it to use slopfab to generate video/image/music.

Ask your agent to create [mods](docs/mods.md): reusable workflows you can share with others for music videos, cinematic action sequences or animated short films.

[Documentation](docs/agent-cli.md)

To package existing Windows and Linux builds, run `package.cmd`. It creates ZIP
and tar.gz archives in `dist/`, including the ready-to-use download mod. The
package version is defined in `VERSION` (currently `0.1.0`). See
[packaging instructions](docs/remote.md#packaging-and-checks) for build locations
and options.

## Model licensing

MiniMax H3 model weights are licensed separately under the MiniMax H3 COMMUNITY LICENSE AGREEMENT.

Fibo 2.0 is released under CC BY-NC 4.0 for non-commercial use. Commercial use is subject to an agreement with BRIA.

YuE2 uses CC BY-NC 4.0; MiniMax Music3 uses Apache 2.0.
