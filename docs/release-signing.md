# Release signing

`slopfab update` requires a detached Ed25519 signature. Its trust anchor is
`resources/update_public_key.hex`, embedded at build time. It never accepts a
key from GitHub, a runtime option, or an environment variable. There is no
unsigned fallback. This is package signing, not Windows Authenticode signing.

The private key is stored outside the checkout at
`%LOCALAPPDATA%\slopfab\release-signing\release.private.pem`, in a directory
restricted to the generating Windows account. Back it up securely: losing it
prevents signing updates that existing installations will trust. Never add it
to the repository or release package. A GitHub compromise alone cannot produce
an accepted update without this key.

To sign a release, install Python's `cryptography` package in a dedicated
environment, build both binaries with the matching `VERSION`, and create the
bundle using `tools/package_bundle.py`. Then run:

```powershell
python tools/sign_release.py sign --private-key "$env:LOCALAPPDATA/slopfab/release-signing/release.private.pem" --package dist/slopfab-0.2.0-windows-linux-x64.zip
```

Upload both the ZIP and the generated `slopfab-0.2.0-windows-linux-x64.zip.sig`
to the corresponding `0.2.0` or `v0.2.0` stable GitHub release. Sign the exact
final ZIP: any subsequent change invalidates the signature. The signer refuses
an unrelated private key and refuses to overwrite an existing signature.

The signature asset is exactly 64 raw bytes. Ed25519 signs this ASCII message
with LF line endings and a final newline:

```text
slopfab-update-v1
bitti-ai/slopfab
VERSION
slopfab-VERSION-windows-linux-x64.zip
DECIMAL_BYTE_COUNT
LOWERCASE_SHA256
```

Binding the version, filename, size and hash prevents relabelling an old signed
package as a new version. The updater authenticates metadata before reporting
availability or downloading the ZIP, and hashes the downloaded ZIP before
extraction. The Windows helper independently checks the signature, hash and
version again. Missing, truncated, forged and wrong-key signatures fail closed.
Updates never downgrade the installed version. A compromised release service
can still withhold updates; signatures do not guarantee availability/freshness.

The initial signing-capable executable must be distributed through a trusted
channel: older checksum-only executables cannot authenticate this first upgrade.
To rotate keys, distribute a new executable embedding the new public key in a
package signed with the old key. New releases can then use the new private key;
clients that miss the transition require a trusted manual installation.

For a new independent distribution only, `tools/sign_release.py keygen
--private-key PATH --public-key PATH` creates a new key pair. Do not casually
replace this project's committed public key: deployed binaries retain their
existing trust anchor.
