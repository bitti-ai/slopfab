"""Cross-check Python signing against the native verifier using disposable test keys."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sign_release import sign

driver = str(Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="slopfab-signing-test-") as temporary:
    root = Path(temporary)
    key = Ed25519PrivateKey.generate()
    private, public = root / "test.private.pem", root / "public.hex"
    private.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                         serialization.NoEncryption()))
    public_hex = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    public.write_text(public_hex)
    package = root / "slopfab-0.2.0-windows-linux-x64.zip"
    package.write_bytes(b"a package fixture, never installed or signed with the production key")
    sign(package, private, public)
    signature = package.with_name(package.name + ".sig")
    import hashlib
    url = "https://github.com/bitti-ai/slopfab/releases/download/v0.2.0/" + package.name
    release = {"draft": False, "prerelease": False, "tag_name": "v0.2.0", "assets": [
        {"name": package.name, "size": package.stat().st_size, "digest": "sha256:" + hashlib.sha256(package.read_bytes()).hexdigest(),
         "browser_download_url": url},
        {"name": package.name + ".sig", "size": 64, "browser_download_url": url + ".sig"}]}
    manifest = root / "release.json"

    def verify(value, valid, operation="signed-release", trusted_key=public_hex):
        manifest.write_text(json.dumps(value))
        result = subprocess.run([driver, operation, str(manifest), str(signature), trusted_key],
                                capture_output=True, text=True, timeout=15)
        assert (result.returncode == 0) == valid, result.stderr

    verify(release, True)
    # A valid signature from another publisher must fail against the embedded trust anchor.
    verify(release, False, "trusted-release")
    verify(release, False, trusted_key="1" * 64)
    for field, value in [("size", package.stat().st_size + 1), ("digest", "sha256:" + "0" * 64)]:
        bad = copy.deepcopy(release)
        bad["assets"][0][field] = value
        verify(bad, False)
    relabelled = json.loads(json.dumps(release).replace("0.2.0", "0.3.0"))
    verify(relabelled, False)
    original = signature.read_bytes()
    signature.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    verify(release, False)
    public.write_text("1" * 64)
    try:
        sign(package, private, public)
        raise AssertionError("signer accepted an unrelated public key")
    except ValueError:
        pass
print("PASS: Python/native Ed25519 interoperability, metadata binding and pinned trust anchor")
