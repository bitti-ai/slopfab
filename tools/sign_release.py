"""Create an Ed25519 release key or sign a slopfab ZIP (requires cryptography)."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def message(version, name, size, digest):
    return f"slopfab-update-v1\nbitti-ai/slopfab\n{version}\n{name}\n{size}\n{digest}\n".encode("ascii")


def generate(private_path, public_path):
    # Create a dedicated directory before writing any secret bytes.
    private_path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    if os.name == "nt":
        who = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"], text=True)
        sid = re.search(r"S-1-[0-9-]+", who)[0]
        subprocess.run(["icacls", str(private_path.parent), "/inheritance:r", "/grant:r",
                        f"*{sid}:(OI)(CI)F"], check=True, capture_output=True)
    key = Ed25519PrivateKey.generate()
    fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                    serialization.NoEncryption()))
    public_path.parent.mkdir(parents=True, exist_ok=True)
    with public_path.open("x", encoding="ascii") as out:
        out.write(key.public_key().public_bytes(serialization.Encoding.Raw,
                                               serialization.PublicFormat.Raw).hex() + "\n")
    print(f"Private key saved to {private_path}; public key saved to {public_path}")


def sign(package, private_path, public_path):
    match = re.fullmatch(r"slopfab-((?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))-windows-linux-x64\.zip", package.name)
    if not match:
        raise ValueError("Expected slopfab-VERSION-windows-linux-x64.zip")
    key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("An Ed25519 private key is required")
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if public.hex() != public_path.read_text(encoding="ascii").strip():
        raise ValueError("Signing key does not match the embedded public key")
    digest = hashlib.sha256()
    size = 0
    with package.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if not 0 < size <= 512 * 1024 * 1024:
        raise ValueError("Package exceeds updater size limit")
    signature = key.sign(message(match[1], package.name, size, digest.hexdigest()))
    destination = package.with_name(package.name + ".sig")
    with destination.open("xb") as out:
        out.write(signature)
    print(f"Signature written to {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["keygen", "sign"])
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--public-key", type=Path, default=Path(__file__).resolve().parents[1] / "resources/update_public_key.hex")
    parser.add_argument("--package", type=Path)
    args = parser.parse_args()
    if args.action == "keygen":
        generate(args.private_key, args.public_key)
    elif args.package:
        sign(args.package, args.private_key, args.public_key)
    else:
        parser.error("sign requires --package")
