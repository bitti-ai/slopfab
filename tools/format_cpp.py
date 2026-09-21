"""Format tracked, project-owned C++ and CUDA files with clang-format 19."""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check without editing files")
    parser.add_argument("--clang-format", default="clang-format", help="Formatter executable")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        version = subprocess.check_output([args.clang_format, "--version"], text=True)
        if "clang-format version 19." not in version:
            parser.error("Use clang-format 19 for reproducible formatting")
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        extensions = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".cu", ".cuh", ".inl"}
        directories = {"src", "include", "tests", "tools", "cmake"}
        files = [
            name.decode("utf-8")
            for name in tracked.split(b"\0")
            if name
            and Path(name.decode("utf-8")).parts[0] in directories
            and Path(name.decode("utf-8")).suffix in extensions
        ]
        options = ["--dry-run", "--Werror"] if args.check else ["-i"]
        for offset in range(0, len(files), 40):
            subprocess.run(
                [args.clang_format, "--style=file", *options, *files[offset:offset + 40]],
                cwd=root,
                check=True,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{'Checked' if args.check else 'Formatted'} {len(files)} C++/CUDA files")


if __name__ == "__main__":
    main()
