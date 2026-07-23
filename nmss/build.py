"""Build the optional native backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE = PACKAGE_ROOT / "csrc" / "nms.c"
DEFAULT_OUTPUT = PACKAGE_ROOT / "lib" / "libnmss.so"


def build_c_backend(
    output: str | os.PathLike[str] | None = None,
    *,
    compiler: str | None = None,
) -> Path:
    """Compile and return the path to the shared C library."""
    output_path = Path(output).resolve() if output else DEFAULT_OUTPUT
    compiler = compiler or os.environ.get("CC", "cc")
    if shutil.which(compiler) is None:
        raise RuntimeError(
            f"C compiler {compiler!r} was not found; install GCC or Clang "
            "or set the CC environment variable"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-O3",
        "-std=c11",
        "-DNDEBUG",
        "-fPIC",
        "-shared",
        str(SOURCE),
        "-lm",
        "-o",
        str(output_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"native backend build failed: {detail}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the optional nmss C backend."
    )
    parser.add_argument("--output", help="custom output library path")
    parser.add_argument("--compiler", help="C compiler executable")
    arguments = parser.parse_args(argv)
    try:
        output = build_c_backend(arguments.output, compiler=arguments.compiler)
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
