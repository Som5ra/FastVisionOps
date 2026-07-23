"""Build the optional FastVisionOps native backend."""

from __future__ import annotations

import argparse
from importlib.machinery import EXTENSION_SUFFIXES
import os
from pathlib import Path
import shutil
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / "csrc" / "vision_ops.c"
SOURCE_OUTPUT = PACKAGE_ROOT / "lib" / "libfastvisionops.so"


def _default_output() -> Path:
    """Prefer an installed wheel library, then the source-tree build."""
    for suffix in EXTENSION_SUFFIXES:
        installed_library = PACKAGE_ROOT / f"_native{suffix}"
        if installed_library.is_file():
            return installed_library
    return SOURCE_OUTPUT


DEFAULT_OUTPUT = _default_output()


def _compile(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True)


def build_native_backend(
    output: str | os.PathLike[str] | None = None,
    *,
    compiler: str | None = None,
    openmp: bool = True,
) -> Path:
    """Compile the shared C library and return its path.

    OpenMP is attempted by default. If the compiler does not support it, the
    same source is rebuilt as a portable single-threaded library.
    """
    output_path = Path(output).resolve() if output else DEFAULT_OUTPUT
    compiler = compiler or os.environ.get("CC", "cc")
    if shutil.which(compiler) is None:
        raise RuntimeError(
            f"C compiler {compiler!r} was not found; install GCC or Clang "
            "or set the CC environment variable"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_command = [
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
    command = base_command[:1] + (["-fopenmp"] if openmp else []) + base_command[1:]
    result = _compile(command)
    if result.returncode and openmp:
        result = _compile(base_command)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"native backend build failed: {detail}")
    return output_path


build_c_backend = build_native_backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile the optional FastVisionOps C backend."
    )
    parser.add_argument("--output", help="custom output library path")
    parser.add_argument("--compiler", help="C compiler executable")
    parser.add_argument(
        "--no-openmp",
        action="store_true",
        help="build a portable single-threaded backend",
    )
    arguments = parser.parse_args(argv)
    try:
        output = build_native_backend(
            arguments.output,
            compiler=arguments.compiler,
            openmp=not arguments.no_openmp,
        )
    except RuntimeError as error:
        parser.exit(1, f"error: {error}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
