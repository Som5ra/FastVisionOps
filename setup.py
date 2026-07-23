"""Setuptools hook that bundles the ctypes native library in wheels."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_HELPER = PROJECT_ROOT / "fastvisionops" / "native" / "build.py"


def _load_build_helper():
    spec = importlib.util.spec_from_file_location(
        "_fastvisionops_build",
        BUILD_HELPER,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native build helper at {BUILD_HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CtypesBuildExt(build_ext):
    """Build the ctypes library at setuptools' platform-specific path."""

    def build_extension(self, extension: Extension) -> None:
        if extension.name != "fastvisionops._native":
            super().build_extension(extension)
            return
        output = Path(self.get_ext_fullpath(extension.name)).resolve()
        build_helper = _load_build_helper()
        build_helper.build_native_backend(output)


setup(
    ext_modules=[
        Extension(
            "fastvisionops._native",
            sources=["fastvisionops/native/csrc/vision_ops.c"],
        )
    ],
    cmdclass={"build_ext": CtypesBuildExt},
)
