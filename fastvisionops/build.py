"""Public native build API.

The implementation lives under :mod:`fastvisionops.native` alongside the
backend and C source. This module preserves the original command and imports.
"""

from .native.build import (
    DEFAULT_OUTPUT,
    PACKAGE_ROOT,
    SOURCE,
    build_c_backend,
    build_native_backend,
    main,
)

__all__ = [
    "DEFAULT_OUTPUT",
    "PACKAGE_ROOT",
    "SOURCE",
    "build_c_backend",
    "build_native_backend",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
