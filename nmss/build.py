"""Backward-compatible native build API.

Use :mod:`fastvisionops.build` in new code.
"""

from fastvisionops.build import (
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
