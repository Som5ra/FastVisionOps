# Native build

The supported build entry point is:

```bash
python -m fastvisionops.build
```

Use a different compiler or output path when required:

```bash
python -m fastvisionops.build --compiler clang --output /tmp/libfastvisionops.so
```

The builder compiles `fastvisionops/csrc/vision_ops.c` with optimized,
reproducible flags and reports compiler errors directly.
