# Native build

The supported build entry point is:

```bash
python -m nmss.build
```

Use a different compiler or output path when required:

```bash
python -m nmss.build --compiler clang --output /tmp/libnmss.so
```

The builder compiles `nmss/csrc/nms.c` with optimized, reproducible flags and
reports compiler errors directly.
