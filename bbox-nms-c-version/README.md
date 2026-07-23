# Legacy C API

This directory preserves the original `Batch_Parallel_Nms` import path.
Maintained native code now lives in `nmss/csrc`, and the shared library is
rebuilt locally:

```bash
python -m nmss.build
```

Existing calls continue to work:

```python
from batch_parallel_nms import Batch_Parallel_Nms

backend = Batch_Parallel_Nms()
indices, class_ids = backend.nms(boxes, scores, 0.5, 0.5)
batch_indices, batch_class_ids = backend.batch_parallel_nms(
    boxes_batch,
    scores_batch,
    0.5,
    0.5,
)
```

New code should use `nmss.c_backend.CBackend`. See the
[root README](../README.md) and
[evaluation report](../docs/evaluation.md) for the current API and verified
benchmark.
