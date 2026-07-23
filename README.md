# NMSs

[![CI](https://github.com/Som5ra/NMSs/actions/workflows/ci.yml/badge.svg)](https://github.com/Som5ra/NMSs/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/backend-NumPy%20%2B%20C-4D77CF.svg)](https://numpy.org/)

**A compact, deterministic non-maximum suppression toolkit for bounding boxes
and boolean masks.**

NMSs provides a validated NumPy reference implementation and an optional
rebuildable C backend. It supports single-class NMS, class-aware and
class-unaware multiclass NMS, boolean-mask NMS, and concurrent batch execution
without requiring a deep-learning framework.

## Why NMSs

- **Correct by construction:** shape, coordinate, dtype, finiteness, and
  threshold checks fail early with actionable messages.
- **Deterministic:** equal scores are resolved by original input index.
- **Coordinate-safe:** negative and fractional `xyxy` coordinates work in both
  Python and C.
- **Fast:** the recorded native speedup is **4.87x–17.16x** for 250–2,500
  boxes on the evaluation host.
- **Reproducible:** the C library is built from source instead of shipping an
  opaque platform-specific binary.
- **Framework-independent:** NumPy is the only runtime dependency.

## Installation

Clone the repository and install the package:

```bash
python -m pip install .
```

The NumPy bbox and mask implementations are immediately available. To compile
the optional native bbox backend on Linux:

```bash
python -m nmss.build
```

The native build requires GCC or Clang. Set `CC` or pass `--compiler` to select
a specific compiler.

## Quick start

### Bounding-box NMS

```python
import numpy as np

from nmss import nms

boxes = np.array(
    [
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 9.0, 9.0],
        [20.0, 20.0, 30.0, 30.0],
    ]
)
scores = np.array([0.90, 0.80, 0.70])

keep = nms(
    boxes,
    scores,
    score_threshold=0.50,
    iou_threshold=0.50,
)
# array([0, 2])
```

### Multiclass NMS

```python
from nmss import multiclass_nms

scores_by_class = np.array(
    [
        [0.90, 0.60],
        [0.80, 0.95],
        [0.70, 0.65],
    ]
)

box_indices, class_ids = multiclass_nms(
    boxes,
    scores_by_class,
    score_threshold=0.50,
    iou_threshold=0.50,
    class_aware=True,
    max_detections=100,
)
```

Class-aware mode suppresses boxes independently for every class. Class-unaware
mode first assigns each box to its highest-scoring class, then suppresses
across the combined set.

### Mask NMS

```python
from nmss import mask_nms

masks = np.zeros((3, 64, 64), dtype=bool)
masks[0, :20, :20] = True
masks[1, 2:18, 2:18] = True
masks[2, 40:, 40:] = True

keep = mask_nms(masks, scores, iou_threshold=0.50)
# array([0, 2])
```

Masks must use boolean dtype. They may have any spatial rank as long as their
shapes match.

## Native acceleration

Build once, then use the API-compatible C backend:

```python
from nmss.c_backend import CBackend

backend = CBackend()

keep = backend.nms(
    boxes,
    scores,
    score_threshold=0.50,
    iou_threshold=0.50,
)
```

For independent images, native calls can run concurrently because the C call
releases Python's global interpreter lock:

```python
results = backend.batch_multiclass_nms(
    boxes_batch,
    scores_batch,
    score_threshold=0.50,
    iou_threshold=0.50,
    workers=8,
)
```

Each result is an `(indices, class_ids)` tuple. Use `workers=1` when
deterministic single-thread execution or minimal scheduling overhead is more
important than batch throughput.

## Coordinate convention

Boxes use `xyxy` order. Choose the geometry explicitly:

| `offset` | Convention | Width |
| ---: | --- | --- |
| `0` | Continuous coordinates, default | `x2 - x1` |
| `1` | Inclusive integer pixel coordinates | `x2 - x1 + 1` |

All scores equal to `score_threshold` are retained. A candidate is suppressed
only when `IoU > iou_threshold`; equality is retained.

## Performance

The benchmark checks C output against NumPy before timing. It uses two warm-up
runs and the median of nine measured runs.

| Boxes | Boxes kept | NumPy (ms) | C (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 178 | 4.745 | 0.277 | **17.16x** |
| 1,000 | 607 | 28.493 | 3.437 | **8.29x** |
| 2,500 | 1,284 | 96.306 | 19.763 | **4.87x** |

Recorded on Linux x86_64 with Python 3.12.13, NumPy 2.3.5, GCC 13.3, and an
Intel Xeon Platinum 8573C host. Performance depends on hardware, box
distribution, suppression rate, compiler, and system load.

Reproduce the measurements:

```bash
python -m benchmarks.benchmark_bbox
python -m benchmarks.benchmark_bbox --format json
```

See the [evaluation report](docs/evaluation.md) for the complete methodology,
batch results, environment, and limitations.

## Validation

Run the complete suite:

```bash
python -m nmss.build
python -m unittest discover -s tests -v
```

Coverage includes:

- bbox and mask IoU behavior;
- class-aware and class-unaware suppression;
- score/IoU boundaries and coordinate offsets;
- empty inputs and stable score ties;
- malformed input rejection;
- randomized Python/C equivalence; and
- serial/concurrent batch equivalence.

CI executes the suite on Python 3.9, 3.12, and 3.13.

## API overview

| API | Purpose | Backend |
| --- | --- | --- |
| `nmss.nms` | Single-class bbox NMS | NumPy |
| `nmss.multiclass_nms` | Aware or unaware bbox NMS | NumPy |
| `nmss.mask_nms` | Single-class boolean-mask NMS | NumPy |
| `nmss.multiclass_mask_nms` | Class-aware boolean-mask NMS | NumPy |
| `nmss.c_backend.CBackend.nms` | Single-class bbox NMS | C |
| `CBackend.multiclass_nms` | Class-aware bbox NMS | C |
| `CBackend.batch_multiclass_nms` | Concurrent image batches | C |

## Repository layout

```text
nmss/                   Maintained Python package and C source
tests/                  Correctness and native-equivalence suite
benchmarks/             Reproducible performance runner
docs/evaluation.md      Methodology, evidence, and limitations
bbox-nms*/ mask-nms/    Backward-compatible import paths
.github/workflows/      Python-version CI matrix
```

The legacy modules retain their original public function names and use
`offset=1` to preserve the old inclusive-pixel behavior. New integrations should
import directly from `nmss`.

## Scope

NMSs currently focuses on greedy CPU NMS. GPU kernels, Soft-NMS, DIoU-NMS,
native mask kernels, and prebuilt platform wheels are intentionally left for
future releases.
