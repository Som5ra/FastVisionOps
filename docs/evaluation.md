# NMSs Evaluation Report

## Executive summary

The repository now has one validated NumPy reference implementation, one
rebuildable C backend, deterministic output rules, and automated coverage for
bounding-box, mask, multiclass, empty-input, invalid-input, and batch behavior.

On the evaluation host, the native implementation was **4.87x to 17.16x
faster** than the NumPy reference for 250 to 2,500 boxes. Processing eight
images concurrently was **1.72x faster** than serial C execution in the recorded
run. Every native result was checked against the Python reference before its
timing was accepted.

## What was evaluated

| Area | Evaluation |
| --- | --- |
| Single-class bbox NMS | Known examples, stable ties, inclusive score threshold, both coordinate offsets |
| Multiclass bbox NMS | Class-aware and class-unaware behavior, global score ordering |
| Mask NMS | IoU, suppression, empty masks, multiclass output |
| Native backend | 128 randomized parameter combinations against NumPy |
| Batch execution | Serial and concurrent native outputs compared item by item |
| Defensive behavior | Invalid shapes, coordinates, thresholds, dtypes, and library paths |

The test suite contains 21 named tests. The randomized native equivalence test
combines:

- four input sizes: 0, 1, 32, and 257 boxes;
- both continuous (`offset=0`) and inclusive-pixel (`offset=1`) coordinates;
- four score thresholds: 0.0, 0.25, 0.8, and 1.0; and
- four IoU thresholds: 0.0, 0.3, 0.7, and 1.0.

## Performance methodology

The benchmark uses deterministic synthetic `xyxy` boxes:

- random seed: 42;
- image extent: 640 × 640;
- box sizes: uniformly sampled from 10 to 160;
- score threshold: 0.25;
- IoU threshold: 0.5;
- warm-up iterations: 2; and
- measured iterations: 9, reported as the median wall-clock duration.

Both measurements include public API validation and array preparation. The
script checks that C and NumPy return identical indices before measuring.
Native code is rebuilt from `nmss/csrc/nms.c` using:

```text
-O3 -std=c11 -DNDEBUG -fPIC -shared -lm
```

### Evaluation environment

| Component | Value |
| --- | --- |
| CPU | Intel Xeon Platinum 8573C, 9 available vCPUs |
| Architecture | x86_64 |
| OS | Linux 6.12.13, glibc 2.39 |
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| Compiler | GCC 13.3.0 |

### Recorded results

| Boxes | Boxes kept | NumPy (ms) | C (ms) | C speedup |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 178 | 4.745 | 0.277 | 17.16x |
| 1,000 | 607 | 28.493 | 3.437 | 8.29x |
| 2,500 | 1,284 | 96.306 | 19.763 | 4.87x |

| Batch | Boxes/image | Workers | Serial C (ms) | Parallel C (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1,000 | 8 | 30.243 | 17.619 | 1.72x |

Timings are host-dependent. The committed benchmark is the source of truth and
should be rerun on the deployment machine rather than treating these values as
a universal guarantee.

## Reproduction

From the repository root:

```bash
python -m nmss.build
python -m unittest discover -s tests -v
python -m benchmarks.benchmark_bbox
python -m benchmarks.benchmark_bbox --format json
```

The last two commands produce Markdown and machine-readable results,
respectively.

## Findings and trade-offs

1. **The original binary was not reproducible.** A checked-in `.so` is tied to
   an unknown compiler and ABI. It has been replaced with a source build.
2. **The original C and Python thresholds disagreed.** C used `>` while Python
   used `>=`. Both backends now retain scores exactly equal to the threshold.
3. **Unsigned coordinates could underflow.** The native backend now accepts
   `float64`, allowing negative and fractional coordinates safely.
4. **Output was not deterministic for score ties.** Both backends now prefer the
   lower original index when scores are equal.
5. **Batch parallelism helps when each item is large enough.** Small items can
   be dominated by thread scheduling, so `workers=1` remains available.

## Current limits

- The native backend currently targets Linux systems with GCC or Clang.
- Greedy NMS remains quadratic in the worst case.
- Mask NMS is vectorized NumPy only; there is no native mask backend yet.
- GPU backends, Soft-NMS, DIoU-NMS, and prebuilt wheels are outside this
  repository's current scope.
