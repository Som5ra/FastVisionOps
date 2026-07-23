# FastVisionOps Evaluation Report

## Executive summary

FastVisionOps unifies image preprocessing and non-maximum suppression behind
one validated NumPy reference layer and one rebuildable C backend. The current
suite has 37 named tests, including randomized native equivalence, malformed
input handling, noncontiguous images, empty batches, deterministic score ties,
and concurrent NMS execution.

On the evaluation host:

- fused preprocessing of one 427×640×3 image took **4.492 ms in NumPy** and
  **0.249 ms natively**, an **18.07x speedup**;
- a batch of 32 images took **139.105 ms in NumPy** and **9.758 ms natively**,
  a **14.25x speedup**;
- native bbox NMS was **4.40x to 17.66x faster** for 250 to 2,500 boxes; and
- eight 1,000-box images took **27.595 ms serially** and **9.995 ms with eight
  workers**, a **2.76x throughput speedup**.

Every native result was compared with its NumPy reference before timing.

## What was evaluated

| Area | Evidence |
| --- | --- |
| Image preprocessing | Exact NumPy reference, randomized shapes, RGB/BGR reversal, single and batched calls |
| Image memory behavior | C-contiguous output, noncontiguous input, empty batch |
| Preprocessing validation | uint8 dtype, dimensions, channel count, mean/std shape and finiteness, nonzero std, thread count |
| Single-class bbox NMS | Known examples, stable ties, inclusive score threshold, both coordinate offsets |
| Multiclass bbox NMS | Class-aware and class-unaware behavior, global score ordering |
| Mask NMS | IoU, suppression, empty comparison batches, empty masks, multiclass output |
| Native NMS | 128 randomized parameter combinations against NumPy |
| Batch NMS | Serial and concurrent native outputs compared item by item |
| Packaging | Source wheel build and bundled native C source |
| C quality | GCC build with `-Wall -Wextra -Werror` |

The randomized native NMS matrix combines four input sizes, two coordinate
offsets, four score thresholds, and four IoU thresholds:

$$4 \times 2 \times 4 \times 4 = 128$$

## Benchmark methodology

Both benchmark runners:

1. generate deterministic inputs from a fixed seed;
2. execute NumPy and native implementations;
3. assert equivalent output;
4. perform two untimed warm-up iterations; and
5. report the median of nine wall-clock measurements using
   `time.perf_counter_ns`.

The reported durations include public API validation, output allocation, and
necessary array preparation. They are operation latency, not kernel-only time.
No speedup assertion is used in CI because shared-runner timing thresholds are
inherently noisy; CI smoke-tests both runners, while correctness is enforced by
the test suite.

### Evaluation environment

| Component | Value |
| --- | --- |
| CPU | Intel Xeon Platinum 8573C, 9 available vCPUs |
| Architecture | x86_64 |
| OS | Linux 6.12.13, glibc 2.39 |
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| Compiler | GCC 13.3.0 |
| Native optimization | `-O3 -DNDEBUG`, OpenMP enabled |

The native source is compiled with:

```text
-O3 -std=c11 -DNDEBUG -fPIC -shared -fopenmp -lm
```

If OpenMP compilation fails, the builder retries without `-fopenmp`.

## Recorded preprocessing results

Configuration:

- input dtype and layout: contiguous uint8 NHWC;
- image shape: 427×640×3;
- batch sizes: 1, 8, and 32;
- native threads: 8;
- mean: `[123.675, 116.28, 103.53]`;
- std: `[58.395, 57.12, 57.375]`; and
- seed: 42, incremented once per batch-size case.

| Batch | NumPy median (ms) | Native median (ms) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 4.492 | 0.249 | 18.07x |
| 8 | 26.263 | 1.888 | 13.91x |
| 32 | 139.105 | 9.758 | 14.25x |

The benchmark measures the fused HWC-to-CHW conversion and channel
normalization path. This is the operation that avoids an intermediate
transposed array and benefits from native parallel execution.

## Recorded bbox NMS results

Configuration:

- random seed: 42;
- image extent: 640×640;
- box sizes: uniformly sampled from 10 to 160;
- score threshold: 0.25; and
- IoU threshold: 0.5.

| Boxes | Boxes kept | NumPy median (ms) | Native median (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 250 | 178 | 4.798 | 0.272 | 17.66x |
| 1,000 | 607 | 23.257 | 3.340 | 6.96x |
| 2,500 | 1,284 | 75.396 | 17.121 | 4.40x |

| Batch | Boxes/image | Workers | Serial native (ms) | Parallel native (ms) | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1,000 | 8 | 27.595 | 9.995 | 2.76x |

The declining single-image NMS speedup at larger input sizes is expected:
greedy NMS remains quadratic in the worst case, while Python/NumPy validation
and dispatch overhead matter proportionally less as the native comparison loop
grows.

## Reproduction

From the repository root:

```bash
python -m fastvisionops.build
python -m unittest discover -s tests -v
python -m benchmarks.benchmark_preprocess
python -m benchmarks.benchmark_bbox
```

Machine-readable runs:

```bash
python -m benchmarks.benchmark_preprocess --format json
python -m benchmarks.benchmark_bbox --format json
```

Useful benchmark controls include `--warmup`, `--repeats`, `--threads`,
`--batches`, `--sizes`, and `--workers`. Run on the deployment host for
capacity planning rather than treating the recorded values as universal.

## Improvements over the original repositories

1. **Unsafe allocation was removed.** FastPreProcess allocated arrays with
   `new[]` but released them with scalar `delete`, which is undefined behavior.
   FastVisionOps allocates output through NumPy and writes into owned buffers.
2. **Input contracts are explicit.** Shape, dtype, statistics, channel
   reversal, coordinates, scores, and thresholds are validated before native
   execution.
3. **Noncontiguous images are correct.** Inputs are made contiguous only when
   the native backend requires it.
4. **Nested OpenMP was removed.** The fused preprocessor uses one parallel loop
   across batch and spatial positions.
5. **The build is reproducible.** Hard-coded Python 3.9 paths, the broken
   pybind11 gitlink, unused OpenCV, and checked-in build output are not part of
   the combined package.
6. **NMS semantics match.** NumPy and C retain scores equal to the threshold,
   support negative/fractional coordinates, and resolve score ties identically.
7. **Measured time is reported.** Every speedup table includes NumPy and native
   milliseconds, not only a ratio.

## Limits and interpretation

- Timings are host-specific and sensitive to CPU frequency, memory bandwidth,
  compiler, process contention, input distribution, and suppression rate.
- The native build currently targets Unix-like systems with GCC or Clang.
- The OpenMP fallback remains correct but is single-threaded and will have a
  different performance profile.
- Greedy NMS is quadratic in the worst case.
- Mask NMS is vectorized NumPy only.
- GPU kernels, resize/color conversion, Soft-NMS, DIoU-NMS, and prebuilt wheels
  are outside this release.
