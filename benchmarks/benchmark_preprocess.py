"""Reproducible NumPy-versus-native image preprocessing benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

from fastvisionops.build import DEFAULT_OUTPUT, build_native_backend
from fastvisionops.native import NativeBackend
from fastvisionops.preprocess import hwc_to_chw_normalize_batched


def generate_inputs(batch: int, height: int, width: int, seed: int):
    generator = np.random.default_rng(seed)
    images = generator.integers(
        0,
        256,
        size=(batch, height, width, 3),
        dtype=np.uint8,
    )
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    return images, mean, std


def median_milliseconds(function, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        timings.append((time.perf_counter_ns() - started) / 1_000_000)
    return statistics.median(timings)


def run_benchmark(
    batches: list[int],
    *,
    height: int,
    width: int,
    seed: int,
    warmup: int,
    repeats: int,
    threads: int,
):
    if not DEFAULT_OUTPUT.is_file():
        build_native_backend()
    backend = NativeBackend()
    results = []
    for position, batch in enumerate(batches):
        images, mean, std = generate_inputs(
            batch,
            height,
            width,
            seed + position,
        )
        expected = hwc_to_chw_normalize_batched(images, mean, std)
        actual = backend.hwc_to_chw_normalize_batched(
            images,
            mean,
            std,
            threads=threads,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        numpy_ms = median_milliseconds(
            lambda: hwc_to_chw_normalize_batched(images, mean, std),
            warmup=warmup,
            repeats=repeats,
        )
        native_ms = median_milliseconds(
            lambda: backend.hwc_to_chw_normalize_batched(
                images,
                mean,
                std,
                threads=threads,
            ),
            warmup=warmup,
            repeats=repeats,
        )
        results.append(
            {
                "batch": batch,
                "shape": f"{height}x{width}x3",
                "threads": threads,
                "numpy_ms": numpy_ms,
                "native_ms": native_ms,
                "speedup": numpy_ms / native_ms,
            }
        )
    return results


def render_markdown(results) -> str:
    lines = [
        "| Batch | Image shape | Threads | NumPy (ms) | Native (ms) | Speedup |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {batch} | {shape} | {threads} | {numpy_ms:.3f} | "
        "{native_ms:.3f} | {speedup:.2f}x |".format(**result)
        for result in results
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--height", type=int, default=427)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    arguments = parser.parse_args(argv)
    if any(batch < 1 for batch in arguments.batches):
        parser.error("all batch sizes must be positive")
    if min(arguments.height, arguments.width, arguments.threads) < 1:
        parser.error("height, width, and threads must be positive")
    if arguments.warmup < 0 or arguments.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")

    results = run_benchmark(
        arguments.batches,
        height=arguments.height,
        width=arguments.width,
        seed=arguments.seed,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        threads=arguments.threads,
    )
    if arguments.format == "json":
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "cpu_count": os.cpu_count(),
                    "results": results,
                },
                indent=2,
            )
        )
    else:
        print(render_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
