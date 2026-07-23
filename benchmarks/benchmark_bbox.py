"""Reproducible NumPy-versus-C bounding-box NMS benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import numpy as np

from nmss.bbox import nms as python_nms
from nmss.build import DEFAULT_OUTPUT, build_c_backend
from nmss.c_backend import CBackend


def generate_inputs(count: int, seed: int):
    generator = np.random.default_rng(seed)
    centers = generator.uniform(0, 640, size=(count, 2))
    sizes = generator.uniform(10, 160, size=(count, 2))
    boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2))
    scores = generator.random(count)
    return boxes, scores


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
    sizes: list[int],
    *,
    seed: int,
    warmup: int,
    repeats: int,
):
    if not DEFAULT_OUTPUT.is_file():
        build_c_backend()
    backend = CBackend()
    results = []
    for position, count in enumerate(sizes):
        boxes, scores = generate_inputs(count, seed + position)
        expected = python_nms(boxes, scores, 0.25, 0.5)
        actual = backend.nms(boxes, scores, 0.25, 0.5)
        np.testing.assert_array_equal(actual, expected)
        python_ms = median_milliseconds(
            lambda: python_nms(boxes, scores, 0.25, 0.5),
            warmup=warmup,
            repeats=repeats,
        )
        native_ms = median_milliseconds(
            lambda: backend.nms(boxes, scores, 0.25, 0.5),
            warmup=warmup,
            repeats=repeats,
        )
        results.append(
            {
                "boxes": count,
                "kept": len(expected),
                "python_ms": python_ms,
                "native_ms": native_ms,
                "speedup": python_ms / native_ms,
            }
        )
    return results


def render_markdown(results) -> str:
    lines = [
        "| Boxes | Kept | NumPy (ms) | C (ms) | Speedup |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {boxes} | {kept} | {python_ms:.3f} | "
        "{native_ms:.3f} | {speedup:.2f}x |".format(**result)
        for result in results
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 1000, 2500])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    arguments = parser.parse_args(argv)
    if any(size < 1 for size in arguments.sizes):
        parser.error("all sizes must be positive")
    if arguments.warmup < 0 or arguments.repeats < 1:
        parser.error("warmup must be non-negative and repeats must be positive")

    results = run_benchmark(
        arguments.sizes,
        seed=arguments.seed,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    if arguments.format == "json":
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
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
