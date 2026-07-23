"""Reproducible NumPy-versus-C bounding-box NMS benchmark."""

from __future__ import annotations

import argparse
import json
import os
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


def run_batch_benchmark(
    batch_size: int,
    boxes_per_image: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    workers: int,
):
    backend = CBackend()
    boxes_batch = []
    scores_batch = []
    for image_index in range(batch_size):
        boxes, scores = generate_inputs(boxes_per_image, seed + image_index)
        boxes_batch.append(boxes)
        scores_batch.append(scores[:, None])

    serial = backend.batch_multiclass_nms(
        boxes_batch,
        scores_batch,
        0.25,
        0.5,
        workers=1,
    )
    parallel = backend.batch_multiclass_nms(
        boxes_batch,
        scores_batch,
        0.25,
        0.5,
        workers=workers,
    )
    for serial_item, parallel_item in zip(serial, parallel):
        np.testing.assert_array_equal(serial_item[0], parallel_item[0])
        np.testing.assert_array_equal(serial_item[1], parallel_item[1])

    serial_ms = median_milliseconds(
        lambda: backend.batch_multiclass_nms(
            boxes_batch,
            scores_batch,
            0.25,
            0.5,
            workers=1,
        ),
        warmup=warmup,
        repeats=repeats,
    )
    parallel_ms = median_milliseconds(
        lambda: backend.batch_multiclass_nms(
            boxes_batch,
            scores_batch,
            0.25,
            0.5,
            workers=workers,
        ),
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "batch_size": batch_size,
        "boxes_per_image": boxes_per_image,
        "workers": workers,
        "serial_ms": serial_ms,
        "parallel_ms": parallel_ms,
        "speedup": serial_ms / parallel_ms,
    }


def render_markdown(results, batch_result) -> str:
    lines = [
        "| Boxes | Kept | NumPy (ms) | C (ms) | Speedup |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {boxes} | {kept} | {python_ms:.3f} | "
        "{native_ms:.3f} | {speedup:.2f}x |".format(**result)
        for result in results
    )
    lines.extend(
        [
            "",
            "| Batch | Boxes/image | Workers | Serial C (ms) | "
            "Parallel C (ms) | Speedup |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            "| {batch_size} | {boxes_per_image} | {workers} | "
            "{serial_ms:.3f} | {parallel_ms:.3f} | {speedup:.2f}x |".format(
                **batch_result
            ),
        ]
    )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[250, 1000, 2500])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-boxes", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
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
    if min(arguments.batch_size, arguments.batch_boxes, arguments.workers) < 1:
        parser.error("batch size, batch boxes, and workers must be positive")

    results = run_benchmark(
        arguments.sizes,
        seed=arguments.seed,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    batch_result = run_batch_benchmark(
        arguments.batch_size,
        arguments.batch_boxes,
        seed=arguments.seed + len(arguments.sizes),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        workers=arguments.workers,
    )
    if arguments.format == "json":
        print(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "numpy": np.__version__,
                    "results": results,
                    "batch_result": batch_result,
                },
                indent=2,
            )
        )
    else:
        print(render_markdown(results, batch_result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
