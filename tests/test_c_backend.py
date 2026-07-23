from pathlib import Path
import tempfile
import unittest

import numpy as np

from nmss.bbox import multiclass_nms_class_aware, nms as python_nms
from nmss.build import build_c_backend
from nmss.c_backend import CBackend


class CBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        library = Path(cls.temporary_directory.name) / "libnmss.so"
        build_c_backend(library)
        cls.backend = CBackend(library)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_randomized_equivalence(self):
        generator = np.random.default_rng(20260723)
        for count in (0, 1, 32, 257):
            starts = generator.uniform(-50, 500, size=(count, 2))
            sizes = generator.uniform(0, 120, size=(count, 2))
            boxes = np.column_stack((starts, starts + sizes))
            scores = generator.random(count)
            for offset in (0.0, 1.0):
                for score_threshold in (0.0, 0.25, 0.8, 1.0):
                    for iou_threshold in (0.0, 0.3, 0.7, 1.0):
                        with self.subTest(
                            count=count,
                            offset=offset,
                            score_threshold=score_threshold,
                            iou_threshold=iou_threshold,
                        ):
                            expected = python_nms(
                                boxes,
                                scores,
                                score_threshold,
                                iou_threshold,
                                offset=offset,
                            )
                            actual = self.backend.nms(
                                boxes,
                                scores,
                                score_threshold,
                                iou_threshold,
                                offset=offset,
                            )
                            np.testing.assert_array_equal(actual, expected)

    def test_multiclass_equivalence(self):
        generator = np.random.default_rng(7)
        starts = generator.uniform(0, 500, size=(128, 2))
        boxes = np.column_stack(
            (starts, starts + generator.uniform(1, 100, size=(128, 2)))
        )
        scores = generator.random((128, 5))
        expected = multiclass_nms_class_aware(boxes, scores, 0.3, 0.5)
        actual = self.backend.multiclass_nms(boxes, scores, 0.3, 0.5)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_parallel_batch_matches_serial_batch(self):
        generator = np.random.default_rng(11)
        boxes_batch = []
        scores_batch = []
        for count in (20, 31, 42, 53):
            starts = generator.uniform(0, 200, size=(count, 2))
            boxes_batch.append(
                np.column_stack(
                    (
                        starts,
                        starts + generator.uniform(1, 50, size=(count, 2)),
                    )
                )
            )
            scores_batch.append(generator.random((count, 3)))
        serial = self.backend.batch_multiclass_nms(
            boxes_batch,
            scores_batch,
            workers=1,
        )
        parallel = self.backend.batch_multiclass_nms(
            boxes_batch,
            scores_batch,
            workers=4,
        )
        for serial_item, parallel_item in zip(serial, parallel):
            np.testing.assert_array_equal(serial_item[0], parallel_item[0])
            np.testing.assert_array_equal(serial_item[1], parallel_item[1])

    def test_workers_require_a_positive_integer(self):
        boxes = [np.array([[0.0, 0.0, 1.0, 1.0]])]
        scores = [np.array([[1.0]])]
        for workers in (True, 1.5, 0):
            with self.subTest(workers=workers):
                with self.assertRaisesRegex(ValueError, "workers"):
                    self.backend.batch_multiclass_nms(
                        boxes,
                        scores,
                        workers=workers,
                    )
        with self.assertRaisesRegex(ValueError, "workers"):
            self.backend.batch_multiclass_nms([], [], workers=0)

    def test_empty_batch_still_validates_nms_arguments(self):
        invalid_cases = [
            {"score_threshold": -0.1},
            {"iou_threshold": 1.1},
            {"offset": 0.5},
            {"max_detections": 1.5},
        ]
        for arguments in invalid_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    self.backend.batch_multiclass_nms(
                        [],
                        [],
                        **arguments,
                    )

    def test_missing_library_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "fastvisionops.build"):
            CBackend("/definitely/missing/libnmss.so")


if __name__ == "__main__":
    unittest.main()
