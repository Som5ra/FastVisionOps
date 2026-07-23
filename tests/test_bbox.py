import unittest

import numpy as np

from nmss.bbox import (
    bbox_iou,
    multiclass_nms_class_aware,
    multiclass_nms_class_unaware,
    nms,
)


class BoundingBoxNmsTests(unittest.TestCase):
    def setUp(self):
        self.boxes = np.array(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 9.0, 9.0],
                [20.0, 20.0, 30.0, 30.0],
            ]
        )
        self.scores = np.array([0.9, 0.8, 0.7])

    def test_nms_suppresses_overlap(self):
        actual = nms(self.boxes, self.scores, 0.5, 0.5)
        np.testing.assert_array_equal(actual, [0, 2])

    def test_score_threshold_is_inclusive(self):
        actual = nms(self.boxes, self.scores, 0.8, 0.5)
        np.testing.assert_array_equal(actual, [0])

    def test_equal_scores_are_stable(self):
        boxes = np.array(
            [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
        )
        actual = nms(boxes, np.ones(3), 0.0, 0.5)
        np.testing.assert_array_equal(actual, [0, 1, 2])

    def test_coordinate_offset_changes_pixel_box_semantics(self):
        boxes = np.zeros((2, 4))
        scores = np.array([1.0, 0.5])
        np.testing.assert_array_equal(
            nms(boxes, scores, offset=0.0), [0, 1]
        )
        np.testing.assert_array_equal(
            nms(boxes, scores, offset=1.0), [0]
        )

    def test_max_detections_limits_output(self):
        actual = nms(
            self.boxes,
            self.scores,
            0.0,
            1.0,
            max_detections=2,
        )
        np.testing.assert_array_equal(actual, [0, 1])
        self.assertEqual(
            nms(self.boxes, self.scores, max_detections=0).size,
            0,
        )

    def test_bbox_iou_handles_zero_area(self):
        actual = bbox_iou(
            np.zeros(4),
            np.array([[0, 0, 0, 0], [0, 0, 1, 1]]),
        )
        np.testing.assert_array_equal(actual, [0.0, 0.0])

    def test_class_aware_nms_sorts_globally(self):
        scores = np.array(
            [[0.9, 0.6], [0.8, 0.95], [0.7, 0.65]]
        )
        indices, classes = multiclass_nms_class_aware(
            self.boxes,
            scores,
            0.5,
            0.5,
        )
        np.testing.assert_array_equal(indices, [1, 0, 2, 2])
        np.testing.assert_array_equal(classes, [1, 0, 0, 1])

    def test_class_unaware_assigns_best_class_first(self):
        scores = np.array(
            [[0.9, 0.1], [0.8, 0.95], [0.2, 0.7]]
        )
        indices, classes = multiclass_nms_class_unaware(
            self.boxes,
            scores,
            0.5,
            0.5,
        )
        np.testing.assert_array_equal(indices, [1, 2])
        np.testing.assert_array_equal(classes, [1, 1])

    def test_empty_input(self):
        actual = nms(np.empty((0, 4)), np.empty(0))
        self.assertEqual(actual.dtype, np.int64)
        self.assertEqual(actual.size, 0)
        indices, classes = multiclass_nms_class_aware(
            np.empty((0, 4)),
            np.empty((0, 2)),
        )
        self.assertEqual(indices.size, 0)
        self.assertEqual(classes.size, 0)

    def test_empty_class_unaware_input_still_validates_arguments(self):
        boxes = np.empty((0, 4))
        scores = np.empty((0, 2))
        invalid_cases = [
            {"score_threshold": -0.1},
            {"iou_threshold": 1.1},
            {"offset": 0.5},
            {"max_detections": -1},
        ]
        for arguments in invalid_cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    multiclass_nms_class_unaware(
                        boxes,
                        scores,
                        **arguments,
                    )

    def test_invalid_input_is_rejected(self):
        invalid_cases = [
            lambda: nms(np.zeros((2, 5)), np.ones(2)),
            lambda: nms(np.array([[1, 0, 0, 1]]), np.ones(1)),
            lambda: nms(np.full((1, 4), np.nan), np.ones(1)),
            lambda: nms(np.zeros((2, 4)), np.ones(1)),
            lambda: nms(np.zeros((1, 4)), np.ones(1), -0.1),
            lambda: nms(
                np.zeros((1, 4)),
                np.ones(1),
                iou_threshold=1.1,
            ),
            lambda: nms(
                np.zeros((1, 4)),
                np.ones(1),
                offset=0.5,
            ),
            lambda: nms(
                np.zeros((1, 4)),
                np.ones(1),
                max_detections=-1,
            ),
        ]
        for invalid_case in invalid_cases:
            with self.subTest(case=invalid_case):
                with self.assertRaises(ValueError):
                    invalid_case()


if __name__ == "__main__":
    unittest.main()
