import unittest

import numpy as np

from nmss.mask import mask_iou, mask_nms, multiclass_mask_nms


class MaskNmsTests(unittest.TestCase):
    def setUp(self):
        self.masks = np.zeros((3, 8, 8), dtype=bool)
        self.masks[0, :4, :4] = True
        self.masks[1, :4, :4] = True
        self.masks[2, 5:, 5:] = True
        self.scores = np.array([0.9, 0.8, 0.7])

    def test_mask_iou(self):
        actual = mask_iou(self.masks[0], self.masks)
        np.testing.assert_allclose(actual, [1.0, 1.0, 0.0])

    def test_empty_masks_have_zero_iou(self):
        masks = np.zeros((2, 3, 3), dtype=bool)
        np.testing.assert_array_equal(mask_iou(masks[0], masks), [0.0, 0.0])

    def test_empty_comparison_batch_returns_empty_iou(self):
        mask = np.zeros((3, 3), dtype=bool)
        masks = np.empty((0, 3, 3), dtype=bool)
        actual = mask_iou(mask, masks)
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual(actual.size, 0)

    def test_mask_nms_suppresses_overlap(self):
        actual = mask_nms(self.masks, self.scores, 0.5, 0.5)
        np.testing.assert_array_equal(actual, [0, 2])

    def test_multiclass_mask_nms(self):
        scores = np.array(
            [[0.9, 0.6], [0.8, 0.95], [0.7, 0.65]]
        )
        indices, classes = multiclass_mask_nms(
            self.masks,
            scores,
            0.5,
            0.5,
        )
        np.testing.assert_array_equal(indices, [1, 0, 2, 2])
        np.testing.assert_array_equal(classes, [1, 0, 0, 1])

    def test_max_detections(self):
        actual = mask_nms(
            self.masks,
            self.scores,
            iou_threshold=1.0,
            max_detections=2,
        )
        np.testing.assert_array_equal(actual, [0, 1])

    def test_max_detections_requires_an_integer(self):
        with self.assertRaisesRegex(ValueError, "max_detections"):
            mask_nms(
                self.masks,
                self.scores,
                max_detections=1.5,
            )

    def test_non_boolean_masks_are_rejected(self):
        with self.assertRaises(TypeError):
            mask_nms(self.masks.astype(np.uint8), self.scores)

    def test_spatial_shape_must_match(self):
        with self.assertRaises(ValueError):
            mask_iou(np.zeros((4, 4), dtype=bool), self.masks)


if __name__ == "__main__":
    unittest.main()
