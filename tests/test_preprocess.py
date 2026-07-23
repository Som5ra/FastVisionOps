import unittest

import numpy as np

from fastvisionops.preprocess import (
    chw_channel_normalize,
    hwc_to_chw,
    hwc_to_chw_normalize,
    hwc_to_chw_normalize_batched,
)


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        self.mean = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        self.std = np.array([2.0, 4.0, 5.0], dtype=np.float32)

    def reference(self, image):
        normalized = (
            image.astype(np.float32) - self.mean[np.newaxis, np.newaxis, :]
        ) / self.std[np.newaxis, np.newaxis, :]
        return np.ascontiguousarray(normalized.transpose(2, 0, 1))

    def test_hwc_to_chw(self):
        actual = hwc_to_chw(self.image)
        expected = np.ascontiguousarray(self.image.transpose(2, 0, 1))
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(actual.flags.c_contiguous)

    def test_fused_normalization_matches_numpy(self):
        actual = hwc_to_chw_normalize(
            self.image,
            self.mean,
            self.std,
        )
        np.testing.assert_allclose(actual, self.reference(self.image))
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(actual.flags.c_contiguous)

    def test_chw_normalization_matches_fused(self):
        chw = hwc_to_chw(self.image)
        actual = chw_channel_normalize(chw, self.mean, self.std)
        np.testing.assert_allclose(actual, self.reference(self.image))

    def test_flip_rb_reverses_normalized_channels(self):
        expected = self.reference(self.image)[::-1]
        actual = hwc_to_chw_normalize(
            self.image,
            self.mean,
            self.std,
            flip_rb=True,
        )
        np.testing.assert_allclose(actual, expected)

    def test_batch_matches_individual_calls(self):
        batch = np.stack([self.image, self.image + 1])
        actual = hwc_to_chw_normalize_batched(
            batch,
            self.mean,
            self.std,
        )
        expected = np.stack(
            [
                hwc_to_chw_normalize(image, self.mean, self.std)
                for image in batch
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_non_contiguous_input_is_supported(self):
        image = self.image[:, ::-1, :]
        self.assertFalse(image.flags.c_contiguous)
        actual = hwc_to_chw_normalize(image, self.mean, self.std)
        np.testing.assert_allclose(actual, self.reference(image))

    def test_empty_batch(self):
        batch = np.empty((0, 4, 5, 3), dtype=np.uint8)
        actual = hwc_to_chw_normalize_batched(
            batch,
            self.mean,
            self.std,
        )
        self.assertEqual(actual.shape, (0, 3, 4, 5))
        self.assertEqual(actual.dtype, np.float32)

    def test_invalid_inputs_are_rejected(self):
        invalid_cases = [
            lambda: hwc_to_chw(self.image.astype(np.float32)),
            lambda: hwc_to_chw(self.image[0]),
            lambda: hwc_to_chw(self.image[:, :, :2], flip_rb=True),
            lambda: hwc_to_chw_normalize(
                self.image,
                self.mean[:2],
                self.std,
            ),
            lambda: hwc_to_chw_normalize(
                self.image,
                self.mean,
                [1.0, 0.0, 1.0],
            ),
            lambda: hwc_to_chw_normalize(
                self.image,
                [0.0, np.nan, 0.0],
                self.std,
            ),
            lambda: hwc_to_chw_normalize(
                self.image,
                self.mean,
                self.std,
                flip_rb=1,
            ),
        ]
        for invalid_case in invalid_cases:
            with self.subTest(case=invalid_case):
                with self.assertRaises((TypeError, ValueError)):
                    invalid_case()


if __name__ == "__main__":
    unittest.main()
