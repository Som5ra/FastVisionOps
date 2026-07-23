from pathlib import Path
import tempfile
import unittest

import numpy as np

from fastvisionops.build import build_native_backend
from fastvisionops.native import NativeBackend
from fastvisionops.preprocess import (
    hwc_to_chw_normalize,
    hwc_to_chw_normalize_batched,
)


class NativePreprocessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_directory = tempfile.TemporaryDirectory()
        library = (
            Path(cls.temporary_directory.name) / "libfastvisionops.so"
        )
        build_native_backend(library)
        cls.backend = NativeBackend(library)

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_randomized_single_image_equivalence(self):
        generator = np.random.default_rng(20260723)
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        for shape in ((1, 1, 3), (7, 11, 3), (64, 96, 3)):
            image = generator.integers(0, 256, shape, dtype=np.uint8)
            for flip_rb in (False, True):
                with self.subTest(shape=shape, flip_rb=flip_rb):
                    expected = hwc_to_chw_normalize(
                        image,
                        mean,
                        std,
                        flip_rb=flip_rb,
                    )
                    actual = self.backend.hwc_to_chw_normalize(
                        image,
                        mean,
                        std,
                        flip_rb=flip_rb,
                        threads=2,
                    )
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        rtol=1e-6,
                        atol=1e-6,
                    )

    def test_randomized_batch_equivalence(self):
        generator = np.random.default_rng(19)
        images = generator.integers(
            0,
            256,
            (5, 37, 53, 4),
            dtype=np.uint8,
        )
        mean = [-1.0, 20.0, 127.5, 250.0]
        std = [1.0, 17.0, 55.0, -2.0]
        expected = hwc_to_chw_normalize_batched(images, mean, std)
        actual = self.backend.hwc_to_chw_normalize_batched(
            images,
            mean,
            std,
            threads=3,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_noncontiguous_input_is_supported(self):
        generator = np.random.default_rng(23)
        image = generator.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        image = image[::2, ::2]
        self.assertFalse(image.flags.c_contiguous)
        expected = hwc_to_chw_normalize(image, [1, 2, 3], [4, 5, 6])
        actual = self.backend.hwc_to_chw_normalize(
            image,
            [1, 2, 3],
            [4, 5, 6],
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(actual.flags.c_contiguous)

    def test_empty_batch(self):
        images = np.empty((0, 20, 30, 3), dtype=np.uint8)
        actual = self.backend.hwc_to_chw_normalize_batched(
            images,
            [0, 0, 0],
            [1, 1, 1],
        )
        self.assertEqual(actual.shape, (0, 3, 20, 30))
        self.assertEqual(actual.dtype, np.float32)

    def test_invalid_threads_are_rejected(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "threads"):
            self.backend.hwc_to_chw_normalize(
                image,
                [0, 0, 0],
                [1, 1, 1],
                threads=-1,
            )

    def test_missing_library_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "fastvisionops.build"):
            NativeBackend("/definitely/missing/libfastvisionops.so")


if __name__ == "__main__":
    unittest.main()
