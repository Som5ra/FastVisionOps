from pathlib import Path
import unittest

from fastvisionops import mask_nms as public_mask_nms
from fastvisionops import nms as public_nms
from fastvisionops.bbox import nms as bbox_compat_nms
from fastvisionops.build import SOURCE, build_native_backend
from fastvisionops.mask import mask_nms as mask_compat_nms
from fastvisionops.native import NativeBackend
from fastvisionops.native.backend import NativeBackend as BackendImplementation
from fastvisionops.native.build import (
    build_native_backend as native_build_native_backend,
)
from fastvisionops.postprocess import mask_nms as postprocess_mask_nms
from fastvisionops.postprocess import nms as postprocess_nms
from fastvisionops.postprocess.bbox import nms as bbox_implementation
from fastvisionops.postprocess.mask import mask_nms as mask_implementation
from fastvisionops.preprocess import hwc_to_chw_normalize
from fastvisionops.preprocess.numpy import (
    hwc_to_chw_normalize as numpy_hwc_to_chw_normalize,
)
from nmss.bbox import nms as nmss_nms
from nmss.mask import mask_nms as nmss_mask_nms


class PackageLayoutTests(unittest.TestCase):
    def test_bbox_import_paths_share_one_implementation(self):
        functions = (
            public_nms,
            bbox_compat_nms,
            postprocess_nms,
            bbox_implementation,
            nmss_nms,
        )
        self.assertTrue(all(function is bbox_implementation for function in functions))

    def test_mask_import_paths_share_one_implementation(self):
        functions = (
            public_mask_nms,
            mask_compat_nms,
            postprocess_mask_nms,
            mask_implementation,
            nmss_mask_nms,
        )
        self.assertTrue(all(function is mask_implementation for function in functions))

    def test_preprocess_and_native_packages_expose_implementations(self):
        self.assertIs(hwc_to_chw_normalize, numpy_hwc_to_chw_normalize)
        self.assertIs(NativeBackend, BackendImplementation)
        self.assertIs(build_native_backend, native_build_native_backend)

    def test_native_source_is_colocated_with_backend(self):
        expected_parent = Path("fastvisionops/native/csrc")
        self.assertEqual(SOURCE.parts[-3:-1], expected_parent.parts[-2:])
        self.assertTrue(SOURCE.is_file())


if __name__ == "__main__":
    unittest.main()
