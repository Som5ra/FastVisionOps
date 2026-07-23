from pathlib import Path
import runpy
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LegacyAdapterTests(unittest.TestCase):
    def test_all_legacy_adapters_import(self):
        adapters = {
            "legacy/bbox-nms/nms.py": "nms_cpu",
            "legacy/bbox-nms-c-version/nms.py": "nms_cpu",
            "legacy/bbox-nms-c-version/batch_parallel_nms.py": (
                "Batch_Parallel_Nms"
            ),
            "legacy/mask-nms/mask_nms.py": "mask_nms_cpu",
        }
        for relative_path, expected_name in adapters.items():
            with self.subTest(path=relative_path):
                namespace = runpy.run_path(PROJECT_ROOT / relative_path)
                self.assertIn(expected_name, namespace)

    def test_obsolete_top_level_directories_are_removed(self):
        for directory in ("bbox-nms", "bbox-nms-c-version", "mask-nms"):
            with self.subTest(directory=directory):
                self.assertFalse((PROJECT_ROOT / directory).exists())


if __name__ == "__main__":
    unittest.main()
