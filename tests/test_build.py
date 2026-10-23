import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

from validate_artifacts import (
    expected_archives, validate_directory, validate_release,
)

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def release(self):
        return {"assets": [{"name": name, "size": 1} for name in expected_archives()]}

    def test_complete_release(self):
        self.assertEqual(len(expected_archives()), 24)
        validate_release(self.release())

    def test_partial_release_is_rebuilt(self):
        release = self.release()
        release["assets"].pop()
        with self.assertRaisesRegex(ValueError, "missing 1"):
            validate_release(release)

    def test_empty_asset_is_not_complete(self):
        release = self.release()
        release["assets"][0]["size"] = 0
        with self.assertRaises(ValueError):
            validate_release(release)

    def test_draft_is_not_complete(self):
        release = self.release()
        release["isDraft"] = True
        with self.assertRaises(ValueError):
            validate_release(release)

    def make_archives(self, directory):
        for name, files in expected_archives().items():
            with zipfile.ZipFile(directory / name, "w") as archive:
                for filename in files:
                    archive.writestr(filename, b"model")

    def test_all_archives_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.make_archives(directory)
            validate_directory(directory)
            next(directory.glob("*.zip")).unlink()
            with self.assertRaisesRegex(ValueError, "Missing archive"):
                validate_directory(directory)

    def test_empty_zip_member_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.make_archives(directory)
            name, files = next(iter(expected_archives().items()))
            with zipfile.ZipFile(directory / name, "w") as archive:
                for filename in files:
                    archive.writestr(filename, b"")
            with self.assertRaisesRegex(ValueError, "nonempty model"):
                validate_directory(directory)

    def test_corrupt_zip_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.make_archives(directory)
            next(directory.glob("*.zip")).write_bytes(b"not a zip")
            with self.assertRaises(zipfile.BadZipFile):
                validate_directory(directory)


class RKNNTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        previous = Path.cwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, previous)
        image_dir = Path("datasets/coco128/images/train2017")
        image_dir.mkdir(parents=True)
        (image_dir / "test.jpg").write_bytes(b"calibration")
        self.rknn = Mock()
        for method in ("config", "load_onnx", "build", "export_rknn"):
            getattr(self.rknn, method).return_value = 0
        self.rknn.export_rknn.side_effect = lambda path: (Path(path).write_bytes(b"model"), 0)[1]
        self.yolo = Mock()
        self.yolo.export.return_value = "model.onnx"
        modules = {
            "ultralytics": types.SimpleNamespace(YOLO=Mock(return_value=self.yolo)),
            "rknn": types.ModuleType("rknn"),
            "rknn.api": types.SimpleNamespace(RKNN=Mock(return_value=self.rknn)),
        }
        spec = importlib.util.spec_from_file_location("rknn_build_test", ROOT / "build_yolo_rknn.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", modules), patch.dict(os.environ, {"MODEL_NAME": "yolo11s", "RKNN_TARGET": "rk3588"}):
            spec.loader.exec_module(self.module)

    def test_success_keeps_int8_and_disables_only_broken_fusions(self):
        self.module.main()
        options = self.rknn.config.call_args.kwargs
        self.assertEqual(options["disable_rules"], [
            "seperate_conv_rs_split",
            "fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa",
        ])
        self.assertEqual(options["target_platform"], "rk3588")
        self.assertTrue(self.rknn.build.call_args.kwargs["do_quantization"])
        self.rknn.release.assert_called_once()
        with zipfile.ZipFile(self.module.ZIP_NAME) as archive:
            self.assertEqual(archive.namelist(), ["yolo11s_rk3588_320.rknn"])

    def test_compiler_error_codes_fail_build(self):
        for method in ("config", "load_onnx", "build", "export_rknn"):
            with self.subTest(method=method):
                mock = getattr(self.rknn, method)
                effect = mock.side_effect
                mock.side_effect = None
                mock.return_value = -1
                with self.assertRaises(RuntimeError):
                    self.module.main()
                mock.return_value = 0
                mock.side_effect = effect
                self.assertFalse(Path(self.module.ZIP_NAME).exists())
        self.assertEqual(self.rknn.release.call_count, 4)

    def test_exception_releases_compiler(self):
        self.rknn.build.side_effect = ValueError("compiler crash")
        with self.assertRaisesRegex(ValueError, "compiler crash"):
            self.module.main()
        self.rknn.release.assert_called_once()

    def test_missing_or_empty_dataset_fails(self):
        Path("datasets/coco128/images/train2017/test.jpg").unlink()
        with self.assertRaisesRegex(RuntimeError, "calibration images"):
            self.module.main()
        self.rknn.config.assert_not_called()

    def test_missing_model_output_fails(self):
        self.rknn.export_rknn.side_effect = None
        with self.assertRaisesRegex(RuntimeError, "nonempty model"):
            self.module.main()
        self.rknn.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
