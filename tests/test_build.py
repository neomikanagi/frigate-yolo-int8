import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import zipfile

import numpy as np

from validate_artifacts import MODELS, archives_for, expected_archives, validate_directory, validate_release
import yolo_common
from yolo_common import ANCHORS, RAW_OUTPUT, agreement, detections, gate

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def release(self):
        return {"assets": [{"name": name, "size": 1} for name in expected_archives()]}

    def test_complete_release(self):
        self.assertEqual(len(expected_archives()), 6 * len(MODELS))
        validate_release(self.release())

    def test_every_backend_per_model(self):
        self.assertEqual(archives_for("yolo26l"), [
            "frigate_yolo26l_fp16_320.zip",
            "frigate_yolo26l_int8_320.zip",
            "frigate_yolo26l_onnx_320.zip",
            "frigate_yolo26l_rk3568_int8_320.zip",
            "frigate_yolo26l_rk3576_int8_320.zip",
            "frigate_yolo26l_rk3588_int8_320.zip",
        ])

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
        for key in ("isDraft", "draft", "prerelease"):
            release = self.release()
            release[key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
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


def raw_output(objects):
    """Frigate yolo-generic tensor [1, 84, 2100] with (anchor, class, score, cx, cy, w, h) objects."""
    out = np.zeros(RAW_OUTPUT, np.float32)
    for anchor, cls, score, cx, cy, w, h in objects:
        out[0, :4, anchor] = (cx, cy, w, h)
        out[0, 4 + cls, anchor] = score
    return out


class QualityGateTests(unittest.TestCase):
    PERSON = (10, 0, 0.9, 100, 100, 50, 120)
    CAT = (20, 15, 0.8, 200, 220, 60, 40)

    def test_frigate_postprocess_keeps_one_box_per_object(self):
        duplicate = (11, 0, 0.85, 102, 101, 50, 118)
        dets = detections(raw_output([self.PERSON, duplicate, self.CAT, (30, 1, 0.3, 50, 50, 10, 10)]))
        self.assertEqual([(cls, round(score, 2)) for cls, score, *_ in dets], [(0, 0.9), (15, 0.8)])
        self.assertEqual([round(v) for v in dets[0][2:]], [75, 40, 125, 160])

    def test_identical_output_passes(self):
        ref = [raw_output([self.PERSON, self.CAT])]
        result = gate("same", ref, ref, min_recall=0.99, max_extra=0, min_cosine=0.99)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["extra"], 0)

    def test_zeroed_class_scores_fail(self):
        """The old RKNN INT8 build: boxes intact, all 80 class scores quantized to 0."""
        ref = [raw_output([self.PERSON, self.CAT])]
        broken = ref[0].copy()
        broken[0, 4:] = 0
        self.assertEqual(agreement(ref, [broken])["recall"], 0.0)
        with self.assertRaisesRegex(RuntimeError, "fails the gate"):
            gate("zeroed", ref, [broken], min_recall=0.5, max_extra=10, min_cosine=0.5)

    def test_flood_of_one_class_fails(self):
        """The HD 530 symptom: one class everywhere."""
        ref = [raw_output([self.PERSON])]
        flood = raw_output([self.PERSON] + [(100 + i, 1, 0.7, 10 + 15 * i, 300, 12, 12) for i in range(15)])
        with self.assertRaisesRegex(RuntimeError, "15 extra"):
            gate("flood", ref, [flood], min_recall=0.5, max_extra=10, min_cosine=0.0)

    def test_raw_shape_matches_frigate(self):
        self.assertEqual(ANCHORS, 2100)
        self.assertEqual(RAW_OUTPUT, [1, 84, 2100])


class FakeNode:
    def __init__(self, name):
        self.name = name


def fake_onnx(node_names):
    graph = types.SimpleNamespace(node=[FakeNode(n) for n in node_names], output=[types.SimpleNamespace(name="output0")])
    return types.SimpleNamespace(load=Mock(return_value=types.SimpleNamespace(graph=graph)))


class RKNNHybridTests(unittest.TestCase):
    def setUp(self):
        modules = {
            "ultralytics": types.SimpleNamespace(YOLO=Mock()),
            "rknn": types.ModuleType("rknn"),
            "rknn.api": types.SimpleNamespace(RKNN=Mock()),
        }
        spec = importlib.util.spec_from_file_location("build_rknn_test", ROOT / "build_rknn.py")
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", modules):
            spec.loader.exec_module(self.module)

    def head_nodes(self, head=23):
        return [f"/model.{head}/{b}.{i}/{b}.{i}.2/Conv" for b in ("cv2", "cv3") for i in range(3)] + ["/model.9/Add"]

    def test_fp16_tail_starts_at_the_raw_head_convolutions(self):
        with patch.dict("sys.modules", {"onnx": fake_onnx(self.head_nodes())}):
            pairs = self.module.head_tail("model.onnx")
        self.assertEqual(len(pairs), 6)
        self.assertIn(["/model.23/cv3.2/cv3.2.2/Conv_output_0", "output0"], pairs)
        self.assertTrue(all(end == "output0" for _, end in pairs))

    def test_head_index_follows_the_architecture(self):
        with patch.dict("sys.modules", {"onnx": fake_onnx(self.head_nodes(head=30))}):
            self.assertEqual(self.module.head_tail("model.onnx")[0][0], "/model.30/cv2.0/cv2.0.2/Conv_output_0")

    def test_missing_head_layer_fails(self):
        with patch.dict("sys.modules", {"onnx": fake_onnx(self.head_nodes()[:-2])}):
            with self.assertRaisesRegex(RuntimeError, "missing"):
                self.module.head_tail("model.onnx")

    def test_compiler_error_codes_fail(self):
        with self.assertRaisesRegex(RuntimeError, "export_rknn failed"):
            self.module.check(-1, "export_rknn")


class ZipTests(unittest.TestCase):
    def test_empty_model_file_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "m.rknn"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "nonempty"):
                yolo_common.write_zip(Path(tmp) / "out.zip", [empty])


if __name__ == "__main__":
    unittest.main()
