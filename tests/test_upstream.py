import tempfile
import unittest
from pathlib import Path

from upstream_watch import (
    PIPELINE, WEIGHTS, decide, fingerprint, is_watched, parse_asset_release, parse_fingerprint,
    pipeline_digest, release_body, releases_to_delete,
)
from validate_artifacts import MODELS

DOWNLOADS = '''
def attempt_download_asset(
    file: str | Path,
    repo: str = "ultralytics/assets",
    release: str = "v8.4.0",
    **kwargs,
) -> str:
    return ""
'''

BLOBS = {
    "ultralytics/engine/exporter.py": "a",
    "ultralytics/utils/export/onnx.py": "b",
    "ultralytics/utils/export/openvino.py": "c",
    "ultralytics/nn/tasks.py": "d",
    "ultralytics/nn/modules/head.py": "e",
    "ultralytics/cfg/models/26/yolo26.yaml": "g",
    "docs/en/models/yolo26.md": "ignored",
    "ultralytics/utils/export/mnn.py": "ignored",
}
DIGESTS = {name: f"sha256:{index}" for index, name in enumerate(WEIGHTS)}


class UpstreamWatchTests(unittest.TestCase):
    def test_weights_follow_the_model_matrix(self):
        self.assertEqual(WEIGHTS, tuple(f"{model}.pt" for model in MODELS))

    def test_watch_list_ignores_docs_and_unrelated_exporters(self):
        self.assertTrue(is_watched("ultralytics/nn/modules/head.py"))
        self.assertFalse(is_watched("ultralytics/utils/export/mnn.py"))
        self.assertFalse(is_watched("docs/en/modes/export.md"))

    def test_asset_pin_and_fingerprint(self):
        self.assertEqual(parse_asset_release(DOWNLOADS), "v8.4.0")
        first = fingerprint("v8.4.0", DIGESTS, BLOBS)
        changed = dict(DIGESTS)
        changed[WEIGHTS[0]] = "sha256:new"
        self.assertNotEqual(first, fingerprint("v8.4.0", changed, BLOBS))
        self.assertEqual(parse_fingerprint(release_body("8.4.198", "v8.4.0", first)), first)

    def test_pipeline_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in PIPELINE:
                (root / name).write_text(name)
            before = pipeline_digest(root)
            (root / "build_rknn.py").write_text("changed")
            after = pipeline_digest(root)
        self.assertNotEqual(before, after)
        self.assertNotEqual(fingerprint("v8.4.0", DIGESTS, BLOBS, before),
                            fingerprint("v8.4.0", DIGESTS, BLOBS, after))

    def test_repository_pipeline_files_exist(self):
        self.assertEqual(len(pipeline_digest()), 64)

    def test_patch_version_alone_does_not_build(self):
        current = fingerprint("v8.4.0", DIGESTS, BLOBS)
        should, reason = decide(False, "8.4.198", current, "8.4.197", True, None, current)
        self.assertFalse(should)
        self.assertEqual(reason, "upstream unchanged")

    def test_export_graph_change_builds(self):
        should, reason = decide(False, "8.4.198", "new", "8.4.197", True, "old", None)
        self.assertTrue(should)
        self.assertEqual(reason, "upstream changed")

    def test_manual_run_and_incomplete_release_still_build(self):
        self.assertEqual(decide(True, "8.4.198", "same", "8.4.198", True, "same", None)[1], "manual run")
        self.assertEqual(decide(False, "8.4.198", "same", "8.4.198", False, "same", None)[0], True)

    def test_missing_release_builds_once(self):
        self.assertEqual(decide(False, "8.4.198", "fp", None, False, None, None)[1], "no complete release")

    def test_prune_keeps_only_named_release(self):
        releases = [{"tag_name": "8.4.197"}, {"tag_name": "8.4.21"}, {"tag_name": "8.4.198"}]
        self.assertEqual(
            [item["tag_name"] for item in releases_to_delete(releases, "8.4.198")],
            ["8.4.197", "8.4.21"],
        )


if __name__ == "__main__":
    unittest.main()
