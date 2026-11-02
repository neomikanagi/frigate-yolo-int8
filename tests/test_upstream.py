import unittest

from upstream_watch import (
    decide, fingerprint, is_watched, parse_asset_release, parse_fingerprint,
    release_body, releases_to_delete,
)

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
    "ultralytics/cfg/models/11/yolo11.yaml": "f",
    "ultralytics/cfg/models/26/yolo26.yaml": "g",
    "docs/en/models/yolo11.md": "ignored",
    "ultralytics/utils/export/mnn.py": "ignored",
}
DIGESTS = {name: f"sha256:{index}" for index, name in enumerate((
    "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt",
))}


class UpstreamWatchTests(unittest.TestCase):
    def test_watch_list_ignores_docs_and_unrelated_exporters(self):
        self.assertTrue(is_watched("ultralytics/nn/modules/head.py"))
        self.assertFalse(is_watched("ultralytics/utils/export/mnn.py"))
        self.assertFalse(is_watched("docs/en/modes/export.md"))

    def test_asset_pin_and_fingerprint(self):
        self.assertEqual(parse_asset_release(DOWNLOADS), "v8.4.0")
        first = fingerprint("v8.4.0", DIGESTS, BLOBS)
        changed = dict(DIGESTS)
        changed["yolo11s.pt"] = "sha256:new"
        self.assertNotEqual(first, fingerprint("v8.4.0", changed, BLOBS))
        self.assertEqual(parse_fingerprint(release_body("8.4.198", "v8.4.0", first)), first)

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
