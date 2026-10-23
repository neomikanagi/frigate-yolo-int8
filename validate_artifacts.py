"""Fail closed on incomplete releases; stdlib only, no model dependencies."""
import argparse
import json
from pathlib import Path
import zipfile

MODELS = ("yolo11s", "yolo11m", "yolo11l", "yolo26s", "yolo26m", "yolo26l")
TARGETS = ("rk3568", "rk3576", "rk3588")


def expected_archives():
    archives = {}
    for model in MODELS:
        archives[f"frigate_{model}_int8_320.zip"] = {
            f"{model}_int8.xml", f"{model}_int8.bin"
        }
        for target in TARGETS:
            archives[f"frigate_{model}_{target}_int8_320.zip"] = {
                f"{model}_{target}_320.rknn"
            }
    return archives


def validate_release(release):
    """An existing tag is not proof of a complete published release."""
    if release.get("isDraft") or release.get("isPrerelease"):
        raise ValueError("Release is not a published stable release")
    assets = release["assets"]
    names = {asset["name"] for asset in assets if asset["size"] > 0}
    missing = set(expected_archives()) - names
    if missing:
        raise ValueError(f"Release missing {len(missing)} nonempty assets: {sorted(missing)}")


def validate_directory(directory):
    for name, expected_files in expected_archives().items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing archive: {path}")
        with zipfile.ZipFile(path) as archive:
            files = {info.filename for info in archive.infolist() if info.file_size > 0}
            missing = expected_files - files
            if missing:
                raise ValueError(f"{name} missing nonempty model files: {sorted(missing)}")
            bad_file = archive.testzip()
            if bad_file:
                raise ValueError(f"{name} has corrupt member: {bad_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--directory", type=Path)
    group.add_argument("--release-json", type=Path)
    args = parser.parse_args()
    try:
        if args.directory is not None:
            validate_directory(args.directory)
        else:
            validate_release(json.loads(args.release_json.read_text()))
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    print(f"[SUCCESS] All {len(expected_archives())} model archives are present")


if __name__ == "__main__":
    main()
