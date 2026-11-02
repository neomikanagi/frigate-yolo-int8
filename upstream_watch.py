"""Build only when ultralytics weights or the export graph change. Stdlib only."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from validate_artifacts import validate_release

ULTRALYTICS = "ultralytics/ultralytics"
ASSETS_REPO = "ultralytics/assets"
WEIGHTS = ("yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt")
# Paths that change the OpenVINO/ONNX graph this pipeline compiles. A PyPI
# patch that does not touch these, or the weight files, is not a rebuild.
WATCH = (
    "ultralytics/engine/exporter.py",
    "ultralytics/utils/export/onnx.py",
    "ultralytics/utils/export/openvino.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/nn/modules/",
    "ultralytics/cfg/models/11/yolo11.yaml",
    "ultralytics/cfg/models/26/yolo26.yaml",
)
FINGERPRINT_KEY = "upstream-fingerprint"


def is_watched(path):
    return any(path == item or (item.endswith("/") and path.startswith(item)) for item in WATCH)


def parse_asset_release(downloads_py):
    match = re.search(
        r"def attempt_download_asset\(.*?^\s*release:\s*str\s*=\s*[\"']([^\"']+)[\"']",
        downloads_py,
        re.S | re.M,
    )
    if not match:
        raise ValueError("cannot find attempt_download_asset release pin")
    return match.group(1)


def parse_fingerprint(body):
    if not body:
        return None
    match = re.search(rf"^{re.escape(FINGERPRINT_KEY)}:\s*([0-9a-f]{{64}})\s*$", body, re.M)
    return match.group(1) if match else None


def fingerprint(weights_tag, weight_digests, blobs):
    missing = [path for path in WATCH if not path.endswith("/") and path not in blobs]
    if missing:
        raise ValueError(f"missing watched files: {missing}")
    missing_weights = [name for name in WEIGHTS if not weight_digests.get(name)]
    if missing_weights:
        raise ValueError(f"missing weight digests: {missing_weights}")
    lines = [f"weights-tag {weights_tag}"]
    lines.extend(f"weight {name} {weight_digests[name]}" for name in WEIGHTS)
    lines.extend(f"blob {path} {blobs[path]}" for path in sorted(blobs) if is_watched(path))
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def release_body(version, weights_tag, fingerprint_value):
    return (
        "OpenVINO and RKNN INT8 models for Frigate.\n"
        f"ultralytics: {version}\n"
        f"weights-tag: {weights_tag}\n"
        f"{FINGERPRINT_KEY}: {fingerprint_value}\n"
    )


def decide(force, latest_version, current_fp, release_tag, release_complete, release_fp, previous_fp):
    """Return (should_build, reason). A newer patch with the same fingerprint does not build."""
    if release_tag == latest_version and release_tag and not release_complete:
        return True, "latest release incomplete"
    if force:
        return True, "manual run"
    if not release_tag or not release_complete:
        return True, "no complete release"
    effective = release_fp or previous_fp
    if not effective:
        return True, "upstream fingerprint unknown"
    if effective == current_fp:
        return False, "upstream unchanged"
    return True, "upstream changed"


def releases_to_delete(releases, keep):
    return [release for release in releases if release.get("tag_name") != keep]


def _request(url, token, method="GET"):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "frigate-yolo-int8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method=method)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429, 502, 503) and attempt < 4:
                time.sleep(2 ** attempt)
                continue
            return exc.code, exc.read()
    raise RuntimeError(f"{method} {url} failed")


def github_json(path, token):
    status, payload = _request(f"https://api.github.com{path}", token)
    if status != 200:
        raise RuntimeError(f"GitHub {path} returned {status}: {payload[:200]!r}")
    return json.loads(payload.decode())


def pypi_version():
    status, payload = _request("https://pypi.org/pypi/ultralytics/json", None)
    if status != 200:
        raise RuntimeError(f"PyPI ultralytics returned {status}")
    version = json.loads(payload.decode())["info"]["version"]
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError(f"unexpected ultralytics version: {version}")
    return version


def _tree_blobs(token, ref):
    data = github_json(f"/repos/{ULTRALYTICS}/git/trees/{ref}?recursive=1", token)
    if data.get("truncated"):
        raise RuntimeError(f"ultralytics tree {ref} was truncated")
    return {item["path"]: item["sha"] for item in data["tree"] if item["type"] == "blob"}


def _downloads_py(token, ref):
    status, payload = _request(
        f"https://raw.githubusercontent.com/{ULTRALYTICS}/{ref}/ultralytics/utils/downloads.py",
        token,
    )
    if status != 200:
        raise RuntimeError(f"downloads.py at {ref} returned {status}")
    return payload.decode()


def _weight_digests(token, weights_tag):
    ref = "latest" if weights_tag == "latest" else f"tags/{weights_tag}"
    data = github_json(f"/repos/{ASSETS_REPO}/releases/{ref}", token)
    digests = {}
    for asset in data.get("assets", []):
        if asset["name"] in WEIGHTS:
            digests[asset["name"]] = asset.get("digest") or ""
    return data["tag_name"], digests


def fetch_fingerprint(token, ref):
    blobs = _tree_blobs(token, ref)
    weights_tag = parse_asset_release(_downloads_py(token, ref))
    resolved_tag, digests = _weight_digests(token, weights_tag)
    return fingerprint(resolved_tag, digests, blobs), resolved_tag


def _release_or_none(repo, tag, token):
    status, payload = _request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", token)
    if status == 404:
        return None
    if status != 200:
        raise RuntimeError(f"release {tag} returned {status}: {payload[:200]!r}")
    return json.loads(payload.decode())


def _complete(release):
    if release is None:
        return False
    try:
        validate_release(release)
    except (ValueError, KeyError):
        return False
    return True


def _latest_complete(repo, token):
    status, payload = _request(f"https://api.github.com/repos/{repo}/releases?per_page=20", token)
    if status != 200:
        raise RuntimeError(f"release list returned {status}")
    for release in json.loads(payload.decode()):
        if _complete(release):
            return release
    return None


def evaluate(repo, token, force=False):
    version = pypi_version()
    current_fp, weights_tag = fetch_fingerprint(token, f"v{version}")
    target = _release_or_none(repo, version, token)
    target_complete = _complete(target)
    basis = target if target_complete else _latest_complete(repo, token)
    release_tag = basis["tag_name"] if basis else None
    release_fp = parse_fingerprint(basis.get("body")) if basis else None
    previous_fp = None
    if basis and not release_fp and release_tag != version:
        previous_fp, _ = fetch_fingerprint(token, f"v{release_tag}")
    elif basis and not release_fp and release_tag == version:
        previous_fp = current_fp
    should, reason = decide(
        force, version, current_fp, release_tag, bool(basis) and _complete(basis), release_fp, previous_fp,
    )
    if target and not target_complete:
        should, reason = True, "latest release incomplete"
    return {
        "version": version,
        "should_build": should,
        "fingerprint": current_fp,
        "weights_tag": weights_tag,
        "reason": reason,
    }


def _paginate(path, token, key=None):
    page = 1
    items = []
    while True:
        separator = "&" if "?" in path else "?"
        data = github_json(f"{path}{separator}per_page=100&page={page}", token)
        batch = data[key] if key else data
        if not batch:
            return items
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


def prune(repo, token, keep, delete_artifacts=True):
    deleted = []
    for release in releases_to_delete(_paginate(f"/repos/{repo}/releases", token), keep):
        status, payload = _request(f"https://api.github.com/repos/{repo}/releases/{release['id']}", token, "DELETE")
        if status not in (204, 404):
            raise RuntimeError(f"delete release {release['tag_name']} returned {status}: {payload[:200]!r}")
        tag = urllib.request.quote(release["tag_name"], safe="")
        status, payload = _request(f"https://api.github.com/repos/{repo}/git/refs/tags/{tag}", token, "DELETE")
        if status not in (204, 404, 422):
            raise RuntimeError(f"delete tag {release['tag_name']} returned {status}: {payload[:200]!r}")
        deleted.append(release["tag_name"])
    artifacts = 0
    if delete_artifacts:
        for artifact in _paginate(f"/repos/{repo}/actions/artifacts", token, "artifacts"):
            status, payload = _request(
                f"https://api.github.com/repos/{repo}/actions/artifacts/{artifact['id']}", token, "DELETE",
            )
            if status not in (204, 404):
                raise RuntimeError(f"delete artifact {artifact['id']} returned {status}: {payload[:200]!r}")
            artifacts += 1
    return deleted, artifacts


def _token():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN is not set")
    return token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--repo", required=True)
    check.add_argument("--force", action="store_true")
    body = sub.add_parser("body")
    body.add_argument("--version", required=True)
    body.add_argument("--weights-tag", required=True)
    body.add_argument("--fingerprint", required=True)
    prune_cmd = sub.add_parser("prune")
    prune_cmd.add_argument("--repo", required=True)
    prune_cmd.add_argument("--keep", required=True)
    args = parser.parse_args()
    if args.command == "body":
        sys.stdout.write(release_body(args.version, args.weights_tag, args.fingerprint))
        return
    if args.command == "prune":
        deleted, artifacts = prune(args.repo, _token(), args.keep)
        print(f"deleted {len(deleted)} releases; deleted {artifacts} actions artifacts", file=sys.stderr)
        return
    result = evaluate(args.repo, _token(), args.force)
    print(f"{result['version']}: {result['reason']}", file=sys.stderr)
    print(f"version={result['version']}")
    print(f"should_build={'true' if result['should_build'] else 'false'}")
    print(f"fingerprint={result['fingerprint']}")
    print(f"weights_tag={result['weights_tag']}")
    print(f"reason={result['reason']}")


if __name__ == "__main__":
    main()
