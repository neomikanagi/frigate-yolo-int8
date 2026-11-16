#!/bin/bash
# frigate-yolo-sync: keep Frigate's YOLO model on the latest frigate-yolo-int8 release.
#
# Install:  install -m 755 frigate-yolo-sync.sh /usr/local/bin/frigate-yolo-sync.sh
# Config:   /etc/default/frigate-yolo-sync (see frigate-yolo-sync.example)
# Schedule: 30 6 * * * /usr/local/bin/frigate-yolo-sync.sh >>/var/log/frigate-yolo-sync.log 2>&1
#
# An asset is replaced only when its sha256 in the release differs from the installed one, so a
# rebuild under the same tag is picked up too. Each new model is loaded and run once inside the
# Frigate container before it replaces the old one; if Frigate does not come back healthy after the
# restart, the previous model is put back.
set -euo pipefail

CONF="${FRIGATE_YOLO_SYNC_CONF:-/etc/default/frigate-yolo-sync}"
# shellcheck disable=SC1090
[[ -f "$CONF" ]] && source "$CONF"

REPO="${REPO:-neomikanagi/frigate-yolo-int8}"
MODEL_DIR="${MODEL_DIR:-/docker/frigate/config/model_cache}"            # host path
CONTAINER_MODEL_DIR="${CONTAINER_MODEL_DIR:-/config/model_cache}"      # same directory inside Frigate
CONFIG_YAML="${CONFIG_YAML:-/docker/frigate/config/config.yaml}"
FRIGATE_CONTAINER="${FRIGATE_CONTAINER:-frigate}"
RESTART_ON_UPDATE="${RESTART_ON_UPDATE:-1}"
PRUNE="${PRUNE:-1}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"
STATE_FILE="${STATE_FILE:-${MODEL_DIR}/.yolo-sync-state}"
API_URL="${API_URL:-https://api.github.com/repos/${REPO}/releases/latest}"
ASSETS="${ASSETS:-}"

log() { echo "[$(date -Is)] frigate-yolo-sync: $*"; }
die() { log "ERROR: $*"; exit 1; }

for cmd in curl unzip sha256sum perl docker flock; do
    command -v "$cmd" >/dev/null 2>&1 || die "missing command: $cmd"
done
[[ -n "$ASSETS" ]] || die "set ASSETS in ${CONF}, e.g. ASSETS=\"frigate_yolo26l_fp16_320.zip\""

exec 9>/run/lock/frigate-yolo-sync.lock
flock -n 9 || { log "another run is active"; exit 0; }

read -r -a WANTED <<<"$ASSETS"

# Prints TAG|<tag> once, then ASSET|<name>|<url>|<sha256> per frigate_*.zip asset.
parse_release() {
    perl -ne '
        if (!$tag && /"tag_name"\s*:\s*"([^"]+)"/) { $tag = $1; print "TAG|$tag\n"; }
        if (/"name"\s*:\s*"(frigate_[^"]+\.zip)"/) { $n = $1; $s = ""; }
        if (defined $n && /"digest"\s*:\s*"sha256:([a-f0-9]+)"/) { $s = $1; }
        if (defined $n && /"browser_download_url"\s*:\s*"([^"]+)"/) { print "ASSET|$n|$1|$s\n"; $n = undef; }
    '
}

state_sha() {
    [[ -f "$STATE_FILE" ]] || return 0
    grep -E "^SHA_$1=" "$STATE_FILE" | head -1 | cut -d= -f2- || true
}

# Load and run the model once inside Frigate's own runtime; fails on a wrong output or load error.
preflight() {
    docker exec -i "$FRIGATE_CONTAINER" python3 - "$1" <<'PY'
import sys
import numpy as np
path = sys.argv[1]
if path.endswith(".rknn"):
    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    assert rknn.load_rknn(path) == 0, "load_rknn failed"
    assert rknn.init_runtime(core_mask=0) == 0, "init_runtime failed"
    out = rknn.inference(inputs=[np.zeros((1, 320, 320, 3), np.uint8)])[0]
    rknn.release()
elif path.endswith(".xml"):
    import openvino as ov
    compiled = ov.Core().compile_model(path, "CPU")
    out = compiled({0: np.zeros((1, 3, 320, 320), np.uint8)})[compiled.outputs[0]]
elif path.endswith(".onnx"):
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = session.run(None, {session.get_inputs()[0].name: np.zeros((1, 3, 320, 320), np.float32)})[0]
else:
    sys.exit(f"unknown model type: {path}")
out = np.asarray(out)
assert list(out.shape) == [1, 84, 2100], f"output shape {out.shape}"
assert np.isfinite(out).all(), "output has NaN/Inf"
print("preflight ok", path)
PY
}

# Inference speed of the first detector from the Frigate API, empty while Frigate is not serving.
detector_speed() {
    local stats
    stats=$(docker exec "$FRIGATE_CONTAINER" curl -sf -m 5 http://127.0.0.1:5000/api/stats 2>/dev/null || true)
    [[ "$stats" =~ \"inference_speed\":\ *([0-9.]+) ]] && echo "${BASH_REMATCH[1]}"
    return 0
}

wait_healthy() {
    local since="$1" deadline=$((SECONDS + HEALTH_TIMEOUT)) status restarts
    while (( SECONDS < deadline )); do
        status=$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$FRIGATE_CONTAINER" 2>/dev/null || echo missing)
        # Docker can still report the pre-restart health for a moment; require the API to list a running detector.
        if [[ ( "$status" == healthy || "$status" == none ) && -n "$(detector_speed)" ]]; then
            restarts=$(docker inspect -f '{{.RestartCount}}' "$FRIGATE_CONTAINER")
            sleep 60  # let the detector process load the model and run
            if [[ "$(docker inspect -f '{{.RestartCount}}' "$FRIGATE_CONTAINER")" != "$restarts" ]] \
                || [[ "$(docker inspect -f '{{.State.Running}}' "$FRIGATE_CONTAINER")" != true ]]; then
                log "Frigate restarted again after the model change"
                return 1
            fi
            if docker logs --since "$since" "$FRIGATE_CONTAINER" 2>&1 \
                | grep -qiE 'Error loading|Failed to load|Failed to initialize|Unable to load|detection appears to (be stuck|have stopped)'; then
                log "Frigate logged a detector error after the model change"
                return 1
            fi
            return 0
        fi
        sleep 5
    done
    log "Frigate not healthy after ${HEALTH_TIMEOUT}s (last status: ${status})"
    return 1
}

prune_models() {
    local dir base keep wanted
    for dir in "$MODEL_DIR"/frigate_*; do
        [[ -d "$dir" ]] || continue
        base=$(basename "$dir")
        keep=0
        for wanted in "${WANTED[@]}"; do
            [[ "$base" == "${wanted%.zip}" || "$base" == "${wanted%.zip}.prev" ]] && keep=1
        done
        [[ "$keep" -eq 0 && -f "$CONFIG_YAML" ]] && grep -q "/${base}/" "$CONFIG_YAML" && keep=1
        if [[ "$keep" -eq 0 ]]; then
            log "removing unused ${base}"
            rm -rf "$dir"
        fi
    done
}

main() {
    local dry_run=0 force=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run=1 ;;
            --force) force=1 ;;
            -h|--help) echo "usage: $0 [--dry-run] [--force]"; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
        shift
    done

    local meta tag="" kind a b c
    local headers=(-H "Accept: application/vnd.github+json")
    [[ -n "${GITHUB_TOKEN:-}" ]] && headers+=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
    declare -A URL SHA
    meta=$(curl -fsSL --retry 3 "${headers[@]}" "$API_URL") || die "cannot fetch ${API_URL}"
    while IFS='|' read -r kind a b c; do
        case "$kind" in
            TAG) tag="$a" ;;
            ASSET) URL["$a"]="$b"; SHA["$a"]="$c" ;;
        esac
    done < <(printf '%s' "$meta" | parse_release)
    [[ -n "$tag" ]] || die "release has no tag_name"
    log "latest release ${tag}; wanted: ${WANTED[*]}"

    local asset dest changed=()
    for asset in "${WANTED[@]}"; do
        [[ -n "${URL[$asset]:-}" ]] || die "release ${tag} has no asset ${asset}"
        dest="${MODEL_DIR}/${asset%.zip}"
        if [[ "$force" -eq 0 && -n "${SHA[$asset]}" && "$(state_sha "$asset")" == "${SHA[$asset]}" \
            && -n "$(ls -A "$dest" 2>/dev/null)" ]]; then
            log "${asset} is current"
        else
            changed+=("$asset")
        fi
    done
    if [[ ${#changed[@]} -eq 0 ]]; then
        [[ "$PRUNE" == 1 && "$dry_run" -eq 0 ]] && prune_models
        log "nothing to update"
        exit 0
    fi
    if [[ "$dry_run" -eq 1 ]]; then
        log "DRY-RUN: would update ${changed[*]}"
        exit 0
    fi

    local work got incoming model swapped=()
    work=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf '$work' '${MODEL_DIR}'/.incoming-*" EXIT
    for asset in "${changed[@]}"; do
        log "downloading ${asset}"
        curl -fsSL --retry 3 --retry-delay 2 -o "${work}/${asset}" "${URL[$asset]}"
        got=$(sha256sum "${work}/${asset}" | awk '{print $1}')
        [[ -z "${SHA[$asset]}" || "$got" == "${SHA[$asset]}" ]] || die "${asset} sha256 mismatch"
        SHA["$asset"]="$got"
        incoming="${MODEL_DIR}/.incoming-${asset%.zip}"
        rm -rf "$incoming"
        mkdir -p "$incoming"
        unzip -qo "${work}/${asset}" -d "$incoming"
        chmod -R a+rX "$incoming"
        for model in "$incoming"/*.rknn "$incoming"/*.xml "$incoming"/*.onnx; do
            [[ -f "$model" ]] || continue
            preflight "${CONTAINER_MODEL_DIR}/$(basename "$incoming")/$(basename "$model")" \
                || die "${asset} failed the preflight run; the installed model is unchanged"
        done
    done

    for asset in "${changed[@]}"; do
        dest="${MODEL_DIR}/${asset%.zip}"
        rm -rf "${dest}.prev"
        [[ -d "$dest" ]] && mv "$dest" "${dest}.prev"
        mv "${MODEL_DIR}/.incoming-${asset%.zip}" "$dest"
        swapped+=("$asset")
        log "installed ${asset} (${SHA[$asset]:0:12})"
    done

    if [[ "$RESTART_ON_UPDATE" == 1 ]] && docker ps --format '{{.Names}}' | grep -qx "$FRIGATE_CONTAINER"; then
        local since
        since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        log "restarting ${FRIGATE_CONTAINER}"
        docker restart "$FRIGATE_CONTAINER" >/dev/null
        if ! wait_healthy "$since"; then
            for asset in "${swapped[@]}"; do
                dest="${MODEL_DIR}/${asset%.zip}"
                if [[ -d "${dest}.prev" ]]; then
                    rm -rf "$dest"
                    mv "${dest}.prev" "$dest"
                fi
            done
            docker restart "$FRIGATE_CONTAINER" >/dev/null
            die "rolled back ${swapped[*]} to the previous model"
        fi
        log "Frigate healthy with the new model; detector inference $(detector_speed) ms"
    fi

    {
        echo "TAG=${tag}"
        echo "UPDATED_AT=$(date -Is)"
        for asset in "${WANTED[@]}"; do
            if [[ " ${changed[*]} " == *" ${asset} "* ]]; then
                echo "SHA_${asset}=${SHA[$asset]}"
            else
                echo "SHA_${asset}=$(state_sha "$asset")"
            fi
        done
    } >"${STATE_FILE}.new"
    mv "${STATE_FILE}.new" "$STATE_FILE"
    [[ "$PRUNE" == 1 ]] && prune_models
    log "done"
}

main "$@"
