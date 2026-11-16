"""Check an RKNN package on the board against the FP32 ONNX, on your own camera frames.

The RKNN simulator does not reproduce every runtime behaviour (hybrid FP16 layers on RK3566 are one
case), so a new RKNN recipe is accepted only after this check on real hardware. It runs inside the
Frigate stable-rk container, which has rknnlite, onnxruntime and OpenCV:

  docker cp tools/rknn_board_check.py frigate:/tmp/ && docker cp yolo_common.py frigate:/tmp/
  docker exec frigate python3 /tmp/rknn_board_check.py \
      --rknn /config/model_cache/frigate_yolo26l_rk3568_int8_320/yolo26l_rk3568_320.rknn \
      --onnx /config/yolo26l_320.onnx --images /config/frames

Frames are center-cropped to a square and resized like a Frigate detection region. Stop Frigate
first for a clean latency number; the NPU is shared.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from yolo_common import agreement, onnx_reference  # noqa: E402


def square_crop(path, size):
    import cv2

    image = cv2.imread(str(path))
    h, w = image.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    crop = cv2.resize(image[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rknn", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        parser.exit(1, f"no images in {args.images}\n")
    frames = [square_crop(p, args.size) for p in paths]
    references = onnx_reference(args.onnx, frames)

    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(args.rknn) != 0 or rknn.init_runtime(core_mask=0) != 0:
        parser.exit(1, "cannot load the RKNN model\n")
    outputs = [rknn.inference(inputs=[frame[None]])[0] for frame in frames]
    for _ in range(3):
        rknn.inference(inputs=[frames[0][None]])
    times = []
    for _ in range(30):
        start = time.perf_counter()
        rknn.inference(inputs=[frames[0][None]])
        times.append((time.perf_counter() - start) * 1000)
    rknn.release()

    result = agreement(references, outputs, min_score=0.4)
    print(f"{len(frames)} frames: found {result['recall']:.1%} of {result['reference_detections']} FP32 detections, "
          f"{result['extra']} extra, class cosine {result['class_cosine']:.4f}, "
          f"latency median {np.median(times):.1f} ms (python call; Frigate shows about 20 ms less)")


if __name__ == "__main__":
    main()
