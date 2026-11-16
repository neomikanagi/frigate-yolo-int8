"""FP32 ONNX package: AMD Ryzen AI (XDNA2) NPUs through the frigate-xdna sidecar, which compiles it
to BF16 on the machine, and Frigate's ONNX detector on ROCm/MIGraphX or CPU.

The graph is the same raw head the other packages use: input `images` float32 [1, 3, 320, 320]
in 0-1 RGB, output [1, 84, 2100].
"""
import shutil
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from yolo_common import IMG_SIZE, coco128_images, detections, export_onnx, load_rgb, model_name, onnx_reference, split_images, write_zip


def main():
    name = model_name()
    _, holdout = split_images(coco128_images())
    onnx_path = Path(export_onnx(YOLO(f"{name}.pt")))

    outputs = onnx_reference(onnx_path, [load_rgb(path) for path in holdout])
    if not all(np.isfinite(out).all() for out in outputs):
        raise RuntimeError("ONNX output has NaN or Inf")
    found = sum(len(detections(out)) for out in outputs)
    print(f"[GATE] {name} ONNX: {found} detections on {len(holdout)} holdout images", flush=True)
    if found < len(holdout):
        raise RuntimeError(f"ONNX found only {found} objects on {len(holdout)} COCO images")

    packaged = Path(f"{name}_{IMG_SIZE}.onnx")
    shutil.copyfile(onnx_path, packaged)
    zip_name = f"frigate_{name}_onnx_{IMG_SIZE}.zip"
    write_zip(zip_name, [packaged])
    print(f"[SUCCESS] {zip_name}")


if __name__ == "__main__":
    main()
