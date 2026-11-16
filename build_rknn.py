"""RKNN INT8 packages for Frigate on Rockchip NPUs (RK3566/RK3568, RK3576, RK3588).

Frigate yolo-generic reads one [1, 84, 2100] tensor that concatenates box coordinates (0-320 px)
with class scores (0-1). Quantized as one INT8 tensor, the step is about 1.3, so every class score
rounds to 0: the NPU runs and Frigate reports nothing. The fix is a hybrid build: everything up to
the last head convolutions stays INT8, and only the box decode, sigmoid and final concat (a few
element-wise ops on 2100 anchors) run in FP16. Those ops add about 1-2 ms on an RK3566.

Toolkit 2.3.2 matches the librknnrt that Frigate stable-rk ships. On RK3566 that runtime returns
shifted boxes when a hybrid model runs convolutions or attention in FP16, so only the decode tail
is FP16. The simulator does not show that; it was measured on the board.
"""
import gc
import os
import re
import shutil
from pathlib import Path

import numpy as np
from rknn.api import RKNN
from ultralytics import YOLO

from yolo_common import IMG_SIZE, coco128_images, export_onnx, gate, load_rgb, model_name, onnx_reference, split_images, write_zip

TARGETS = ("rk3568", "rk3576", "rk3588")
CALIBRATION = 50  # min-max calibration; 20, 50 and 100 images measured within noise on RK3566, 50 best
# Measured on the COCO128 holdout (simulator): 26s 96% / 0 extra / 0.964, 26l 99% / 3 / 0.966.
GATE = dict(min_recall=0.90, max_extra=8, min_cosine=0.90)


def head_tail(onnx_path):
    """Subgraphs from the raw box/class convolutions of the Detect head to the output: these stay FP16."""
    import onnx

    graph = onnx.load(onnx_path).graph
    names = {node.name for node in graph.node}
    heads = {int(m.group(1)) for n in names if (m := re.fullmatch(r"/model\.(\d+)/cv2\.0/cv2\.0\.2/Conv", n))}
    if len(heads) != 1:
        raise RuntimeError(f"cannot find the Detect head in {onnx_path}: {sorted(heads)}")
    head = heads.pop()
    output = graph.output[0].name
    pairs = []
    for branch in ("cv2", "cv3"):
        for level in range(3):
            conv = f"/model.{head}/{branch}.{level}/{branch}.{level}.2/Conv"
            if conv not in names:
                raise RuntimeError(f"Detect head layer {conv} is missing")
            pairs.append([f"{conv}_output_0", output])
    return pairs


def write_dataset(images, directory):
    """NCHW uint8 arrays: the toolkit applies mean/std itself, as at runtime."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, path in enumerate(images):
        array = directory / f"{index:03d}.npy"
        np.save(array, load_rgb(path).transpose(2, 0, 1)[None])
        lines.append(f"{array.resolve()}\n")
    dataset = directory / "dataset.txt"
    dataset.write_text("".join(lines))
    return dataset


def check(ret, step):
    if ret != 0:
        raise RuntimeError(f"RKNN {step} failed: {ret}")


def build_target(name, onnx_path, target, dataset, holdout_rgb, references):
    work = Path(f"{name}_{target}_rknn_{IMG_SIZE}").resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    previous = Path.cwd()
    os.chdir(work)  # step 1 writes <model>.model/.data/.quantization.cfg into the cwd
    config = dict(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=target)
    base = Path(onnx_path).stem
    rknn = RKNN(verbose=False)
    try:
        check(rknn.config(**config), "config")
        check(rknn.load_onnx(model=str(onnx_path)), "load_onnx")
        check(rknn.hybrid_quantization_step1(dataset=str(dataset), proposal=False, custom_hybrid=head_tail(onnx_path)),
              "hybrid_quantization_step1")
        rknn.release()
        rknn = RKNN(verbose=False)
        check(rknn.hybrid_quantization_step2(model_input=f"{base}.model", data_input=f"{base}.data",
                                             model_quantization_cfg=f"{base}.quantization.cfg"),
              "hybrid_quantization_step2")
        packaged = work / f"{name}_{target}_{IMG_SIZE}.rknn"
        check(rknn.export_rknn(str(packaged)), "export_rknn")
        check(rknn.init_runtime(), "simulator init_runtime")
        outputs = [rknn.inference(inputs=[rgb[None]], data_format="nhwc")[0] for rgb in holdout_rgb]
        gate(f"{name} RKNN {target} (simulator)", references, outputs, **GATE)
    finally:
        rknn.release()
        os.chdir(previous)
    zip_name = f"frigate_{name}_{target}_int8_{IMG_SIZE}.zip"
    write_zip(zip_name, [packaged])
    print(f"[SUCCESS] {zip_name}", flush=True)


def main():
    name = model_name()
    targets = os.environ.get("RKNN_TARGETS", " ".join(TARGETS)).split()
    calibration, holdout = split_images(coco128_images())
    onnx_path = Path(export_onnx(YOLO(f"{name}.pt"))).resolve()
    holdout_rgb = [load_rgb(path) for path in holdout]
    references = onnx_reference(onnx_path, holdout_rgb)
    dataset = write_dataset(calibration[:CALIBRATION], Path(f"{name}_rknn_calibration").resolve())
    for target in targets:
        print(f"[INFO] RKNN INT8 (FP16 decode tail) {name} for {target}", flush=True)
        build_target(name, onnx_path, target, dataset, holdout_rgb, references)
        gc.collect()


if __name__ == "__main__":
    main()
