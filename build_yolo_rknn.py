import os, shutil, zipfile, gc
from pathlib import Path
from ultralytics import YOLO
from rknn.api import RKNN

MODEL_NAME = os.environ.get("MODEL_NAME", "yolo11s")
RKNN_TARGET = os.environ.get("RKNN_TARGET", "rk3588")
IMG_SIZE = 320
OUTPUT_DIR = Path(f"{MODEL_NAME}_{RKNN_TARGET}_int8_pack_{IMG_SIZE}")
ZIP_NAME = f"frigate_{MODEL_NAME}_{RKNN_TARGET}_int8_{IMG_SIZE}.zip"

def main():
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    print(f"[INFO] Preparing RKNN quantization for: {MODEL_NAME} ({RKNN_TARGET})")

    model = YOLO(f"{MODEL_NAME}.pt")

    if not os.path.exists("datasets/coco128"):
        print("[INFO] Pre-downloading calibration dataset (COCO128)...")
        try:
            model.val(data='coco128.yaml', imgsz=32, plots=False)
        except Exception as e:
            print(f"[WARNING] Validation trigger failed, but dataset might be downloaded: {e}")

    print(f"[INFO] Exporting to ONNX...")
    onnx_path = model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        dynamic=False,
        simplify=True
    )

    img_dir = Path("datasets/coco128/images/train2017")
    img_files = sorted(img_dir.glob("*.jpg"))[:20]
    if not img_files:
        raise RuntimeError("COCO128 calibration images are missing or empty")
    with open("dataset.txt", "w") as f:
        for img in img_files:
            f.write(f"{img.absolute()}\n")

    print(f"[INFO] Converting ONNX to RKNN for {RKNN_TARGET}...")
    rknn = RKNN(verbose=False)
    try:
        # Toolkit 2.5.0 raises IndexError in these fusion passes for YOLO11
        # (DFL Conv/Reshape/Split) and YOLO26 (attention). The compiler itself
        # recommends disable_rules in the failure log; keep INT8 enabled.
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=RKNN_TARGET,
            quantized_dtype='asymmetric_quantized-8',
            quantized_algorithm='normal',
            disable_rules=[
                'seperate_conv_rs_split',
                'fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa',
            ],
        )
        if ret != 0:
            raise RuntimeError(f"RKNN configuration failed: {ret}")
        if rknn.load_onnx(model=onnx_path) != 0:
            raise RuntimeError("Load ONNX failed")
        if rknn.build(do_quantization=True, dataset="dataset.txt") != 0:
            raise RuntimeError("Build RKNN failed")

        final_rknn_path = OUTPUT_DIR / f"{MODEL_NAME}_{RKNN_TARGET}_{IMG_SIZE}.rknn"
        if rknn.export_rknn(str(final_rknn_path)) != 0:
            raise RuntimeError("Export RKNN failed")
        if not final_rknn_path.is_file() or final_rknn_path.stat().st_size == 0:
            raise RuntimeError("RKNN compiler did not produce a nonempty model")
    finally:
        rknn.release()

    print(f"[INFO] Packaging {MODEL_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in OUTPUT_DIR.glob("**/*"):
            z.write(f, f.relative_to(OUTPUT_DIR))

    print(f"[SUCCESS] Build completed! Output file: {ZIP_NAME}")

    del model, rknn
    gc.collect()

if __name__ == "__main__":
    main()
