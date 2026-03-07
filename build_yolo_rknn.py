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
    if img_dir.exists():
        img_files = list(img_dir.glob("*.jpg"))[:20]
        with open("dataset.txt", "w") as f:
            for img in img_files:
                f.write(f"{img.absolute()}\n")
    else:
        print("[ERROR] COCO128 dataset not found for calibration!")
        return

    print(f"[INFO] Converting ONNX to RKNN for {RKNN_TARGET}...")
    rknn = RKNN(verbose=False)
    
    rknn.config(
        mean_values=[[0, 0, 0]], 
        std_values=[[255, 255, 255]], 
        target_platform=RKNN_TARGET,
        quantized_dtype='asymmetric_quantized-8',
        quantized_algorithm='normal'
    )

    if rknn.load_onnx(model=onnx_path) != 0:
        print("[ERROR] Load ONNX failed!")
        return

    if rknn.build(do_quantization=True, dataset="dataset.txt") != 0:
        print("[ERROR] Build RKNN failed!")
        return

    final_rknn_path = OUTPUT_DIR / f"{MODEL_NAME}_{RKNN_TARGET}_{IMG_SIZE}.rknn"
    if rknn.export_rknn(str(final_rknn_path)) != 0:
        print("[ERROR] Export RKNN failed!")
        return

    print(f"[INFO] Packaging {MODEL_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in OUTPUT_DIR.glob("**/*"):
            z.write(f, f.relative_to(OUTPUT_DIR))

    print(f"[SUCCESS] Build completed! Output file: {ZIP_NAME}")

    rknn.release()
    del model, rknn
    gc.collect()

if __name__ == "__main__":
    main()
