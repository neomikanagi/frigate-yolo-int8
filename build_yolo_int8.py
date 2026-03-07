import os, shutil, zipfile, gc
from pathlib import Path
from ultralytics import YOLO
import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat

MODEL_NAME = os.environ.get("MODEL_NAME", "yolo26s")
IMG_SIZE = 320
OUTPUT_DIR = Path(f"{MODEL_NAME}_int8_pack_{IMG_SIZE}")
ZIP_NAME = f"frigate_{MODEL_NAME}_int8_{IMG_SIZE}.zip"

def main():
    if OUTPUT_DIR.exists(): shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()

    print(f"[INFO] Starting quantization for model: {MODEL_NAME} (INT8, Size={IMG_SIZE})")

    if not os.path.exists("datasets/coco128"):
        print("[INFO] Pre-downloading calibration dataset (COCO128)...")
        try: 
            YOLO(f"{MODEL_NAME}.pt").export(format='openvino', int8=True, data='coco128.yaml', imgsz=32)
            gc.collect()
        except: 
            pass

    print(f"[INFO] Loading {MODEL_NAME}.pt...")
    model = YOLO(f"{MODEL_NAME}.pt")

    print(f"[INFO] Exporting to INT8 (dynamic=False)...")
    export_path = model.export(
        format="openvino",
        imgsz=IMG_SIZE,
        int8=True,
        data="coco128.yaml",
        dynamic=False
    )

    print(f"[INFO] Executing manual PPP fix (Input type U8 -> F32)...")
    gen_xml = next((f for f in os.listdir(export_path) if f.endswith('.xml')), None)

    if gen_xml:
        core = ov.Core()
        ov_model = core.read_model(os.path.join(export_path, gen_xml))

        ppp = PrePostProcessor(ov_model)
        ppp.input().tensor().set_element_type(ov.Type.u8) \
                            .set_layout(ov.Layout("NCHW")) \
                            .set_color_format(ColorFormat.RGB)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        ppp.input().preprocess().convert_element_type(ov.Type.f32) \
                                .scale([255., 255., 255.])

        ov_model = ppp.build()

        final_xml_path = OUTPUT_DIR / f"{MODEL_NAME}_int8.xml"
        ov.save_model(ov_model, final_xml_path, compress_to_fp16=True)
        print(f"[SUCCESS] Model PPP fixed and saved to: {final_xml_path}")
    else:
        print(f"[ERROR] XML file not found in {export_path}!")
        return

    print(f"[INFO] Packaging {MODEL_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in OUTPUT_DIR.glob("**/*"):
            z.write(f, f.relative_to(OUTPUT_DIR))

    print(f"[SUCCESS] Build completed! Output file: {ZIP_NAME}")

    del model, ov_model, core
    gc.collect()

if __name__ == "__main__":
    main()
