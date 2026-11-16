"""OpenVINO packages for Frigate: INT8 (NNCF) and FP16, with uint8 RGB NCHW input baked in.

INT8 pays off on CPUs with VNNI/AMX and on Xe/Arc GPUs. Gen9-Gen11 iGPUs (HD/UHD 5xx-6xx, Iris Plus)
have no INT8 hardware: they run INT8 IR as FP16 plus quantize/dequantize overhead, so the FP16
package is faster and more accurate there.
"""
import gc
import shutil
from pathlib import Path

import nncf
import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor
from ultralytics import YOLO

from yolo_common import (
    IMG_SIZE, RAW_OUTPUT, coco128_images, export_onnx, gate, load_rgb, model_name, nchw_float,
    onnx_reference, split_images, write_zip,
)

# Measured on the COCO128 holdout; the INT8 floor catches broken calibration, not normal INT8 drift.
GATES = {
    "int8": dict(min_recall=0.85, max_extra=8, min_cosine=0.90),
    "fp16": dict(min_recall=0.97, max_extra=2, min_cosine=0.99),
}


def bake_uint8_rgb_input(ov_model):
    """Frigate then passes its uint8 RGB NCHW tensor straight through, with no float copy on the CPU."""
    ppp = PrePostProcessor(ov_model)
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NCHW")).set_color_format(ColorFormat.RGB)
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    ppp.input().preprocess().convert_element_type(ov.Type.f32).scale([255.0, 255.0, 255.0])
    return ppp.build()


def assert_frigate_ready(ov_model):
    if len(ov_model.outputs) != 1:
        raise RuntimeError(f"expected one output, got {len(ov_model.outputs)}")
    shape = [dim.get_length() for dim in ov_model.outputs[0].partial_shape]
    if shape != RAW_OUTPUT:
        raise RuntimeError(f"OpenVINO output is {shape}, want {RAW_OUTPUT}")
    if ov_model.inputs[0].get_element_type() != ov.Type.u8:
        raise RuntimeError(f"input must be u8, got {ov_model.inputs[0].get_element_type()}")


def head_ignored_scope(ov_model):
    """Keep the Detect head and every Sigmoid in float, as Ultralytics does for its own INT8 export."""
    ops = ov_model.get_ordered_ops()
    sigmoids = [op.get_friendly_name() for op in ops if op.get_type_name() == "Sigmoid"]
    head = sigmoids[-1].split("/", 1)[0]
    names = [
        op.get_friendly_name() for op in ops
        if op.get_type_name() == "Sigmoid" or op.get_friendly_name().startswith((f"{head}/", f"{head}.dfl"))
    ]
    return nncf.IgnoredScope(names=names, validate=False)


def main():
    name = model_name()
    calibration, holdout = split_images(coco128_images())
    holdout_rgb = [load_rgb(path) for path in holdout]

    references = onnx_reference(export_onnx(YOLO(f"{name}.pt")), holdout_rgb)

    exported = Path(YOLO(f"{name}.pt").export(format="openvino", imgsz=IMG_SIZE, half=False, dynamic=False, nms=None))
    core = ov.Core()
    fp32 = core.read_model(next(exported.glob("*.xml")))

    print(f"[INFO] Quantizing {name} with NNCF on {len(calibration)} COCO128 images")
    int8 = nncf.quantize(
        fp32,
        nncf.Dataset([nchw_float(load_rgb(path)) for path in calibration]),
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=len(calibration),
        ignored_scope=head_ignored_scope(fp32),
    )

    for precision, ov_model in (("int8", int8), ("fp16", fp32)):
        ov_model = bake_uint8_rgb_input(ov_model)
        assert_frigate_ready(ov_model)
        out_dir = Path(f"{name}_{precision}_openvino_{IMG_SIZE}")
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir()
        xml = out_dir / f"{name}_{precision}.xml"
        ov.save_model(ov_model, xml, compress_to_fp16=True)

        compiled = core.compile_model(str(xml), "CPU")
        outputs = [compiled({0: rgb.transpose(2, 0, 1)[None].copy()})[compiled.outputs[0]] for rgb in holdout_rgb]
        gate(f"{name} OpenVINO {precision}", references, outputs, **GATES[precision])

        zip_name = f"frigate_{name}_{precision}_{IMG_SIZE}.zip"
        write_zip(zip_name, [xml, xml.with_suffix(".bin")])
        print(f"[SUCCESS] {zip_name}")
        del compiled
        gc.collect()


if __name__ == "__main__":
    main()
