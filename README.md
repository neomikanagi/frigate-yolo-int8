# Frigate YOLO INT8 Auto-Builder

An automated build pipeline that generates highly optimized OpenVINO INT8 quantized YOLO models (YOLO11 & YOLO26), specifically tailored for [Frigate NVR](https://frigate.video/).

## 🚀 Features

* **Upstream Tracking:** A daily check looks at ultralytics, and a build starts only when the pretrained weights or the OpenVINO/ONNX export graph change. A new patch version by itself does not rebuild. A release is published only after all six models succeed and all 24 archives pass validation, and older releases are removed so only the latest assets remain. Incomplete releases are retried rather than treated as finished.
* **Hardware Optimized:** OpenVINO INT8 for Intel CPUs. On a Gen9 iGPU such as HD 530, compare `device: CPU` before trusting GPU detections; a wrong class there is the GPU plugin, not the label map.
* **Zero-Copy Pipeline (PPP Fixed):** The models are built with OpenVINO PrePostProcessor (PPP) hardcoded. They directly accept `uint8` format, `NCHW` layout, and `RGB` color space. This prevents Frigate from doing expensive CPU-bound floating-point conversions and saves memory bandwidth.
* **Included Models:** YOLO11 and YOLO26, small / medium / large. Both are exported with `nms=None`, so the packaged head is the raw `[1, 84, 2100]` tensor Frigate `yolo-generic` parses. A build that emits the end-to-end `[1, 300, 6]` head fails instead of being published.

## Build and test

Run regression tests with `python3 -m unittest discover -s tests -v` (stdlib only).
For a local container build, use `docker build --build-arg ULTRALYTICS_VERSION=<version> -t yolo-builder .`.
The GitHub Actions **Run workflow** button forces a full rebuild, including an existing release.

RKNN Toolkit 2.3.2 is pinned because Frigate `stable-rk` ships `librknnrt` 2.3.2.
The RKNN archives are FP16, not INT8: Toolkit 2.3.2 INT8 keeps the box channels
and zeroes the 80 class channels on RK3568, so the NPU runs and Frigate returns
no objects. Mean is 0 and std is 255. An RK3566 board loads the rk3568 archive.
The zip names still say `int8` so the existing release matrix stays valid.

## 🛠️ How to Use in Frigate

To get the maximum performance and avoid the "Smurf/Avatar effect" (wrong color channels lowering confidence scores), you **must** configure your `frigate.yml` to match the hardcoded PPP settings.

Pick **one** of the two options below depending on your hardware — do not merge them, `frigate.yml` only allows a single `detectors`/`model` section.

### Option A: Intel CPU/iGPU (OpenVINO)

```yaml
detectors:
  ov:
    type: openvino
    device: GPU # or CPU

model:
  model_type: yolo-generic
  path: /config/model_cache/yolo11l_int8.xml  # Point to your downloaded model
  width: 320
  height: 320
  input_tensor: nchw         # CRITICAL: Matches the PrePostProcessor layout
  input_pixel_format: rgb    # CRITICAL: Prevents BGR/RGB color inversion
  labelmap_path: /labelmap/coco-80.txt
```

On HD 530, start with `device: CPU`. Use `GPU` only if the same model does not flood one class.

### Option B: Rockchip NPU (RKNN)

```yaml
detectors:
  rknn:
    type: rknn
    num_cores: 0

model:
  path: /config/model_cache/yolo26l_rk3568_320.rknn  # RK3566 uses this rk3568 build
  model_type: yolo-generic
  width: 320
  height: 320
  input_tensor: nhwc
  input_pixel_format: rgb
  input_dtype: int
  labelmap_path: /labelmap/coco-80.txt
```

`device` is not a Frigate RKNN field. Omitting `model_type` selects SSD postprocessing, so the NPU can be busy while no boxes are returned. Run the container privileged so `/proc/device-tree/compatible` is visible.
