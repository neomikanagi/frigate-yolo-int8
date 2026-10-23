# Frigate YOLO INT8 Auto-Builder

An automated build pipeline that generates highly optimized OpenVINO INT8 quantized YOLO models (YOLO11 & YOLO26), specifically tailored for [Frigate NVR](https://frigate.video/).

## 🚀 Features

* **Upstream Tracking:** Automatically monitors the official `ultralytics` PyPI package and builds that exact version. A release is published only after all six models succeed and all 24 OpenVINO/RKNN archives pass validation. Incomplete releases are retried rather than treated as finished.
* **Hardware Optimized:** Uses `NNCF` mixed INT8 quantization for the best balance of accuracy and inference speed, especially on older Intel CPUs and iGPUs (e.g., Intel 6th Gen Skylake).
* **Zero-Copy Pipeline (PPP Fixed):** The models are built with OpenVINO PrePostProcessor (PPP) hardcoded. They directly accept `uint8` format, `NCHW` layout, and `RGB` color space. This prevents Frigate from doing expensive CPU-bound floating-point conversions and saves memory bandwidth.
* **Included Models:** * YOLO11 (Small, Medium, Large) - Recommended for older hardware for supreme stability and low latency.
  * YOLO26 (Small, Medium, Large) - The latest NMS-Free architecture (Note: May cause numeric overflow/hallucinations on older non-VNNI hardware).

## Build and test

Run regression tests with `python3 -m unittest discover -s tests -v` (stdlib only).
For a local container build, use `docker build --build-arg ULTRALYTICS_VERSION=<version> -t yolo-builder .`.
The GitHub Actions **Run workflow** button forces a full rebuild, including an existing release.

RKNN Toolkit 2.5.0 is pinned. Its `seperate_conv_rs_split` and
`fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa` fusion passes crash on the
YOLO11/YOLO26 graphs, so these two optimizations are disabled as recommended by
the compiler diagnostics. INT8 quantization remains enabled for all three targets.

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
  labelmap_path: /config/labelmap.txt
```

### Option B: Rockchip NPU (RKNN)

```yaml
detectors:
  rknn:
    type: rknn
    device: rk3588

model:
  path: /config/model_cache/yolo11l_rk3588_320.rknn
  width: 320
  height: 320
  input_tensor: nhwc
  input_pixel_format: rgb
```
