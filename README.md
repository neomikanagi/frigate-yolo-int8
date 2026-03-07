# Frigate YOLO INT8 Auto-Builder

An automated build pipeline that generates highly optimized OpenVINO INT8 quantized YOLO models (YOLO11 & YOLO26), specifically tailored for [Frigate NVR](https://frigate.video/).

## 🚀 Features

* **Upstream Tracking:** Automatically monitors the official `ultralytics` PyPI package. When a new version is released, it triggers a matrix build and publishes a new GitHub Release automatically.
* **Hardware Optimized:** Uses `NNCF` mixed INT8 quantization for the best balance of accuracy and inference speed, especially on older Intel CPUs and iGPUs (e.g., Intel 6th Gen Skylake).
* **Zero-Copy Pipeline (PPP Fixed):** The models are built with OpenVINO PrePostProcessor (PPP) hardcoded. They directly accept `uint8` format, `NCHW` layout, and `RGB` color space. This prevents Frigate from doing expensive CPU-bound floating-point conversions and saves memory bandwidth.
* **Included Models:** * YOLO11 (Small, Medium, Large) - Recommended for older hardware for supreme stability and low latency.
  * YOLO26 (Small, Medium, Large) - The latest NMS-Free architecture (Note: May cause numeric overflow/hallucinations on older non-VNNI hardware).

## 🛠️ How to Use in Frigate

To get the maximum performance and avoid the "Smurf/Avatar effect" (wrong color channels lowering confidence scores), you **must** configure your `frigate.yml` to match the hardcoded PPP settings.

Add the following to your `frigate.yml`:

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
