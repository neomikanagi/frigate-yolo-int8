# Frigate YOLO Auto-Builder

Daily-built YOLO26 detection models for [Frigate NVR](https://frigate.video/) `yolo-generic`, one release per Ultralytics version, for Intel (OpenVINO), Rockchip (RKNN) and AMD Ryzen AI NPUs (ONNX):

| Release asset | Hardware | Frigate detector |
|---|---|---|
| `frigate_yolo26{s,m,l}_fp16_320.zip` | Intel Gen9–Gen11 iGPU (HD/UHD 5xx–6xx, Iris Plus) | `openvino` |
| `frigate_yolo26{s,m,l}_int8_320.zip` | Intel CPU with VNNI/AMX, Xe/Arc GPU, Intel NPU | `openvino` |
| `frigate_yolo26{s,m,l}_rk3568_int8_320.zip` | RK3566 / RK3568 | `rknn` |
| `frigate_yolo26{s,m,l}_rk3576_int8_320.zip`, `…_rk3588_int8_320.zip` | RK3576, RK3588 | `rknn` |
| `frigate_yolo26{s,m,l}_onnx_320.zip` | AMD Ryzen AI (XDNA2) NPU via frigate-xdna; ROCm; CPU | `zmq` / `onnx` |

All models take a 320×320 input and output the raw `[1, 84, 2100]` head (`nms=None`). YOLO26's end-to-end `[1, 300, 6]` head is not what Frigate parses, so a build that produces it fails.

## Measured on real hardware (yolo26l)

Agreement is against FP32 ONNX on real indoor/outdoor camera frames (detections ≥ 0.4 found again with the same class).

| Device | Package | Inference | Agreement with FP32 |
|---|---|---|---|
| i5-6500T, HD 530 iGPU | FP16 | 52 ms | 98% of 60 detections, 0 extra (49 frames) |
| i5-6500T, HD 530 iGPU | INT8 | 55 ms | 93% of 60 detections, 7 extra (49 frames) |
| RK3566 (Panther X2), 1 NPU core | RKNN INT8 | 83 ms in Frigate | 88% of 119 detections; no person or cat missed (98 frames) |
| RK3566 | RKNN FP16 (not published) | ~250 ms in Frigate | 97% of 119 detections |

The HD 530 times are single-request latency inside the Frigate container while Frigate kept running.

HD 530 has no INT8 hardware (OpenVINO reports only FP32/FP16 for it), so INT8 there is FP16 plus quantize overhead: use the FP16 package on Gen9–Gen11 iGPUs.

## Release quality gate

Each package is run on the 28 COCO128 images that are not used for calibration and compared with the FP32 ONNX export: the share of FP32 detections it finds again (same class, IoU > 0.5), the confident detections FP32 does not have, and class-score cosine. A package below its floor fails the build, and a release is published only when every package of every model passes, so a model that loads but reports nothing, or floods one class, does not reach a release. RKNN is checked in the toolkit simulator.

## Frigate configuration

`frigate.yml` takes a single `detectors`/`model` pair; use the block for your hardware.

### Intel (OpenVINO)

```yaml
detectors:
  ov:
    type: openvino
    device: GPU          # or CPU

model:
  model_type: yolo-generic
  path: /config/model_cache/frigate_yolo26l_fp16_320/yolo26l_fp16.xml   # int8: frigate_yolo26l_int8_320/yolo26l_int8.xml
  width: 320
  height: 320
  input_tensor: nchw
  input_pixel_format: rgb
  labelmap_path: /labelmap/coco-80.txt
```

The IR has a uint8 RGB NCHW input built in, so Frigate's tensor goes straight in. `input_tensor` and `input_pixel_format` must match it, or colours swap and scores drop.

### Rockchip (RKNN)

```yaml
detectors:
  rknn:
    type: rknn
    num_cores: 0

model:
  model_type: yolo-generic
  path: /config/model_cache/frigate_yolo26l_rk3568_int8_320/yolo26l_rk3568_320.rknn   # RK3566 uses rk3568
  width: 320
  height: 320
  input_tensor: nhwc
  input_pixel_format: rgb
  input_dtype: int
  labelmap_path: /labelmap/coco-80.txt
```

Built with RKNN Toolkit 2.3.2, the `librknnrt` in Frigate `stable-rk`. Without `model_type` Frigate uses SSD post-processing and returns nothing. Run the container privileged so `/proc/device-tree/compatible` is readable. `device` is not an RKNN field.

Why the RKNN build is hybrid: Frigate reads one tensor that holds box coordinates (0–320 px) and class scores (0–1). As a single INT8 tensor its step is about 1.3, so every class score becomes 0 — the NPU is busy and no object is ever reported. The network runs in INT8 up to the last head convolutions; only the box decode, sigmoid and concat run in FP16. On RK3566 the stock runtime returned shifted, duplicated boxes when convolutions or attention were also FP16, so nothing else is. That was measured on the board; the simulator does not show it.

### AMD Ryzen AI NPU (XDNA2), e.g. Ryzen AI 9 HX (PRO) 370

Frigate reaches XDNA2 NPUs through the community [frigate-xdna](https://github.com/mitchins/frigate-xdna) sidecar, which compiles a float ONNX to BF16 on the machine (first compile 7–25 min, then cached). Use the ONNX package as its local model:

```yaml
# compose: mount the unzipped folder at /models in the xdna container, FXDNA_MODELS=local://yolo26l_320
detectors:
  xdna:
    type: zmq
    endpoint: tcp://xdna:5555

model:
  model_type: yolo-generic
  path: /config/model_cache/frigate_yolo26l_onnx_320/yolo26l_320.onnx
  width: 320
  height: 320
  input_tensor: nchw
  input_pixel_format: rgb
  input_dtype: float
  labelmap_path: /labelmap/coco-80.txt
```

frigate-xdna's own model inspector classifies the graph as its supported `yolo-raw` contract (`images` float32 `[1, 3, 320, 320]` in 0–1, output `[1, 84, 2100]`). frigate-xdna is qualified on Strix Halo (Ryzen AI Max 300) with YOLOv9; Strix Point (Ryzen AI 300, HX 370) is XDNA2 too but not yet qualified there, and YOLO26 on its compiler is untested. The same ONNX runs on Frigate's `onnx` detector with ROCm (Radeon iGPU) or CPU.

## Keeping a device up to date

[`tools/frigate-yolo-sync.sh`](tools/frigate-yolo-sync.sh) runs from cron on the Frigate host. It installs a release asset when its sha256 changes (a rebuild under the same tag counts), runs the new model once inside the Frigate container before switching, restarts Frigate, and puts the previous model back if Frigate is not healthy afterwards. Unused `frigate_*` model folders are removed.

```bash
install -m 755 tools/frigate-yolo-sync.sh /usr/local/bin/frigate-yolo-sync.sh
cp tools/frigate-yolo-sync.example /etc/default/frigate-yolo-sync   # set ASSETS
echo '30 6 * * * root /usr/local/bin/frigate-yolo-sync.sh >>/var/log/frigate-yolo-sync.log 2>&1' > /etc/cron.d/frigate-yolo-sync
```

## Builds

GitHub Actions checks daily at 03:00 UTC and builds when the YOLO26 weights, the Ultralytics export graph or this repository's build files change; **Run workflow** forces a build. Each release replaces the previous one.

Local build, one model:

```bash
docker build --build-arg ULTRALYTICS_VERSION=<version> -t yolo-builder .
docker run --rm -e MODEL_NAME=yolo26l -v "$PWD:/app" yolo-builder python3 build_openvino.py   # also build_rknn.py, build_onnx.py
```

Tests: `python3 -m unittest discover -s tests -v` (needs numpy).
