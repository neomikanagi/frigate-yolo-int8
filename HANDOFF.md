# 交接

## 现在能否用

- 流水线产出 YOLO26 s/m/l × 6 个包（OpenVINO INT8、OpenVINO FP16、ONNX、RKNN rk3568/rk3576/rk3588），每个包过质量闸门才发布。yolo26l 三个构建脚本已在本地按 CI 方式跑通；s/m 的 RKNN 闸门数值见下。
- 实机验证（yolo26l）：HD 530 核显 FP16 52 ms、INT8 55 ms 且 INT8 多 7 个误检，Gen9 核显用 FP16；RK3566 RKNN INT8 混合版在 Frigate 界面 83 ms，98 张家用摄像头画面上人/猫 0 漏检。
- AMD Ryzen AI（XDNA2）NPU 这一路产出 FP32 ONNX，frigate-xdna 自带检查器判定为 `yolo-raw` 兼容；NPU 编译由 sidecar 在机器上做（BF16）。HX 370 未经 frigate-xdna 认证，YOLO26 在其编译器上未测，买到机器后实测。

## 关键决定与坑

- RKNN 不能整网 INT8：输出张量把 0–320 的框坐标和 0–1 的类别分拼在一起，单一 INT8 刻度下类别分全变 0（NPU 有负载但 Frigate 不出框）。现做法是 `custom_hybrid` 只把检测头最后几层卷积之后的解码尾巴放 FP16（`build_rknn.py:head_tail`）。
- RK3566 + 运行时 2.3.2 上，混合模型里一旦有 **FP16 卷积或注意力**，板上就出错位重复框，模拟器看不出来。试过注意力 FP16、整个 PSA 块 FP16、检测头卷积 FP16、Toolkit 2.5.0 编译，板上都更差；只放解码尾巴是板上最好的。新改法必须上板评估，不能只信模拟器。
- RKNN 量化：min-max（normal）+ 50 张校准；20/50/100 张板上差异在噪声内。mmse 单个 l 模型 >1.5 h，CI 不可用；KL 板上更差。RK3568 不支持 w8a16。
- Toolkit 钉 2.3.2 = Frigate stable-rk 的 librknnrt。Frigate 换运行时版本时要同步改 `requirements.txt` 并重新上板评估。
- HD 530 的 OpenVINO GPU 能力只有 FP32/FP16，没有 INT8：INT8 IR 在 Gen9–Gen11 上不会更快。
- 发布 tag = Ultralytics 版本；构建文件变化（`upstream_watch.PIPELINE`）也会触发重建并覆盖同 tag 的 release。设备端 `tools/frigate-yolo-sync.sh` 按资产 sha256 判断更新，所以同 tag 重建也会被拉取。

## 未完成

- 无。

## 入口

`README.md`（用法与实测数据）、`build_rknn.py`、`build_openvino.py`、`build_onnx.py`、`yolo_common.py`（导出、闸门）、`tools/frigate-yolo-sync.sh`。
