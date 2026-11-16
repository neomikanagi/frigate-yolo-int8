# 交接

云编译能出 24 个包。RKNN 钉在 Toolkit 2.3.2，产物是 FP16。RK3568 上，2.3.2 和 2.5.0 的 INT8 都把 80 个类别通道量化成 0，框还在；换 COCO128 校准或关掉 `output_optimize` 也一样。不要改回 `do_quantization=True`。zip 名仍带 `int8`，只为现有发布矩阵。FP16 能出类别分，但 yolo11l 约 240 ms，yolo26l 近景约 0.60，过不了 Frigate 默认 0.7。

两个导出都写 `nms=None`。320 模型在打包前必须是单输出 `[1, 84, 2100]`；OpenVINO 输入还必须是 `u8`。不是这个形状就失败，不发布。RK3566 用 `*_rk3568_320.rknn`。Frigate 必须写 `model_type: yolo-generic`，RKNN 检测器没有 `device` 字段。

HD 530 的类别是否还漂，不由这次 RKNN 版本钉死决定。同一份 xml 先比 `device: CPU` 和 `GPU`；只有 GPU 错就是核显执行，不是标签或输出形状。

入口：`README.md`、`build_yolo_rknn.py`、`build_yolo_int8.py`。
