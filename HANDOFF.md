# 交接

云编译能出 24 个包。RKNN 钉在 Toolkit 2.3.2 / `w8a8`，对齐 Frigate `stable-rk` 的 `librknnrt 2.3.2`。不要改回 2.5.0，也不要加 `disable_rules`：那两条融合在 2.5.0 上会崩，关掉后能出文件，但精度没有核过，而且 2.5 的模型对不上 2.3.2 runtime。

两个导出都写 `nms=None`。320 模型在打包前必须是单输出 `[1, 84, 2100]`；OpenVINO 输入还必须是 `u8`。不是这个形状就失败，不发布。RK3566 用 `*_rk3568_320.rknn`。Frigate 必须写 `model_type: yolo-generic`，RKNN 检测器没有 `device` 字段。

HD 530 的类别是否还漂，不由这次 RKNN 版本钉死决定。同一份 xml 先比 `device: CPU` 和 `GPU`；只有 GPU 错就是核显执行，不是标签或输出形状。

入口：`README.md`、`build_yolo_rknn.py`、`build_yolo_int8.py`。
