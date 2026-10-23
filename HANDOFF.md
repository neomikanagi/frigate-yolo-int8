# 多 AI 交接

## 2026-10-23 — Pi

- **状态：** 修复已完成；本地 12 项回归测试、actionlint 通过，准备推送并验证 GitHub 全量构建。
- **改了：** RKNN 禁用编译器日志明确指出崩溃的两条融合规则，保持 INT8；固定 RKNN/ONNX 优化依赖；编译失败、缺标定图和空输出抛错，并在 finally 释放编译器。OpenVINO 缺 XML 也不再静默成功。Docker 使用探测到的准确 Ultralytics 版本。
- **为什么 / 思路：** 邮件对应 run 39921754717（8.4.184），OpenVINO 六项成功，RKNN 在 `seperate_conv_rs_split`（YOLO11）和 `fuse_exmatmul_add_mul_exsoftmax13_exmatmul_to_sdpa`（YOLO26）内部 IndexError。原 release job 的 always() 即使矩阵失败也发布，且上游检查只看 tag 存在，导致残缺版本被当作成功。现仅完整矩阵成功且 24 个 ZIP 的模型成员非空、CRC 通过才发布；已有版本缺文件会自动重建，不删历史版本。
- **请你：** 本轮继续跟踪完整 Actions 运行并更新验收结果。不要恢复失败后发布或仅凭 tag 跳过构建的逻辑；没有修改正在运行的 Frigate、模型缓存、宿主服务或排期。
