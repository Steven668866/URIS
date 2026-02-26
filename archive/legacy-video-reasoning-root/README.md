# Legacy Video Reasoning Root（归档说明）

本目录保存的是旧版“单文件视频问答助手”阶段的根目录文件，已从项目根目录移动到这里以减少混杂。

子目录说明：

- `docs/`：旧版使用指南、优化总结、Colab 部署文档、语音/摄像头说明等
- `apps/`：旧版 `app_colab_a100.py`、优化版、离线版与 Colab notebook
- `scripts/`：旧版辅助脚本（模型下载、视频抽帧、启动脚本等）
- `tests/`：旧版根目录测试脚本与 `test_samples/`
- `misc/`：历史误生成目录或其他杂项

保留原因：

- 这些内容对回溯旧方案、复现实验记录仍有价值
- 但不属于当前平台主线（`app.py` + `src/uris_platform/`）
