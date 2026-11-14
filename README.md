# openpi_rebot

基于 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) 修改，用于在 Rebot 双臂机器人上微调和部署 π₀.₅。上游的安装与通用用法见原仓库 README。

## 主要修改

- **Rebot UMI 数据与训练**：新增 `pi05_rebot_umi` 训练配置（`src/openpi/training/config.py`）和数据集适配（`src/openpi/training/rebot_dataset.py`）。动作采用 UMI 风格的相对末端位姿（50 步 horizon），从 pi05_base 做 JAX 全参数微调。归一化统计在 `assets/pi05_rebot_umi/`。
- **训练脚本**：`scripts/train.py` 增加验证评估，`checkpoints.py` 调整了 checkpoint 保留策略；另有数据准备、检查与启动脚本：`scripts/prepare_rebot.py`、`scripts/check_rebot_data.py`、`scripts/run_rebot.sh`。
- **推理服务**：`scripts/serve_rebot.py` 是 WebSocket 策略服务，输出 `(50, 26)` 的双臂绝对末端目标，支持 mock 模式。
- **openpi-client**：`WebsocketClientPolicy` 增加可选 `timeout` 和 `close()`，并放开了 NumPy <2 的限制。

## 文档

- 训练：[docs/rebot_umi.md](docs/rebot_umi.md)
- 部署：[docs/rebot_serving.md](docs/rebot_serving.md)
