# openpi_rebot

基于 [openpi](https://github.com/Physical-Intelligence/openpi) 修改，用于 Rebot 双臂机器人上 π₀.₅ 的 post-train 与部署。

## 相对上游的修改

Rebot 相关代码集中在两处，其余保持上游原样：

- `src/openpi/rebot/`：动作表示（`actions.py`）、训练数据集（`dataset.py`）、推理服务的协议适配（`link.py`）。
- `scripts/`：`prepare_rebot.py` 数据准备、`check_rebot_data.py` 数据检查、`train_rebot.sh` 训练启动、`serve_rebot.py` 推理服务。

对上游文件的改动只有三处：`training/config.py` 注册 `pi05_rebot_post_train` 配置，`training/data_loader.py` 接入 Rebot 数据集，`scripts/train.py` 增加指标落盘（`training_metrics.jsonl`）并让步数从 1 计。

## 文档

- [训练](docs/rebot_post_train.md)：目录约定、数据准备、冒烟和正式训练。
- [部署](docs/rebot_serving.md)：启动策略服务并与 rebot_control 对接。
