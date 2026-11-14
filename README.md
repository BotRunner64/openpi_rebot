# openpi_rebot

基于 [openpi](https://github.com/Physical-Intelligence/openpi) 修改，用于 Rebot 双臂机器人上 π₀.₅ 的 post-train 与部署。

## 主要修改

- **Post-train**：新增 Rebot 数据集适配和训练配置，动作为相对末端位姿，从 pi05_base 全参数微调。训练中加入了验证评估，并调整了 checkpoint 保留策略。
- **推理服务**：新增 WebSocket 策略服务，输出双臂绝对末端目标。
- **openpi-client**：增加可选 `timeout` 和 `close()`，支持 NumPy 2。

详见 [训练文档](docs/rebot_post_train.md) 和 [部署文档](docs/rebot_serving.md)。
