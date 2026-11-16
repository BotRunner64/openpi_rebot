# Rebot π₀.₅ post-train

从 `pi05_base` 全参数微调，所有 episode 都用于训练，不留验证集。

## 目录

数据、模型和日志都在仓库目录下，均已 gitignore：

```text
data/<task>/          LeRobot v3 数据集，每个任务一个目录（packing、unscrew…）
data/post_train_h50/  prepare_rebot.py 的输出：训练窗口、动作块、归一化统计
models/pi05_base/      基座权重
models/big_vision/     PaliGemma tokenizer
models/<config>/<exp>/ 训练输出：checkpoint 和 training_metrics.jsonl
logs/                  训练日志、离线 W&B、JAX 编译缓存
```

启智集群上 `data/<task>` 和 `models` 是软链接，分别指向项目存储的 `public/dataset/rebot/<task>` 和 `public/model`。Python 环境是 conda 环境 `openpi_rebot`。

数据来自 ModelScope 的 `BingqianWu/rebot-manipulation`，由 rebot_control 的 `rebot export` 生成。

## 流程

```bash
conda activate openpi_rebot

# 1. 准备数据：默认使用 data/ 下所有数据集，也可以用 --tasks 指定
python scripts/prepare_rebot.py --overwrite
python scripts/check_rebot_data.py

# 2. 冒烟（2 张 GPU，50 步，几分钟）
scripts/train_rebot.sh smoke pi05_rebot_post_train_smoke

# 3. 正式训练（单节点 8 张 GPU）
scripts/train_rebot.sh <实验名>
```

- 增减数据集后重新跑第 1 步。它只读取源数据，约十几秒，同时重新计算归一化统计。
- `train_rebot.sh` 总是带 `--resume`：同一个实验名再次运行就是续训，平台重启任务后会接着训而不是覆盖。换一组数据或超参时换一个实验名。
- 训练节点不联网，W&B 以离线模式写到 `logs/wandb`。
- 第二个参数之后的内容原样传给 `scripts/train.py`，例如 `scripts/train_rebot.sh exp pi05_rebot_post_train --num-train-steps 30000`。

## 配置

`src/openpi/training/config.py` 里的 `pi05_rebot_post_train`：

| 项 | 值 |
|---|---|
| 模型 | π₀.₅，action horizon 50，bfloat16，不冻结参数 |
| 优化 | AdamW，峰值学习率 2.5e-5，1000 步 warmup，cosine，EMA 0.99 |
| 批量 | 全局 64，FSDP 每组 2 张 GPU |
| 步数 | 20000，每 1000 步存一次；保留最新一个和每 5000 步一个 |
| 数据读取 | 32 个 worker |

`pi05_rebot_post_train_smoke` 是同一配置的缩小版（batch 16、50 步、第 25 和 50 步存 checkpoint）。

读取视频是瓶颈。在训练节点上实测 8 / 16 / 24 / 32 个 worker 分别为 67 / 130 / 168 / 244 样本每秒，而 8 张 H200 约 0.5 秒一步，需要约 130 样本每秒。

## 状态与动作

状态是 26 维，左臂后右臂，每臂为 xyz、按列展开的旋转矩阵（R9）、夹爪开合度。

模型的动作目标相对于当前观测位姿：对每条手臂、观测时刻 t，第 k 个目标是 `inv(T[t]) @ T[t+k+1]`（k = 0..49），存为 3 个位置分量、旋转矩阵前两列（6D）和**绝对**的未来夹爪开合度，每臂 10 维，共 20 维，由模型补齐到 32 维。推理时输出变换先反归一化，再还原成绝对的 26 维目标状态。

注意事项：

- 源数据里 `action[t]` 就是 `state[t+1]`。动作块直接从状态构造，不要再移位一次；准备脚本会校验这一点。
- 训练窗口步长为 1，要求同一 episode 内有 50 个真实的未来状态，不做末尾填充，也不跨 episode。
- 位置和夹爪用 q01/q99 归一化，统计覆盖所有 episode 的状态和所有未来目标；旋转分量不做归一化。
- 图像在训练时从原始视频按需解码，不预先缓存；缩放和增强沿用 openpi 的默认处理。

对应代码在 `src/openpi/rebot/actions.py`，测试：`python -m pytest src/openpi/rebot`。
