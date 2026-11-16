# Rebot policy server

在本 openpi 仓库根目录执行：

```bash
conda activate openpi_rebot
python scripts/serve_rebot.py --port 8000
```

服务只依赖 openpi。本地 checkpoint、tokenizer 和模型配置均从本仓库目录解析，不需要 rebot_control 的路径或 Python 模块。默认加载 `models/pi05_rebot_post_train_20000`，采样 10 步；可以用 `--checkpoint` 和 `--num-steps` 覆盖。先完成合成观测预热，再进入 WebSocket 监听。运行时每次推理使用客户端提交的 prompt，预热文本不会作为默认任务指令。

默认监听 `127.0.0.1:8000`。跨机器连接时显式指定 `--host 0.0.0.0`，客户端使用服务所在主机地址。

通讯使用 rebot_control 仓库定义的策略无关协议 `rebot-policy-link/1`（WebSocket + 普通 MessagePack，协议说明见该仓库 `docs/policy_link.md`）。控制端发来的观测是采集格式的各路流：`state/left_eef` 等四路状态表和 `rgb/top` 等三路 JPEG；回复是这四路状态流的未来行。所有模型相关的转换都在 `src/openpi/rebot/link.py`：

- 相机名 `top`、`left_wrist`、`right_wrist` 映射到 `base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`，JPEG 解码后按原分辨率交给模型变换缩放到 224。
- 每路流取最后一行，按左臂后右臂拼成 26 维 state（每臂 xyz + 列优先 R9 + openness）。
- 模型输出的 `(50,26)` 绝对未来状态拆回四路流，第 k 行的时间戳为观测参考时刻后 (k+1)/30 秒。

服务元数据声明需要的状态流和相机流，控制端据此要求相机参数。

在 rebot_control 仓库根目录启动客户端：

```bash
conda activate rebot_control
python rebot_policy.py --port 8000 \
  --instruction "YOUR_TRAINING_INSTRUCTION" --hardware \
  --camera top=/dev/video4 \
  --camera left_wrist=/dev/video0 \
  --camera right_wrist=/dev/video2
```

instruction 替换为训练任务原文，相机编号按实际设备映射。控制端负责相机、实测 FK、异步 action chunk 调度、IK 和电机；本服务只接收观测并返回模型输出。

无模型/硬件的传输验证，在本仓库运行：

```bash
python scripts/serve_rebot.py --mock-policy --mock-latency .12 --port 8001
```

在控制仓库运行：

```bash
python rebot_policy.py --port 8001 --instruction hold --mock-cameras --duration 3
```

以上服务均以前台进程手动启动，Ctrl-C 关闭。Mock 服务会明确声明 mock 元数据，实机控制端拒绝使用 mock 策略。Mock 策略来自协议包，不加载 openpi。

## 协议包

服务脚本导入 `rebot_policy_link`。它是 rebot_control 仓库 `policy_link/` 下的单文件纯 Python 包，在该仓库构建 wheel 后装进本环境：

```bash
conda activate openpi_rebot
uv build /path/to/rebot_control/policy_link --wheel --out-dir /tmp/rebot_policy_link
uv pip install --no-deps /tmp/rebot_policy_link/rebot_policy_link-0.1.0-py3-none-any.whl
```

运行时两个仓库互不依赖对方目录，控制端不使用 `openpi-client`。适配层测试：`python -m pytest src/openpi/rebot/link_test.py`。
