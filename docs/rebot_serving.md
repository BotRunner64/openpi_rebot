# Rebot UMI policy server

在本 openpi 仓库根目录执行：

```bash
conda activate ../conda
python scripts/serve_rebot.py --port 8000
```

服务只依赖 openpi。本地 checkpoint、缓存和模型配置均从本仓库所在的部署目录解析，不需要 rebot_control 的路径或 Python 模块。默认加载 `../checkpoints/pi05_rebot_umi_20000`，采样 10 步；可以用 `--checkpoint` 和 `--num-steps` 覆盖。先完成合成观测预热，再进入 WebSocket 监听。运行时每次推理使用客户端提交的 prompt，预热文本不会作为默认任务指令。

默认监听 `127.0.0.1:8000`。跨机器连接时显式指定 `--host 0.0.0.0`，客户端使用服务所在主机地址。WebSocket 使用 openpi 自带的 NumPy MessagePack 协议，元数据声明列优先旋转、30Hz、50 步 horizon、`absolute_eef26`。响应动作形状为 `(50,26)`：先左臂再右臂，每臂 xyz + R9 + openness，UMI 相对动作已还原为绝对未来状态。

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

以上服务均以前台进程手动启动，Ctrl-C 关闭。Mock 服务会明确声明 mock 元数据，实机控制端拒绝使用 mock 策略。

## 控制端 SDK

控制端使用本仓库 `packages/openpi-client` 的官方客户端和图像工具，不再自写协议。该客户端增加了可选 `timeout` 与 `close()`，默认不传 timeout 的等待行为保留；纯客户端支持 NumPy 1/2，模型环境的 NumPy <2 约束保持不变。

在本仓库构建 wheel：

```bash
uv build packages/openpi-client --wheel --out-dir ../wheels
```

把 wheel 安装进控制环境即可，运行时不依赖本仓库目录或跨仓库 PYTHONPATH。
