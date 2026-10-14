# Nezha Dual-Encoder Locomotion

## 常用命令速查

以下命令均在仓库根目录执行，并假定已经激活对应的 Python 环境。

无界面训练：

```bash
python legged_gym/scripts/train_nezha_mine.py \
  --headless \
  --num_envs 4096 \
  --max_iterations 100000
```

使用指定 checkpoint 进行 Isaac Gym Play；Play 启动时默认同时导出完整的
TorchScript 策略文件：

```bash
python legged_gym/scripts/play_nezha_mine.py \
  --load_run <运行目录名> \
  --checkpoint best_policy \
  --checkpoint_history_order oldest_first \
  --command_x 1.0 \
  --command_y 0.0 \
  --command_yaw 0.0
```

导出的策略位于
`logs/nezha3_mine/exported/<运行目录名>/policy.pt`。如果只需要导出策略、
不启动 Isaac Gym 仿真：

```bash
python legged_gym/scripts/export_nezha_mine.py \
  --run <运行目录名> \
  --history-order oldest_first
```

使用刚导出的策略启动带界面的 MuJoCo sim-to-sim 仿真（macOS）：

```bash
.venv/bin/mjpython deploy/deploy_mujoco/deploy_mujoco.py \
  --policy logs/nezha3_mine/exported/<运行目录名>/policy.pt \
  --duration 60
```

Linux 可将上述 `.venv/bin/mjpython` 替换为 `python`。MuJoCo 窗口中使用
`W/S` 调节前后速度、`A/D` 调节横向速度、`Q/E` 调节转向，空格停止，
`R` 复位。

本仓库是面向 Nezha 轮腿机器人的训练、Isaac Gym play、策略导出、MuJoCo sim-to-sim 与真机接入框架。默认任务为 `nezha_mine`，不注册也不依赖其他机器人任务。

## 方法与代码结构

门控模态双编码器以 `wheel_gym_CQ/logs/nezha3_mine/Apr10_11-39-09_trimesh_baseline_门控——三类` 保存的实际训练源码为基准，并按 `LZHMine` 分层：

- `rsl_rl/rsl_rl/modules/mine_estimator.py`：历史观测 source encoder、训练期 target encoder、可选三模态 soft gate、模态原型和 prototype swap loss。
- `rsl_rl/rsl_rl/modules/mine_actor_critic.py`：估计值和 latent 与最新一帧本体观测融合后送入 actor；包含与在线推理完全一致的 TorchScript exporter。
- `rsl_rl/rsl_rl/algorithms/mine_ppo.py`：保留原实现的 PPO 参数组和 estimator 独立优化器；策略路径会 detach estimator，因此不会在一次 mini-batch 中被 PPO 重复更新。
- `rsl_rl/rsl_rl/runners/mine_on_policy_runner.py`：rollout、终止状态特权观测、日志和带网络配置的 checkpoint。
- `rsl_rl/rsl_rl/utils/mine_checkpoint.py`：训练续训、play、导出共用的模型配置读取与旧 checkpoint 识别。
- `legged_gym/envs/nezha/`：Nezha 16 维动作、65 维单帧观测、混合腿关节位置/轮关节速度控制与训练配置。
- `deploy/deploy_mujoco/`：统一的 MuJoCo sim-to-sim 部署目录，包含配置、入口、策略运行时、部署策略快照、Nezha MJCF/mesh/地形资产及地形工具。

与原实验一致，所有新训练采用 `oldest-first` 历史顺序。Actor 历史为 `2 x 65 = 130`，critic 历史为 `3 x 375 = 1125`。Actor 使用最后的最新帧，门控网络使用最前的最旧帧。新训练设置 `is_privileged_obs=True`，target encoder 使用下一时刻 critic 历史中最新的完整 375 维特权帧，网络为 `375 → 256 → 128 → 64 → 16`。

特权帧保留原有排列：`[0:65]` 为无噪声本体观测，`[65:371]` 为 306 维额外特权观测，`[371:375]` 为 4 维估计监督真值（机体线速度 3 维、基座高度 1 维）。306 维包括关节加速度 16、力矩 16、基座质量 1、质心 3、KP 16、KD 16、全身接触力 51、地形高度 187。Target 使用仿真真值；source encoder 仍从 130 维本体历史中预测这 4 个量。

三种模态的固定编号为 `0=wheel`、`1=leg`、`2=hybrid`。Apr10 基准使用自标注门控损失，额外的语义锚定默认关闭。该基准还会在 `_reward_dof_vel` 中把四个轮速写为零，因此部署与 MuJoCo 的策略观测也会屏蔽这四个轮速；真实轮速仍用于轮电机 PD 力矩。

`NezhaMINECfgPPO.policy.use_mode_gate` 控制门控，默认 `True`。无门控模式不创建门控网络参数或三模态原型，source latent 直接归一化后进入 Actor；模态损失与模态日志关闭，4 维估计监督和双编码器 prototype swap loss 继续训练。用于双编码器对齐的 32 个原型与三模态原型用途不同，会保留。两种模式都使用 375 维 target，部署时都只输入 130 维本体历史、输出 16 维动作。

## 环境准备

训练、play 和策略导出使用 Python 3.8 的 `.venv-train`；MuJoCo 使用 Python 3.11
的 `.venv-sim`。安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

当前工作机已有的环境名为 `.venv`（训练/导出）和 `.venv-mujoco`（MuJoCo）；可直接激活对应环境，下面的安装流程用于新建环境。

### Python 3.8 训练环境

Isaac Gym Preview 4 默认放在仓库同级的 `../isaacgym/`：

```bash
uv python install 3.8
uv venv .venv-train --python 3.8
source .venv-train/bin/activate

uv pip install \
  torch==2.0.1 \
  torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118

uv pip install \
  numpy==1.23.5 \
  scipy==1.10.1 \
  matplotlib==3.7.5 \
  tensorboard==2.14.0 \
  wandb==0.24.2 \
  pyyaml pillow imageio ninja setuptools

uv pip install --no-deps --no-build-isolation \
  -e ../isaacgym/python \
  -e ./rsl_rl \
  -e .
```

### Python 3.11 MuJoCo 环境

```bash
uv python install 3.11
uv venv .venv-sim --python 3.11
source .venv-sim/bin/activate
uv pip install "mujoco==3.1.2" torch numpy pyyaml
```

## 训练

```bash
source .venv-train/bin/activate
python legged_gym/scripts/train_nezha_mine.py --headless
```

同一个入口可随时选择新实验的门控模式，命令行优先于配置文件：

```bash
# 有门控
python legged_gym/scripts/train_nezha_mine.py --headless --mode_gate on

# 无门控
python legged_gym/scripts/train_nezha_mine.py --headless --mode_gate off
```

不传 `--mode_gate` 时，新训练使用配置中的 `use_mode_gate`；续训和 play 自动采用 checkpoint 的模式。改变结构需要启动新训练；对旧 checkpoint 指定相反的模式会明确报错，不会静默跳过门控。

断点续训：

```bash
python legged_gym/scripts/train_nezha_mine.py \
  --resume \
  --load_run <运行目录名> \
  --checkpoint -1 \
  --headless
```

新训练输出位于 `logs/nezha3_mine/<时间>_<run_name>_gated_target375/` 或 `..._ungated_target375/`，方便区分实验。Checkpoint 保存完整策略配置，续训/导出会恢复门控、target 输入宽度、网络隐藏层和历史顺序。

修改维度后的短训练检查（2 次完整 rollout/PPO 更新）：

```bash
python legged_gym/scripts/train_nezha_mine.py \
  --headless --num_envs 16 --max_iterations 2 --no_wandb \
  --mode_gate off --run_name startup_check
```

改用 `--mode_gate on` 可检查有门控模式。旧的 65 维 target checkpoint 也可自动识别，并按原有结构续训/导出；将旧的 65 维实验改为 375 维需要重新训练。没有配置元数据的旧 checkpoint 默认按历史 Nezha 网络构造，历史顺序默认 `oldest_first`；更早的 `newest_first` 模型需使用显式历史顺序参数。

## Isaac Gym Play 与导出

Play：

```bash
source .venv-train/bin/activate
python legged_gym/scripts/play_nezha_mine.py \
  --load_run <运行目录名> \
  --checkpoint -1
```

Play 自动识别门控；无门控时继续显示速度/高度估计，不生成模态分类和权重。仅在旧模型历史顺序不同的情况下使用 `--checkpoint_history_order newest_first`。

导出策略：

```bash
# 指定 logs/nezha3_mine/ 下的运行目录名
python legged_gym/scripts/export_nezha_mine.py --run <运行目录名>

cp logs/nezha3_mine/exported/<运行目录名>/policy.pt \
  deploy/deploy_mujoco/nezha/policy/policy.pt
```

直接导出单个 checkpoint：

```bash
python legged_gym/scripts/export_nezha_mine.py \
  --checkpoint logs/model_20000.pt \
  --output deploy/deploy_mujoco/nezha/policy/policy.pt
```

导出自动识别门控模式和 65/375 维 target。新导出模型携带门控状态、本体输入维度和历史顺序；旧模型仍受支持。要并存两个实验，可分别导出到 `policy_gated.pt` 和 `policy_ungated.pt`，无需改动导出代码。

## MuJoCo Sim-to-Sim

机器人固定出生在 `x=-5, y=0` 的公共平地区，前方是3条横向一字
排开的评测路线：

- `y=-5 m`：5级20 cm楼梯，踏面70 cm，最高1.0 m；
- `y=0 m`：高0.40 m、长6.0 m的矩形高台；
- `y=+5 m`：170个随机凸起组成的确定性碎石路，最高12 cm。

Linux：

```bash
source .venv-sim/bin/activate
python deploy/deploy_mujoco/deploy_mujoco.py
```

默认持续运行，关闭仿真窗口后退出。

默认配置加载 `nezha/policy/policy.pt`，`mode_gate: auto` 从模型读取模式。切换实验只需选择对应文件（相对路径以 `deploy/deploy_mujoco/` 为基准）：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py --policy nezha/policy/policy_gated.pt
python deploy/deploy_mujoco/deploy_mujoco.py --policy nezha/policy/policy_ungated.pt

# 无窗口验证：非零速度命令会真正运行策略，而非停留在零指令待机状态
python deploy/deploy_mujoco/deploy_mujoco.py \
  --policy nezha/policy/policy_ungated.pt --headless --duration 2 --vx 0.3
```

也可以通过 `--mode-gate on/off` 或 YAML 的 `mode_gate: on/off` 验证所加载的模型是否符合预期；该选项只校验模式，不修改模型内部计算。YAML 中请使用字符串 `"on"` / `"off"`，避免解析成布尔值。模式不匹配会报错。无门控时跳过门控调用及日志；关节控制、观测缩放和动作维度采用同一套实现。`history_steps: 2` 与 `history_order: auto` 适用于这两种默认模型。

macOS：

```bash
source .venv-sim/bin/activate
mjpython deploy/deploy_mujoco/deploy_mujoco.py
```

键盘控制：`W/S` 前后、`A/D` 横移、`Q/E` 转向、空格停止、`R` 复位。

## 真机部署接入

真机侧只需把机器人 SDK 的状态读取和力矩发送接到 `NezhaPolicyRuntime`：

```python
from deploy.deploy_mujoco.runtime import NezhaPolicyRuntime

runtime = NezhaPolicyRuntime(
    policy_path="deploy/deploy_mujoco/nezha/policy/policy.pt",
    default_dof_pos=default_dof_pos,
    p_gains=p_gains,
    d_gains=d_gains,
    torque_limits=torque_limits,
)

actions, torques = runtime.step(
    base_ang_vel=imu_gyro_body,
    projected_gravity=gravity_body,
    commands=[vx, vy, yaw_rate],
    dof_pos=joint_positions,
    dof_vel=joint_velocities,
)
# robot_sdk.send_torques(torques.cpu().numpy())
```

参数模板与 MuJoCo 配置统一放在
`deploy/deploy_mujoco/configs/nezha_mine.yaml`。上真机前必须再核对实际关节顺序、正方向、零位、减速比、力矩限制、急停与通信 watchdog；框架不会假定某一种厂商通信 SDK。

## 回归测试与 Git 文件管理

在训练环境运行门控/无门控的 PPO 更新、TorchScript 导出、旧模型兼容、元数据与优化器恢复、部署推理检查：

```bash
python -m unittest legged_gym.tests.test_mine_modes -v
```

`.gitignore` 忽略虚拟环境（包括 `.venv-mujoco`）、训练日志、checkpoint/导出权重、录像截图、缓存、构建产物和本机账户/代理状态。权重留在本地，通过运行目录和文件名选择。

URDF/Xacro、配套 STL/OBJ 网格、MJCF/XML、配置文件、地形生成脚本，以及仿真场景实际引用的地形 `.bin` / `.json` 继续纳入 Git。不要整体忽略 `resources/` 或机器人资源目录，否则新机器无法加载机器人和地形。
