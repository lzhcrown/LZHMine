# Nezha Gated-Modal Locomotion

本仓库是面向 Nezha 轮腿机器人的训练、Isaac Gym play、策略导出、MuJoCo sim-to-sim 与真机接入框架。默认任务为 `nezha_mine`，不注册也不依赖其他机器人任务。

## 方法与代码结构

门控模态双编码器以 `wheel_gym_CQ/logs/nezha3_mine/Apr10_11-39-09_trimesh_baseline_门控——三类` 保存的实际训练源码为基准，并按 `LZHMine` 分层：

- `rsl_rl/rsl_rl/modules/mine_estimator.py`：历史观测 source encoder、训练期 target encoder、三模态 soft gate、模态原型和 prototype swap loss。
- `rsl_rl/rsl_rl/modules/mine_actor_critic.py`：估计值和门控 latent 与最新一帧本体观测融合后送入 actor；包含与在线推理完全一致的 TorchScript exporter。
- `rsl_rl/rsl_rl/algorithms/mine_ppo.py`：保留原实现的 PPO 参数组和 estimator 独立优化器；策略路径会 detach estimator，因此不会在一次 mini-batch 中被 PPO 重复更新。
- `rsl_rl/rsl_rl/runners/mine_on_policy_runner.py`：rollout、终止状态特权观测、日志、断点与 best policy。
- `legged_gym/envs/nezha/`：Nezha 16 维动作、65 维单帧观测、混合腿关节位置/轮关节速度控制与训练配置。
- `deploy/deploy_mujoco/`：统一的 MuJoCo sim-to-sim 部署目录，包含配置、入口、策略运行时、部署策略快照、Nezha MJCF/mesh/地形资产及地形工具。

与原实验一致，所有新训练采用 `oldest-first` 历史顺序。Actor 历史为 `2 x 65 = 130`，critic 历史为 `3 x 375 = 1125`。原实现有一个必须保留的特殊约定：Actor 使用最后的最新帧，门控网络使用最前的最旧帧。`is_privileged_obs=False`，所以 target encoder 使用最新375维特权帧的前65维；四个监督目标仍是该帧最后的机体线速度和基座高度。

三种模态的固定编号为 `0=wheel`、`1=leg`、`2=hybrid`。Apr10 基准使用自标注门控损失，额外的语义锚定默认关闭。该基准还会在 `_reward_dof_vel` 中把四个轮速写为零，因此部署与 MuJoCo 的策略观测也会屏蔽这四个轮速；真实轮速仍用于轮电机 PD 力矩。

## 环境准备

训练、play 和策略导出使用 Python 3.8 的 `.venv-train`；MuJoCo 使用 Python 3.11
的 `.venv-sim`。安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

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

断点续训：

```bash
python legged_gym/scripts/train_nezha_mine.py \
  --resume \
  --load_run <运行目录名> \
  --checkpoint -1 \
  --headless
```

训练输出位于 `logs/nezha3_mine/<时间>_trimesh_baseline_test/`。

## Isaac Gym Play 与导出

Play：

```bash
source .venv-train/bin/activate
python legged_gym/scripts/play_nezha_mine.py \
  --load_run <运行目录名> \
  --checkpoint best_policy
```

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
