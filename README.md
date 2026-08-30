# Nezha Gated-Modal Locomotion

本仓库是面向 Nezha 轮腿机器人的训练、Isaac Gym play、策略导出、MuJoCo sim-to-sim 与真机接入框架。默认任务为 `nezha_mine`，不注册也不依赖其他机器人任务。

## 方法与代码结构

门控模态双编码器从 `LZHMine` 的 MINE 路线迁入，并按 `LZHloco` 分层：

- `rsl_rl/rsl_rl/modules/mine_estimator.py`：历史观测 source encoder、训练期特权观测 target encoder、三模态 soft gate、模态原型与 prototype swap loss。
- `rsl_rl/rsl_rl/modules/mine_actor_critic.py`：估计值和门控 latent 与最新一帧本体观测融合后送入 actor；包含与在线推理完全一致的 TorchScript exporter。
- `rsl_rl/rsl_rl/algorithms/mine_ppo.py`：PPO 与 estimator 独立优化器，避免同一 estimator 参数被两个 Adam 状态重复更新。
- `rsl_rl/rsl_rl/runners/mine_on_policy_runner.py`：rollout、终止状态特权观测、日志、断点与 best policy。
- `legged_gym/envs/nezha/`：Nezha 16 维动作、65 维单帧观测、混合腿关节位置/轮关节速度控制与训练配置。
- `deploy/nezha/`：不依赖仿真器的观测历史和混合控制运行时，供真机 SDK 接入。
- `mujoco/nezha_sim.py`：复用同一部署运行时的 sim-to-sim 入口。

所有阶段统一采用 `newest-first` 历史顺序：当前帧位于扁平历史的最前面。Actor 历史为 `2 x 65 = 130`；四个估计目标是机体坐标系线速度和基座高度。

## 环境准备

需要 NVIDIA Isaac Gym Preview 4、可用的 CUDA/PyTorch 环境以及 `ninja`（Isaac Gym 的 `gymtorch` 扩展首次导入时需要）：

```bash
cd /home/bit/LZHLoco/LZHloco
source env/bin/activate
```

本机专用环境为 `env`（提示符 `LZHLoco-train`），其中 `legged_gym` 和
`rsl_rl` 均以 editable 方式指向本仓库。WSZLoco 使用其自己仓库下的 `.venv`，
两个环境彼此独立。

可用以下命令确认当前没有串用 WSZLoco：

```bash
python -m pip show legged-gym rsl-rl | grep -E 'Name|Editable project location'
```

仓库已经指向 Nezha URDF：

```text
resources/robots/nezha2_description1126/urdf/nezha2_description.urdf
```

## 训练

```bash
cd /home/bit/LZHLoco/LZHloco
python legged_gym/scripts/train_nezha_mine.py \
  --headless \
  --num_envs 4096 \
  --max_iterations 100000
```

断点续训：

```bash
python legged_gym/scripts/train_nezha_mine.py \
  --resume \
  --load_run <运行目录名> \
  --checkpoint -1 \
  --headless
```

训练输出位于 `logs/nezha_mine/<时间>_gated_modal_dual_encoder/`。

## Isaac Gym Play 与导出

```bash
python legged_gym/scripts/play_nezha_mine.py \
  --load_run <运行目录名> \
  --checkpoint best_policy \
  --command_x 1.0 \
  --command_y 0.0 \
  --command_yaw 0.0
```

这是独立于通用 `play.py` 的 Nezha-MINE 专用脚本。它会加载 checkpoint，使用固定速度指令，输出门控概率及速度/高度估计诊断，记录全部 16 个关节的位置、速度、力矩、动作与目标，并导出完整在线策略（source encoder、modal gate、modal prototypes 和 actor）。完整 play 日志写入 `logs/nezha_mine/play/<时间>/nezha_mine_play.csv`。

```text
logs/nezha_mine/exported/<运行目录名>/policy.pt
```

若使用 `--load_run -1`，导出目录名为 `latest`。导出模型输入形状为 `[batch, 130]`，输出为 `[batch, 16]`；`get_mode_probabilities()` 可用于诊断三个门控概率。

## MuJoCo Sim-to-Sim

安装可选依赖：

```bash
pip install mujoco pyyaml
```

当前已有 Nezha URDF，但没有经过验证的浮动基座 Nezha MJCF。请在 `mujoco/nezha_config.yaml` 设置 `model_path`，或通过 `--model` 传入自己的浮动基座 MJCF。入口会先校验 free joint 和全部 16 个 Nezha 关节，不会回退到其他机器人模型。

```bash
python mujoco/nezha_sim.py \
  --model /absolute/path/to/nezha.xml \
  --policy logs/nezha_mine/exported/latest/policy.pt \
  --vx 0.5
```

无窗口快速验证可加 `--headless --duration 10`。

## 真机部署接入

真机侧只需把机器人 SDK 的状态读取和力矩发送接到 `NezhaPolicyRuntime`：

```python
from deploy.nezha import NezhaPolicyRuntime

runtime = NezhaPolicyRuntime(
    policy_path="logs/nezha_mine/exported/latest/policy.pt",
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

参数模板在 `deploy/nezha/config.yaml`。上真机前必须再核对实际关节顺序、正方向、零位、减速比、力矩限制、急停与通信 watchdog；框架不会假定某一种厂商通信 SDK。

## 测试

不启动 Isaac Gym 的网络结构、loss、门控概率、TorchScript 一致性和历史顺序测试：

```bash
PYTHONPATH=.:rsl_rl pytest -q legged_gym/tests/test_mine_policy.py
```

完整 Isaac Gym 环境仍需要带 GPU、正确 Isaac Gym/PyTorch 组合并成功编译 `gymtorch` 后再进行 smoke test。
