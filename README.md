# Nezha Gated-Modal Locomotion

本仓库是面向 Nezha 轮腿机器人的训练、Isaac Gym play、策略导出、MuJoCo sim-to-sim 与真机接入框架。默认任务为 `nezha_mine`，不注册也不依赖其他机器人任务。

## 方法与代码结构

门控模态双编码器以 `wheel_gym_CQ/logs/nezha3_mine/Apr10_11-39-09_trimesh_baseline_门控——三类` 保存的实际训练源码为基准，并按 `LZHMine` 分层：

- `rsl_rl/rsl_rl/modules/mine_estimator.py`：历史观测 source encoder、训练期 target encoder、三模态 soft gate、模态原型和 prototype swap loss。
- `rsl_rl/rsl_rl/modules/mine_actor_critic.py`：估计值和门控 latent 与最新一帧本体观测融合后送入 actor；包含与在线推理完全一致的 TorchScript exporter。
- `rsl_rl/rsl_rl/algorithms/mine_ppo.py`：保留原实现的 PPO 参数组和 estimator 独立优化器；策略路径会 detach estimator，因此不会在一次 mini-batch 中被 PPO 重复更新。
- `rsl_rl/rsl_rl/runners/mine_on_policy_runner.py`：rollout、终止状态特权观测、日志、断点与 best policy。
- `legged_gym/envs/nezha/`：Nezha 16 维动作、65 维单帧观测、混合腿关节位置/轮关节速度控制与训练配置。
- `deploy/nezha/`：不依赖仿真器的观测历史和混合控制运行时，供真机 SDK 接入。
- `mujoco/nezha_sim.py`：复用同一部署运行时的 sim-to-sim 入口。

与原实验一致，所有新训练采用 `oldest-first` 历史顺序。Actor 历史为 `2 x 65 = 130`，critic 历史为 `3 x 375 = 1125`。原实现有一个必须保留的特殊约定：Actor 使用最后的最新帧，门控网络使用最前的最旧帧。`is_privileged_obs=False`，所以 target encoder 使用最新375维特权帧的前65维；四个监督目标仍是该帧最后的机体线速度和基座高度。

三种模态的固定编号为 `0=wheel`、`1=leg`、`2=hybrid`。Apr10 基准使用自标注门控损失，额外的语义锚定默认关闭。该基准还会在 `_reward_dof_vel` 中把四个轮速写为零，因此部署与 MuJoCo 的策略观测也会屏蔽这四个轮速；真实轮速仍用于轮电机 PD 力矩。

## 环境准备

需要 NVIDIA Isaac Gym Preview 4、可用的 CUDA/PyTorch 环境以及 `ninja`（Isaac Gym 的 `gymtorch` 扩展首次导入时需要）：

```bash
cd /home/bit/LZHMine
source .venv/bin/activate
```

本机专用环境为 `.venv`，其中 `legged_gym` 和
`rsl_rl` 均以 editable 方式指向本仓库。WSZLoco 使用其自己仓库下的 `.venv`，
两个环境彼此独立。

可用以下命令确认当前没有串用 WSZLoco：

```bash
python -m pip show legged-gym rsl-rl | grep -E 'Name|Editable project location'
```

仓库已经指向 Nezha URDF：

```text
resources/robots/urdf2.1/urdf/nezha3_description.urdf
```

## 训练

```bash
cd /home/bit/LZHMine
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

训练输出位于 `logs/nezha3_mine/<时间>_trimesh_baseline_test/`。

本次对齐修正了历史帧顺序并恢复 `feet_air_time=1.0`。此前按 `newest-first` 训练的 LZHMine checkpoint 不能作为“完全一致”的实验续训，应从新实验开始训练。

## Isaac Gym Play 与导出

只导出策略、不创建仿真和图片（推荐用于 sim-to-sim）：

```bash
python legged_gym/scripts/export_nezha_mine.py \
  --run <运行目录名> \
  --history-order oldest_first
```

该命令会自动选择运行目录中编号最大的 checkpoint。对本次修正前训练的旧 LZHMine checkpoint 使用 `--history-order newest_first`。

```bash
python legged_gym/scripts/play_nezha_mine.py \
  --load_run <运行目录名> \
  --checkpoint best_policy \
  --checkpoint_history_order oldest_first \
  --command_x 1.0 \
  --command_y 0.0 \
  --command_yaw 0.0
```

这是独立于通用 `play.py` 的 Nezha-MINE 专用脚本。它会加载 checkpoint，使用固定速度指令，输出门控概率及速度/高度估计诊断，记录全部 16 个关节的位置、速度、力矩、动作与目标，并导出完整在线策略（source encoder、modal gate、modal prototypes 和 actor）。完整 play 日志写入 `logs/nezha3_mine/play/<时间>/nezha_mine_play.csv`。旧的 Sep01 LZHMine checkpoint 必须改用 `--checkpoint_history_order newest_first`。

```text
logs/nezha3_mine/exported/<运行目录名>/policy.pt
```

若使用 `--load_run -1`，导出目录名为 `latest`。导出模型输入形状为 `[batch, 130]`，输出为 `[batch, 16]`；`get_mode_probabilities()` 可用于诊断三个门控概率。

## MuJoCo Sim-to-Sim

安装可选依赖：

```bash
pip install mujoco pyyaml
```

`mujoco/nezha_config.yaml` 默认使用 `mujoco/models/nezha_course_scene.xml`。机器人固定出生在 `x=-5, y=0` 的公共平地区；其前方是横向一字排开的7条路线，依次覆盖下楼梯、粗糙下坡、下坡、平地、上坡、粗糙上坡和上楼梯。先用 `A/D` 在平地区横移对准路线，再用 `W` 前进即可，无需重启或用命令行选择地形。各路线直接截取训练 curriculum 第5难度行中的实际地形数据。MuJoCo 模型关闭机器人自碰撞、加入训练基准的关节摩擦，并在每个5 ms物理步重新计算 PD 力矩。

训练地形配置变化后，重新生成综合测试赛道：

```bash
source .venv/bin/activate
export TORCH_EXTENSIONS_DIR=/home/bit/LZHMine/.cache/torch_extensions
python mujoco/generate_terrain_course.py
```

横向路线中心坐标为：`y=-15` 下楼梯、`-10` 粗糙下坡、`-5` 下坡、`0` 平地、`+5` 上坡、`+10` 粗糙上坡、`+15` 上楼梯。

```bash
/home/bit/WSZLoco/WSZloco/.venv-sim/bin/python mujoco/nezha_sim.py \
  --policy logs/nezha3_mine/exported/latest/policy.pt
```

启动后机器人先隐藏预沉降，然后在零指令下冻结待机；按 `W/S` 调前后速度、`A/D` 调侧向速度、`Q/E` 调转向、空格停止、`R` 复位。要诊断策略本身在零指令下为何输出轮速，可加 `--run-zero-policy`。无窗口快速验证可加 `--headless --duration 10`。

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
