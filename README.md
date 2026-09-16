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

需要 NVIDIA Isaac Gym Preview 4、可用的 CUDA/PyTorch 环境以及 `ninja`（Isaac Gym 的 `gymtorch` 扩展首次导入时需要）：

```bash
cd /home/bit/LZHMine
source .venv/bin/activate
```

本机专用环境为 `.venv`，其中 `legged_gym` 和
`rsl_rl` 均以 editable 方式指向本仓库。WSZLoco 使用其自己仓库下的 `.venv`，
两个环境彼此独立。

### Weights & Biases

训练默认同时记录 TensorBoard 和 W&B。首次在训练机配置时执行：

```bash
source .venv/bin/activate
python -m pip install wandb tensorboard
wandb login
```

`wandb login` 会将 API key 保存在训练机的用户配置中，不要把 key 写入代码、
配置文件或 Git。默认 W&B 项目为 `LZHMine`，entity 使用当前登录账号，run
名称与本地 `logs/nezha3_mine/<时间>_<run_name>/` 目录名一致。

开始训练后，W&B 会记录：

- PPO、估计器、swap、mode 和 semantic mode loss；
- 平均 episode reward、episode 长度及各奖励分量；
- wheel/leg/hybrid 门控概率和语义活动指标；
- 学习率、动作噪声、FPS、采样时间和学习时间。

临时关闭或覆盖项目设置：

```bash
python legged_gym/scripts/train_nezha_mine.py --headless --no_wandb

python legged_gym/scripts/train_nezha_mine.py --headless \
  --wandb_project LZHMine \
  --wandb_entity <你的W&B用户名或团队名>
```

网络不稳定时可先离线记录，之后再同步：

```bash
python legged_gym/scripts/train_nezha_mine.py --headless --wandb_mode offline
wandb sync logs/nezha3_mine/<运行目录>/wandb/offline-run-*
```

对应默认配置位于
`legged_gym/envs/nezha/nezha_mine_config.py` 的 `runner` 部分。W&B 仅记录
曲线和配置，不会自动上传 checkpoint 或部署策略。

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

`logs/` 保留训练运行、checkpoint/model 和原始导出记录。确认某个导出策略
用于部署后，将它显式复制为部署快照：

```bash
cp logs/nezha3_mine/exported/<运行目录名>/policy.pt \
  deploy/deploy_mujoco/nezha/policy/policy.pt
```

MuJoCo 默认只读取该部署快照，因此训练产物和当前部署版本不会混在一起。

若使用 `--load_run -1`，导出目录名为 `latest`。导出模型输入形状为 `[batch, 130]`，输出为 `[batch, 16]`；`get_mode_probabilities()` 可用于诊断三个门控概率。

## MuJoCo Sim-to-Sim

在 macOS 项目根目录中用 `uv` 创建独立的仿真环境（无需也无法在 Mac
上安装 Isaac Gym）：

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch numpy pyyaml mujoco
```

`deploy/deploy_mujoco/configs/nezha_mine.yaml` 默认使用
`deploy/deploy_mujoco/nezha/mjcf/scene_course.xml`。
机器人固定出生在 `x=-5, y=0` 的公共平地区，前方是3条横向一字
排开的评测路线：

- `y=-5 m`：5级20 cm楼梯，踏面70 cm，最高1.0 m；
- `y=0 m`：高0.40 m、长6.0 m的矩形高台；
- `y=+5 m`：170个随机凸起组成的确定性碎石路，最高12 cm。

先用 `A/D` 在平地区横移对准路线，再用 `W` 前进即可。地形使用
深蓝—浅蓝高对比棋盘格材质。MuJoCo 模型关闭机器人自碰撞、加入训练
基准的关节摩擦，并在每个5 ms物理步重新计算 PD 力矩。这三种地形是
评测用的压力测试场景，其中20 cm楼梯和40 cm高台超出当前训练地形分布。

该评测赛道只依赖 NumPy，Mac 和 Linux 都可直接重新生成：

```bash
source .venv/bin/activate
python deploy/deploy_mujoco/tools/generate_terrain_course.py
```

横向路线中心坐标为：`y=-5` 楼梯、`y=0` 高台、`y=+5` 碎石路。

先执行无窗口 smoke test：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py \
  --headless --duration 3 --vx 0.5
```

macOS 的交互式 viewer 必须通过 MuJoCo 安装的 `mjpython` 启动：

```bash
.venv/bin/mjpython deploy/deploy_mujoco/deploy_mujoco.py \
  --duration 60
```

配置和命令行中的相对资产路径都以 `deploy/deploy_mujoco/` 为基准。
`--policy` 也可以直接传入某个导出目录，脚本会在其中查找
`policy.pt`。如果 `latest` 不存在，脚本会自动选择 `exported/`
下最新的策略。`--duration` 表示仿真时间，而不是 headless 运行的墙钟时间。

启动后机器人先隐藏预沉降，然后在零指令下冻结待机；按 `W/S` 调前后速度、`A/D` 调侧向速度、`Q/E` 调转向、空格停止、`R` 复位。要诊断策略本身在零指令下为何输出轮速，可加 `--run-zero-policy`。无窗口快速验证可加 `--headless --duration 10`。

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

## 测试

不启动 Isaac Gym 的网络结构、loss、门控概率、TorchScript 一致性和历史顺序测试：

```bash
PYTHONPATH=.:rsl_rl pytest -q legged_gym/tests/test_mine_policy.py
```

完整 Isaac Gym 环境仍需要带 GPU、正确 Isaac Gym/PyTorch 组合并成功编译 `gymtorch` 后再进行 smoke test。
