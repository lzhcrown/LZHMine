## 使用 uv 安装两个 Python 环境

在 Linux x86_64 上将训练和 MuJoCo 仿真分别安装，避免 Isaac Gym 的 Python 3.8 依赖进入 Python 3.11 环境：

| 用途 | 虚拟环境 | Python | 主要依赖 |
|---|---|---|---|
| 训练、play、导出 | `.venv` | 3.8（本机 3.8.20） | Isaac Gym Preview 4、PyTorch 2.4.1 / CUDA 12.1、NumPy 1.23.5、TensorBoard 2.14.0、protobuf 4.25.8 |
| MuJoCo 仿真 | `.venv-mujoco` | 3.11（本机 3.11.16） | MuJoCo 3.2.7、PyTorch 2.4.1 CPU、NumPy 1.26.4 |

依赖版本分别固定在 [`requirements/train.txt`](requirements/train.txt) 和 [`requirements/mujoco.txt`](requirements/mujoco.txt)。两端 PyTorch 均为 2.4.1，便于直接加载训练端导出的 TorchScript 策略；MuJoCo 环境默认在 CPU 上推理。PyTorch 安装源见[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)，MuJoCo 使用[官方 Python 包](https://mujoco.readthedocs.io/en/3.2.7/python.html)。

本机已安装 uv，并已解压 Isaac Gym 到 `/home/crown/isaacgym`。重建或补齐环境可执行：

```bash
cd /home/crown/LZHMine
bash scripts/setup_envs.sh

# 也可只安装其中一个环境
bash scripts/setup_envs.sh train
bash scripts/setup_envs.sh mujoco
```

脚本默认从 `$HOME/isaacgym/python` 安装 SDK；其他位置通过 `ISAAC_GYM_PATH=/实际路径/isaacgym/python bash scripts/setup_envs.sh train` 指定。SDK 需事先自行解压，脚本不会下载它。训练还需要可用的 NVIDIA 驱动和 C++ 编译器（`g++`）；首次启动会自动编译 `gymtorch`。uv 在找不到所需 Python 时会自动下载解释器。已有环境会复用并检查 Python 版本，不会自动删除。

直接使用下文的 `.venv/bin/python` 或 `.venv-mujoco/bin/python` 即可，也可以按需激活：

```bash
source .venv/bin/activate          # Python 3.8：训练、play、导出
deactivate
source .venv-mujoco/bin/activate   # Python 3.11：MuJoCo 仿真
deactivate
```

`.python-version` 默认指定 3.8；仿真请明确使用 `.venv-mujoco/bin/python`。安装脚本结尾会执行 `uv pip check`。需要检查代码与仿真依赖时可运行现有测试；下面显式设置测试进程的 `PYTHONPATH`，避免已激活的 ROS 环境自动加载其他 Python 版本的 pytest 插件：

```bash
PYTHONPATH="$PWD:$PWD/rsl_rl" OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest legged_gym/tests tests/test_cts_deployment.py -q
PYTHONPATH="$PWD:$PWD/rsl_rl" OMP_NUM_THREADS=1 \
  .venv-mujoco/bin/python -m pytest tests/test_cts_deployment.py deploy/deploy_mujoco/tests -q
```

快速验证真实 GPU 训练可将下文命令改为 `--max_iterations 1 --num_envs 16 --run_name uv_env_smoke --no_wandb`。这只用于检查运行链路，生成的单轮策略不代表训练效果；正式训练需另外启动并设置训练预算。

## 训练

### G12 高台与常规地形混合训练

Nezha 的 WF 训练地图有 20 行难度、20 列地形。原随机障碍占用的第 14–17 列依次改为两列凹台、两列凸台，合计占全图的 20%。每个 9 m × 9 m 的格子中央有一个 3.2 m × 3.2 m 方台，四周留出平地。凹台的出生点在坑底中央，凸台的出生点在台前低处，初始朝向正对台面。

训练从原有的第 0–9 级延伸到第 19 级。第 `r` 级的难度按 `d = (r + u) / 10` 采样，`u` 在 [0, 1) 内均匀取值。上、下楼梯的单阶踏面均为 **35 cm**，沿行进方向占 7 个 5 cm 网格，各等级的单阶高度沿用 `0.05 + 0.207d` m。凹台深度和凸台高度在 G0–G11 中均为 `0.05 + 0.225d` m，在 G12 中均为 `0.20 + 0.50d` m。实际高度按 5 mm 栅格量化。

G12 沿用 G7 的学生和 Actor。凹台和凸台共同使用高台奖励与 0.5–1.0 m/s 的前进命令。平地、楼梯及其余地形使用 G7 的奖励与命令采样。

高台奖励配置在 [platform_rewards.py](legged_gym/terrains/platform_rewards.py)。平面线速度跟踪权重为 3，核分母为 0.25。高台惩罚机身系垂向速度平方和滚转、偏航角速度平方，允许机身俯仰。关节姿态、力矩平方、关节速度和加速度、碰撞、动作差分、软位置限位及失败终止使用 Dreamwaq M20 的权重。高台按 M20 先将普通项加总并截负为零，再加终止项。训练时按所分配的地形选择奖励和 Critic 价值输出，导出学生策略使用与 G7 相同的本体观测。

从零训练：

```bash
.venv/bin/python experiments/g12_high_platform/train.py --from_scratch --num_envs 4096 --max_iterations 20000 --seed 10
```

从已有 G7 训练检查点微调，把路径换成实际的模型文件。两个价值输出均从 G7 的价值估计初始化。

```bash
.venv/bin/python experiments/g12_high_platform/train.py --init_from /path/to/g7/model_20000.pt --num_envs 4096 --max_iterations 5000 --seed 10
```

G12 的日志与检查点写入 `logs/nezha_moe_cts/g12_high_platform/`。从旧 G7 检查点用 `--init_from` 会启用新版 20 级地图。用 `--resume` 续训会恢复检查点保存的地图，包括旧版 10 级课程。TensorBoard 的 `Train/ordinary_reward` 和 `Train/high_platform_reward` 分别记录各自最近 100 个完整回合的平均累计奖励。机器人接近所属训练地形边界时按截断处理，以免驶入相邻地形后继续沿用前一地形的奖励。

### G0–G11 对照实验

基于 g7 的动作残差与时间平滑对照已增加为 g8–g11。四组配置、共同检查点初始化、训练/评估命令及结果判定见 [残差实验说明](experiments/RESIDUAL_EXPERIMENTS.md)。首轮使用 `--init_from` 从同一个 g7 checkpoint 公平微调。

```bash
cd /home/crown/LZHMine

# G0
.venv/bin/python experiments/g0_mlp/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0

# G1
.venv/bin/python experiments/g1_student_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0

# G2
.venv/bin/python experiments/g2_actor_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0

# G3
.venv/bin/python experiments/g3_student_actor_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0

# G4
.venv/bin/python experiments/g4_critic_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0

# G5
.venv/bin/python experiments/g5_student_critic_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0

# G6
.venv/bin/python experiments/g6_actor_critic_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0

# G7
.venv/bin/python experiments/g7_all_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10 --rl_device cuda:0
```

## Play

```bash
cd /home/crown/LZHMine

# 选择对应实验：加载本组最新 checkpoint，播放并自动导出本组 policy.pt。

# G0
.venv/bin/python experiments/g0_mlp/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0

# G1
.venv/bin/python experiments/g1_student_moe/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0

# G2
.venv/bin/python experiments/g2_actor_moe/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0

# G3
.venv/bin/python experiments/g3_student_actor_moe/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0

# G4
.venv/bin/python experiments/g4_critic_moe/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0

# G5
.venv/bin/python experiments/g5_student_critic_moe/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0

# G6
.venv/bin/python experiments/g6_actor_critic_moe/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0

# G7
.venv/bin/python experiments/g7_all_moe/play.py \
  --load_run -1 --checkpoint -1 \
  --play_num_envs 1 --play_steps 5000 \
  --command_x 0.5 --command_y 0.0 --command_yaw 0.0 --rl_device cuda:0
```

## 查看 TensorBoard

```bash
cd /home/crown/LZHMine

# 在另一个终端选择一条命令启动，浏览器访问对应地址。

# 全部实验：http://127.0.0.1:6007/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts \
  --host 127.0.0.1 --port 6008

# G0：http://127.0.0.1:6100/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g0_mlp \
  --host 127.0.0.1 --port 6100

# G1：http://127.0.0.1:6101/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g1_student_moe \
  --host 127.0.0.1 --port 6101

# G2：http://127.0.0.1:6102/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g2_actor_moe \
  --host 127.0.0.1 --port 6102

# G3：http://127.0.0.1:6103/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g3_student_actor_moe \
  --host 127.0.0.1 --port 6103

# G4：http://127.0.0.1:6104/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g4_critic_moe \
  --host 127.0.0.1 --port 6104

# G5：http://127.0.0.1:6105/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g5_student_critic_moe \
  --host 127.0.0.1 --port 6105

# G6：http://127.0.0.1:6106/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g6_actor_critic_moe \
  --host 127.0.0.1 --port 6106

# G7：http://127.0.0.1:6107/#scalars
.venv/bin/tensorboard \
  --logdir /home/crown/LZHMine/logs/nezha_moe_cts/g7_all_moe \
  --host 127.0.0.1 --port 6107
```

## Isaac Gym 观看与指标原始数据采集

八组学生策略支持十种固定测试地形：凹形/凸形楼梯各设单阶 35、25、30 cm，高台顶面 5×5 m、高 45 cm，以及平地、粗糙地面、20° 斜坡（水平长 6.5 m、宽 5.6 m）。使用 `wheel_gym_CQ` 的四轮穿越判据。模型默认读取 `logs/<组名>/model_20000.pt`。

```bash
# 只看八组通过 30 cm 凸形楼梯，不写采集文件
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group all --checkpoint 20000 --viewer --no-record \
  --terrain-types stairs_convex_30cm --episodes-per-env 3

# 正式采集：每组每地形 500 次，20 个并行槽各 25 轮，共 40000 次
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group all --checkpoint 20000 --envs-per-terrain 20 --episodes-per-terrain 500 \
  --metric-successes-per-terrain 10 \
  --spawn-xy-jitter 0.02 --speed 0.8 --seed 10 --max-seconds 30 \
  --output-dir evaluations/cts_20000_terrain_v12_500_success10_seed10
```

观看时相机跟随机器人，N/B 切换目标，空格暂停，Esc 结束。正式采集的每组 5000 次尝试全部写入结果及索引；仅每种地形前 10 次成功保留详细指标，分别写入 `power.csv`、`cot.csv`、`tracking.csv`、`stability.csv`。`success_rate.csv` 统计全部尝试的成功率。各表以唯一 `trial_id` 关联，失败和超额成功不保存详细轨迹。地形名、八组名称、观看和采集完整命令见 [使用说明](legged_gym/evaluation/README.md)。

## MuJoCo 仿真

```bash
cd /home/crown/LZHMine

# 先执行对应实验的 play，生成 policy.pt；仿真自动读取本组策略。

# G0
.venv-mujoco/bin/python experiments/g0_mlp/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0

# G1
.venv-mujoco/bin/python experiments/g1_student_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0

# G2
.venv-mujoco/bin/python experiments/g2_actor_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0

# G3
.venv-mujoco/bin/python experiments/g3_student_actor_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0

# G4
.venv-mujoco/bin/python experiments/g4_critic_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0

# G5
.venv-mujoco/bin/python experiments/g5_student_critic_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0

# G6
.venv-mujoco/bin/python experiments/g6_actor_critic_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0

# G7
.venv-mujoco/bin/python experiments/g7_all_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0
```
