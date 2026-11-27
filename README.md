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

### G12 高台实验 v1 与 v2

`g12_v1` 对应提交 `95a92ef`，普通地形使用 G7 奖励，凸台和凹台使用独立的高台奖励。新版 `g12_v2` 基于 `g12_v4`，将 v4 的 16 项高台奖励用于平地、楼梯、坡地、凸台和凹台，全地形目标机身净高为 0.55 m。地形仍决定指令采样范围和 Critic 价值输出。详见 [v2 配置](experiments/g12_v2/config.py) 和 [v4 配置](experiments/g12_v4/config.py)。

| 设置 | `g12_v4`（对照） | 新版 `g12_v2` |
|---|---|---|
| 指令采样 | 高台前进 0.5–1.0 m/s，横向和偏航角速度指令为 0，约每 5 秒重采样；常规地形使用各自的 WF 指令 | 同 v4 |
| 目标机身净高 | 全地形 0.54 m | 全地形 0.55 m，净高误差平方权重固定为 −10 |
| 髋外展偏差 / 静止站姿 / 机身倾斜 | 常规地形 −8 / −3 / −2，高台 −0.5 / −0.5 / −0.5 | 全地形 −6 / −2.25 / −1.5，分别取 v4 常规地形权重的 75% |
| 常规地形奖励 | G7 常规奖励，含轮地接触项与净高、垂向速度权重调度 | 与高台共用固定权重，包括运动站姿的绝对偏差和 −0.10、垂向速度平方 −2、滚转和偏航角速度平方和 −0.15 |
| 回报计算 | 高台各项直接求和，小计最低为 0，最后加入失败终止项；常规地形各项分别限幅 | 所有地形采用 v4 的高台求和方式，失败终止项权重 −0.8；躯干碰撞在所有地形均计入碰撞项 |

两个版本均使用 20 级地图和 Dreamwaq M20 的升降级规则。重置前，取机身末位置与出生参考点的水平距离 `D`。9 m 地块上，`D > 4.5 m` 升一级。其余回合若 `D < 0.5 × 回合时长上限 × 末步平面指令速率` 则降一级，否则保持原级。凸台的出生参考点在台前低处。课程等级衡量水平移动能力，固定高度评估可单独统计登顶后在另一侧平稳落地的成功率。原 `g12_high_platform` 使用同一套地形高度和深度公式，按回合内的最远水平位移超过 4.0 m 晋级。

比较 v1 和 v2 时，两条命令可使用同一个 G7 检查点、种子和训练轮数。`--resume` 会恢复检查点保存的奖励配置。从已有权重开始训练并应用新版 v2 的奖励配置时使用 `--init_from`。

```bash
.venv/bin/python experiments/g12_v1/train.py \
  --init_from /path/to/g7/model_20000.pt --num_envs 4096 --max_iterations 3000 --seed 10
.venv/bin/python experiments/g12_v2/train.py \
  --init_from /path/to/g7/model_20000.pt --num_envs 4096 --max_iterations 3000 --seed 10
```

从头训练可将 `--init_from` 及其路径换成 `--from_scratch`。从已有 v1 策略开始 v2 实验时，用 v2 的 `train.py --init_from /path/to/v1/model_XXXX.pt`，它会载入已有权重、重新初始化优化器并应用 v2 配置。`--resume` 按检查点中保存的实验标识和环境配置续训。若此前已用过渡配置以 `g12_v1` 名义训练，其检查点仍带 `g12_v1` 标识，可通过 v1 入口恢复当次训练，也可通过 v2 的 `--init_from` 开始新版实验。

每版的训练、Isaac Gym 回放、导出和 MuJoCo 仿真入口分别为对应实验目录中的 `train.py`、`play.py`、`export.py` 和 `simulate.py`。日志和检查点写入 `logs/nezha_moe_cts/<版本>/`，默认导出位置为 `deploy/deploy_mujoco/nezha/policy/<版本>/policy.pt`，其中 `<版本>` 为 `g12_v1` 或 `g12_v2`。以下示例读取上传到 `logs/<版本>/model_20000.pt` 的检查点，分别回放两个版本的 50 cm 凸台并保存轨迹。记录时，`terminal=1` 的行用 `termination_reason` 标明平稳落地（`stable_landing`）、跌倒（`fall`）或超时（`time_limit`）。只观看时可省略 `--platform_trace` 及其路径。

```bash
for version in g12_v1 g12_v2; do
  .venv/bin/python "experiments/${version}/play.py" \
    --checkpoint_path "logs/${version}/model_20000.pt" --platform_shape block \
    --platform_height 0.50 --platform_width 5.0 --command_x 0.8 --command_y 0 --command_yaw 0 \
    --play_num_envs 1 --play_steps 3000 --seed 10 --no_export \
    --platform_trace "logs/${version}_block50_seed10.csv"
done
```

### G12 v3 高台运动站姿对照

`g12_v3` 从 `g12_v1` 派生，把高台运动站姿项 `run_still` 的权重从 −0.05 调到 G7 平地使用的 −0.10。在凸台和凹台的整个回合，只要平面速度指令的模大于 0.1 m/s，就累计四条腿共 12 个实际关节角相对默认角的绝对偏差。每条腿的髋、大腿和小腿参考角依次为 0、0.8、−1.6 rad，四个驱动轮继续使用速度控制。[v3 配置](experiments/g12_v3/config.py)

从已经验证可登台的 **v1 训练检查点** 开始微调。示例中的 1000 为追加训练轮数。

```bash
.venv/bin/python experiments/g12_v3/train.py \
  --init_from /path/to/g12_v1/model_XXXX.pt --num_envs 4096 --max_iterations 1000 --seed 10
```

使用各版本的 `play.py` 和相同的 `--platform_height 1.00 --platform_width 5.0 --command_x 0.8 --command_y 0 --command_yaw 0` 对照原 v1 和微调后的 v3。指定 `--platform_shape block --platform_trace <独立CSV路径>` 后，`leg_pose_mae_rad` 记录每一步 12 个腿关节相对默认角的平均绝对偏差。按相同种子分别统计越过高台并在另一侧平稳落地的成功率，在这些成功回合中比较四轮都在台面、仍有前进指令时的关节误差和实际前进速度。

### G12 v4 机身净高对照

`g12_v4` 继承 v3，将普通地形（含平地和楼梯）的目标净高从 0.50 m 调为 0.54 m，凸台和凹台从 0.52 m 调为 0.54 m。净高取机身根坐标相对局部采样地面平均高度的垂直距离。[v4 配置](experiments/g12_v4/config.py)

从 v3 检查点开始微调的示例

```bash
.venv/bin/python experiments/g12_v4/train.py \
  --init_from /path/to/g12_v3/model_XXXX.pt --num_envs 4096 --max_iterations 1000 --seed 10
```

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

Isaac Gym 每轮从出生开始计时，上限为 **30 仿真秒**。楼梯或高台要求机器人到达中央阶面或台面，然后在另一侧平地平稳落地。凹形地形先下至底部，再从对岸回到平地。通过、跌倒或超时会自动开始下一轮。

有窗口的 `play.py` 默认持续运行，按 Esc、关闭窗口或 Ctrl+C 退出。无窗口的 `--headless` 默认执行 5000 个控制步。可用 `--play_steps N` 指定**整次回放**的步数，耗尽时结束进程，即使当轮仍在进行。例如控制周期为 0.02 s 时，`--play_steps 3000` 共运行 60 仿真秒。

在任一实验的 `play.py` 或固定地形评估命令中加入 `--joint-angles-csv logs/joint_angles/walk.csv`，即可在同一 CSV 逐步记录 Actor 的 16 维原始动作、12 个腿关节的目标角与实际角、4 个轮轴转角、指令与实际速度、四轮竖直接触力，以及 Student 和 Actor 的专家权重。每行带有机器人编号、回合编号和仿真时间。完整示例见[同步记录说明](legged_gym/evaluation/README.md#记录四条腿的实际关节角)。

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

八组学生策略支持十种默认固定测试地形，包括单阶高 35、25、30 cm 的凹形和凸形楼梯，5×5 m、高 45 cm 的高台，以及平地、粗糙地面和 20° 斜坡。单组使用 `--viewer --no-record` 时，各机器人每轮最长运行 30 仿真秒并自动重试。楼梯和高台以机器人走过中央阶面或台面并在另一侧平地持续稳定支撑为成功。模型默认读取 `logs/<组名>/model_20000.pt`。

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
  --output-dir evaluations/cts_20000_protocol14_500_success10_seed10
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
