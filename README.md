# Nezha MoE-CTS · Isaac Gym

本分支的唯一机器人训练任务是 `nezha_moe_cts`。教师–学生网络与 MoE 门控采用 `legbot_lab` 的 `WF-CTS-MOE` 设计（参考提交 `daff9fbe3d3eed222b9e39e1dcefa1996f690834`），运行平台为 Isaac Gym。旧 MINE 任务、模型、训练／play／导出入口和部署 runtime 已删除；已有训练日志和模型文件不作删除，新架构需要重新训练。

## 网络与训练

- 教师：295 维特权观测 → MLP [512, 256] → 32 维 L2 归一化 latent。
- 学生：228 维本体历史 → 8 个专家与 Softmax gate → 加权求和 → 32 维 L2 归一化 latent。专家共享特征主干，后接分组线性输出；隐藏层配置为 [512, 256, 256]。
- 共享 Actor：当前本体观测 57 + latent 32 = 89 维，隐藏层 [512, 256, 128]，输出 16 维动作。
- 共享 Critic：当前特权观测 295 + detached latent 32 = 327 维，隐藏层 [512, 256, 128]。
- 默认 75% 环境由教师控制，25% 由学生控制。两组都采样训练共享 Actor/Critic，PPO 更新教师编码器但不更新学生编码器，价值损失不回传编码器。
- 每轮先完成全部 PPO 更新，再用学生控制环境中保存的**同一时刻**本体／特权观测进行 latent MSE 蒸馏，另加系数 0.01 的专家使用均衡损失。两个优化器分别保存和恢复。
- PPO 的教师与学生 surrogate loss 分别取均值后相加；value/entropy 对合并批次取均值。默认 rollout 24 步、5 epochs、4 mini-batches、gamma 0.99、lambda 0.95。
- gate 输出 `expert_0…expert_7` 的权重，专家没有预设的轮式／腿式／混合模式标签。

## 默认观测：完全使用 WF 输入维数

下面顺序即网络输入顺序。scale 在原始值加噪声、裁剪后使用。

| 输入项 | 本体单帧 | 特权单帧 | 本体 scale | 特权 scale |
|---|---:|---:|---|---|
| 机体线速度 | — | 3 | — | 2 |
| 机体角速度 | 3 | 3 | 0.25 | 0.25 |
| 机体系重力投影 | 3 | 3 | 1 | 1 |
| x/y/yaw 速度指令 | 3 | 3 | [2, 2, 0.25] | 1 |
| 相对默认姿态的关节位置 | 16 | 16 | 1 | 1 |
| 关节速度 | 16 | 16 | 0.05 | 0.05 |
| 上一次动作（控制缩放前） | 16 | 16 | 1 | 1 |
| 关节加速度 | — | 16 | — | 0.0001 |
| 施加的关节力矩 | — | 16 | — | 0.01 |
| 轮接触力历史特征 | — | 16 | — | 0.001 |
| 地形高度 | — | 187 | — | 2.5 |
| **合计** | **57** | **295** | | |

学生使用 4 帧历史，共 **228** 维。采用 WF 的 **term-major、oldest-first** 顺序：先放某一观测项的 4 帧，再放下一项；不是把 4 个完整的 57 维帧直接串起来。Actor 当前帧从相同历史中提取，保证噪声一致；环境 reset 后用新 episode 的第一帧填满历史。

轮关节索引为 [3, 7, 11, 15]。这 4 个关节的位置观测恒为 0，轮速保留真实值。默认本体原始均匀噪声幅度：角速度 ±0.2、重力 ±0.05、关节位置 ±0.03、关节速度 ±2；轮位置仍为 0。特权输入不加噪声。

接触特征为四轮接触力范数的 [最近三次最大值、当前、前一次、前两次]，共 16 维，按每个 0.005 s 物理步采样。高度使用 17×11 个 yaw 对齐点（x -0.8…0.8 m，y -0.5…0.5 m），按 WF 的 xy 网格顺序展开；值为机体高度减地面高度再减 0.5 m。高度原始值裁剪到 ±1，其余项默认 ±100，之后乘 scale。

Isaac Gym 的高度来自原有 heightfield 邻点采样，接触来自 PhysX net contact tensor；它们与 Isaac Lab 的射线／接触传感器并非完全相同的数值实现。这里对齐输入定义、维数、顺序和预处理，保留 Nezha 的 URDF、地形课程、奖励与动力学随机化。腿部动作尺度为 0.25，轮速尺度为 **10**，控制频率 50 Hz。

## 训练、续训、play 和导出

在仓库根目录、已安装 Isaac Gym 的训练环境执行。本机可使用 `.venv-train/bin/python`，并设置：

```bash
export PYTHONPATH="$PWD/rsl_rl:$PWD"
export PATH="$PWD/.venv-train/bin:$PATH"
```

训练与续训：

```bash
python legged_gym/scripts/train_nezha_moe_cts.py --headless --no_wandb

python legged_gym/scripts/train_nezha_moe_cts.py --headless --no_wandb \
  --resume --load_run <运行目录名> --checkpoint -1
```

默认日志目录为 `logs/nezha_moe_cts/<时间>_<run_name>/`。默认配置启用 W&B；`--no_wandb` 使用本地 TensorBoard。日志包含教师／学生回报、PPO 损失、蒸馏损失、均衡损失、gate entropy 和各专家使用率。训练需至少两个环境；当前 runner 面向单设备训练。

Checkpoint 保存网络与观测 schema、环境配置、控制参数、两个优化器和训练计数。续训先恢复观测配置再创建环境，可以用 `--num_envs` 调整并行数量；物理状态和历史重新初始化，不是逐物理步确定性恢复。旧架构 checkpoint 会明确报错。

学生 play（默认一个环境）与独立导出：

```bash
python legged_gym/scripts/play_nezha_moe_cts.py \
  --load_run <运行目录名> --checkpoint -1 --play_num_envs 1

python legged_gym/scripts/export_nezha_moe_cts.py \
  --run <运行目录名>

# 也可指定 checkpoint 和输出路径；此导出入口不需要 Isaac Gym/GPU
python legged_gym/scripts/export_nezha_moe_cts.py \
  --checkpoint /absolute/path/model_200.pt --output /absolute/path/policy.pt
```

导出的 `policy.pt` 只包含学生 MoE、共享 Actor 及必要元数据，输入 [N, 228]、输出 [N, 16]。历史由 runtime 管理；教师、Critic 和优化器不进入部署模型。导出会检查 TorchScript 与 checkpoint 学生输出一致，同时写出便于查看的 `policy.json`。

MuJoCo：

```bash
.venv-sim/bin/python deploy/deploy_mujoco/deploy_mujoco.py \
  -c nezha_moe_cts.yaml --policy /absolute/path/exported/policy.pt \
  --headless --duration 3 --vx 0.5
```

传感器观测、裁剪、缩放和历史顺序由模型中的 schema 驱动。部署 runtime 校验模型附带的名义姿态、PD 增益、关节顺序、控制周期和动作尺度等控制参数。详细说明见 [deploy/readme.md](deploy/readme.md)。

## 后续修改入口

| 内容 | 文件 |
|---|---|
| 观测项、scale、noise、clip、历史排列的统一定义 | `rsl_rl/rsl_rl/utils/cts_observations.py` |
| Isaac Gym 原始观测采集、接触历史、reset | `legged_gym/envs/nezha/nezha_moe_cts_env.py` |
| 历史帧数、专家数、latent、网络层宽、训练参数 | `legged_gym/envs/nezha/nezha_moe_cts_config.py` |
| Nezha 控制、奖励、随机化与资产配置 | `legged_gym/envs/nezha/nezha_robot_config.py` |
| MoE 与教师／学生 Actor-Critic | `rsl_rl/rsl_rl/modules/moe.py`、`actor_critic_moe_cts.py` |
| 两阶段训练与 rollout | `rsl_rl/rsl_rl/algorithms/moe_cts.py`、`storage/rollout_storage_cts.py` |

调整历史长度用 `NezhaMoECTSCfg.env.frame_stack`；调整观测项可修改 `default_observation_config()`，或在 `cfg.env.observation_config` 提供完整 schema。维数从 schema 自动推导。新增传感量时，需同时实现环境和部署的原始量采集；不要只修改维数常量。CTS 实际预处理不读取旧基类的 `normalization.obs_scales`。

## 验证

```bash
OMP_NUM_THREADS=1 PYTHONPATH="$PWD/rsl_rl:$PWD" \
  .venv-train/bin/python -m pytest legged_gym/tests tests -q
```

自动化验证覆盖输入维数、逐项预处理、轮速、历史／reset、梯度隔离、同刻蒸馏、尾批次完整采样、checkpoint 续训、TorchScript 一致性和部署控制参数校验。短时仿真仅验证接口与数值运行，策略运动表现需要正式训练与评估。

实际 Isaac Gym 短时物理回归（需训练环境，默认 16 个环境和小地形）：

```bash
TORCH_EXTENSIONS_DIR=/tmp/cts_torch_extensions python legged_gym/tests/test_env.py --headless --rl_device cpu --sim_device cpu --pipeline cpu --smoke_terrain trimesh
```

通用入口 `legged_gym/scripts/train.py` 和 `play.py` 也分别转入上述 CTS 训练与学生评估流程。
