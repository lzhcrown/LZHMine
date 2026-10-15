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
.venv-mujoco/bin/mjpython deploy/deploy_mujoco/deploy_mujoco.py \
  --policy logs/nezha3_mine/exported/<运行目录名>/policy.pt \
  --duration 60
```

Linux 激活 `.venv-mujoco` 后可将上述 `mjpython` 替换为 `python`。MuJoCo 窗口中使用
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

训练、play 和策略导出使用 Python 3.8 的 `.venv`；MuJoCo 使用 Python 3.11
的 `.venv-mujoco`。安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

下面的命令会新建或复现这两个互相隔离的环境。

### Python 3.8 训练环境

当前工作机的 Isaac Gym Preview 4 位于 `/home/bit/isaac/isaacgym/`；迁移到
其他机器时只需修改 `ISAAC_GYM_DIR`：

```bash
ISAAC_GYM_DIR=/home/bit/isaac/isaacgym
uv python install 3.8
UV_VENV_SEED=false uv venv .venv --python 3.8
source .venv/bin/activate

uv pip install \
  torch==2.4.1 \
  torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  numpy==1.23.5 \
  scipy==1.10.1 \
  matplotlib==3.7.5 \
  tensorboard==2.14.0 \
  wandb==0.24.2 \
  pyyaml==6.0.3 \
  pillow==10.4.0 \
  imageio==2.35.1 \
  ninja==1.13.0 \
  setuptools==75.3.4

uv pip install --no-deps --no-build-isolation \
  -e "${ISAAC_GYM_DIR}/python" \
  -e ./rsl_rl \
  -e .
```

### Python 3.11 MuJoCo 环境

```bash
uv python install 3.11
UV_VENV_SEED=false uv venv .venv-mujoco --python 3.11
source .venv-mujoco/bin/activate
uv pip install "torch==2.4.1+cpu" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install "mujoco==3.10.0" "numpy==1.26.4" "pyyaml==6.0.3"
```

## 训练

```bash
source .venv/bin/activate
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

### 训练诊断指标

每次迭代在 TensorBoard/W&B 中记录以下指标，终端显示主要摘要。
`training_diagnostics.jsonl` 同时保存每轮汇总与每个小批次的 KL、重置/非重置
样本数、实际学习率和估计器漂移。`reset` 指执行本条 transition 的动作后发生
重置（`done[t]`），不是下一回合的第一条观测；某分组没有样本时只记录零计数，
不写该组 KL，避免把“未测量”误读成零。

| 指标 | 含义 | 健康范围 |
|---|---|---|
| `Diagnostics/kl_first` | 第一个 mini-batch（参数尚未更新）的 KL，用于检查 rollout 存储是否与动作所用输入一致 | ≈ 1.6e-4（KL 公式中 `+1e-5` 项的下限）；明显偏大说明存储被污染 |
| `Diagnostics/pre_update_kl`、`pre_update_kl_reset`、`pre_update_kl_non_reset` | 任何更新前，完整 rollout 及其两组样本的 KL | 16 维动作时约 1.6e-4；用于数据一致性验收，不能只依赖随机首批 |
| `Diagnostics/pre_update_reset_count` / `pre_update_non_reset_count` | 完整 rollout 中两组样本数 | 验收窗口必须实际覆盖两组 |
| `Diagnostics/pre_update_value_max_abs_error` | 更新前重算 Critic 与采样时 value 的最大绝对误差 | 应接近浮点误差 |
| `Diagnostics/kl_mean` / `kl_max` | 各 mini-batch KL 的均值/最大值 | 自适应学习率下应在 `desired_kl=0.01` 附近 |
| `Diagnostics/kl_reset` / `kl_non_reset` | 更新过程中按各组样本数加权的小批次 KL 均值 | 与更新前一致性检查区分；样本计数包含重复 epoch |
| `Diagnostics/post_update_kl` 及其 `_reset` / `_non_reset` 分组 | 完成本轮所有更新后，相对于 rollout 策略的 KL | 用于发现最后一次更新造成、调度器尚未观察到的漂移 |
| `Diagnostics/kl_estimator_shift` | 单次估计器更新本身引起的 actor 输出 KL | 远小于 0.01；若接近 0.02 会持续压低学习率 |
| `LearningRate/min` / `max` / `mean` / `floor_fraction` | 本轮实际小批次更新的学习率统计及处于 1e-5 下限的比例 | 与 `Loss/learning_rate`（迭代末值）一起看 |
| `Latent/latent_std`、`latent_std_min`、`latent_std_dim_00` … `15` | 更新前完整 rollout 的策略 latent 总体标准差（`unbiased=False`），包括逐维值 | 接近零的维度提示塌缩；均值不能替代逐维检查 |
| `Latent/latent_cos_to_mean` / `latent_cos_to_mean_median` | 更新前完整 rollout 中，各样本与均值方向的余弦相似度均值/中位数 | → 1 提示塌缩；中位数直接对全部样本计算，不平均小批次中位数 |
| `Latent/swap_vs_uniform` | swap loss 与"均匀预测" `log(K)/K` 之比 | → 1 表示对比学习不再提供信息 |
| `Latent/swav_perplexity` | 32 个原型的实际使用数（困惑度） | → 1 表示所有样本落入同一原型 |
| `ModeGate/gate_entropy`、`gate_batch_entropy`、`mode_proto_max_cos` | 门控逐样本熵、batch 平均分布的熵、模态原型间最大 \|cos\|（仅门控模式） | 逐样本熵低而 batch 熵高才说明门控在区分状态；`max_cos` → 1 表示原型共线塌缩 |

每个运行目录还会保存：

- `run_info.json`：启动命令、git commit/分支/是否有未提交修改、完整训练与环境配置；
- `git_diff.patch`：仅当启动时存在未提交修改时生成，记录相对 HEAD 的改动；
- `training_diagnostics.jsonl`：包含每轮指标及所有小批次明细，可用下述汇总脚本指定窗口分析；
- `best_policy.pt`：最近 100 个 episode 平均回报最高时保存（至少完成 `best_policy_min_episodes` 个 episode 后才开始比较）。地形课程会影响回报，部署前仍需单独评估。

所有 checkpoint 均包含 `train_cfg`、`env_cfg`、`git` 与 `best_mean_reward`，续训时恢复最佳回报记录。
新日志迭代号、`model_N.pt`、检查点 `iter` 与最佳策略的 `infos.iteration` 均表示
“已经完成 N 次更新”，从 1 开始。续训以检查点内部 `iter` 为准，不从旧文件名推断；
`--max_iterations` 在续训时表示**追加**的更新次数。加载优化器时，同时从其参数组
恢复 PPO/估计器的学习率属性，兼容没有额外学习率字段的旧检查点；
`load_optimizer=False` 则保留新优化器与新学习率。
检查点还保存累计步数/耗时，但未保存仿真状态、随机数状态及课程运行状态，
因此续训不保证与完全不中断的训练逐位一致。Git 补丁仅覆盖已跟踪文件，正式对照前
应将新增脚本和修改一并纳入实验提交并同步；本地冒烟输出不作为实验基线。

### 阶段 0 验收与后续验证

先激活训练环境，在根目录运行 CPU 回归测试：

```bash
source .venv/bin/activate
PYTHONPATH=.:rsl_rl OMP_NUM_THREADS=1 python -m unittest -v \
  legged_gym.tests.test_mine_training_pipeline \
  legged_gym.tests.test_mine_modes
```

然后分别检查真实 Isaac Gym 有门控/无门控流程（需要 NVIDIA GPU）：

```bash
python legged_gym/scripts/check_nezha_pipeline.py --mode_gate off
python legged_gym/scripts/check_nezha_pipeline.py --mode_gate on
```

该脚本在当前进程内使用 16 环境、0.2 秒回合、小地形网格和 1e-5 初始学习率，
确保每轮含重置和非重置样本；不修改正式配置文件。它逐条检查 Actor/Critic
存储观测与采样输入完全相同，训练 2 轮后创建新 runner 加载检查点，再训练 1 轮；
同时校验编号、最佳策略、学习率恢复和诊断的有限数值。通过时输出 `"status": "PASS"`，
结果保存在 `logs/phase0_smoke/<运行目录>/stage0_verification.json`。
这项检查不评估奖励或长期学习率表现。若受限会话的用户缓存不可写，可额外设置
`TORCH_EXTENSIONS_DIR=/tmp/lzhmine-torch-extensions` 和 `MPLCONFIGDIR=/tmp/lzhmine-mpl`。

阶段 1 的实验 A 使用正式配置，从头开始，不加载旧实验检查点。例如固定门控开启、
seed=10、4096 环境、target375，先运行 1500 次：

```bash
python legged_gym/scripts/train_nezha_mine.py \
  --headless --num_envs 4096 --max_iterations 1500 \
  --seed 10 --mode_gate on --no_wandb --run_name phase1_A_seed10
```

训练结束后，将实际运行目录填入下面命令；预先固定窗口为第 501–1500 次更新：

```bash
python legged_gym/scripts/summarize_training_diagnostics.py \
  logs/nezha3_mine/<运行目录> --start 501 --end 1500 --check-integrity
```

汇总脚本要求窗口完整，拒绝重复/缺失迭代和非有限数值；退出码 0 且
`integrity_pass=true` 表示该窗口的分组更新前 KL < 1e-3、Critic 重算最大误差
< 1e-4，并实际覆盖两类样本。16 维动作的 KL 含约 1.6e-4 数值偏置，不要求严格等于零。
另外检查 `lr_floor_iteration_fraction < 0.10`，同时报告小批次下限比例；观察
更新期 KL 是否在约 0.01 的量级、是否存在大幅末轮漂移。`lr_target_pass` 只判断
迭代末学习率下限占比，不代表阶段 1 自动通过。若 1500 次仍不足以判断稳定性，
按预先约定延长窗口，不根据结果挑选有利区间。

阶段 0 不改变交换损失、温度、VICReg、门控结构或奖励设置。

### 阶段 1：A/B 估计器更新顺序对照

实验 A 回答的是：**只修复训练流程、保留原有算法，学习率是否还会长期触底？**
它不是门控效果实验，也不以解决表示塌缩为目标。A 同时为 B 提供可复查的流程基准。
已完成的 `Oct15_10-20-53_phase1_A_seed10_gated_target375` 使用提交
`720ef747efda797485d74054ac1c88ee2183f89f`（启动时工作区干净）。其 501–1500 窗口中，
迭代末触底率 0.7%，小批次触底率 0.255%，小批次平均 KL 0.01407，更新后平均 KL 0.01466，
重置/非重置更新前 KL 均约 1.6e-4，Critic 重算误差为零，满足实验 A 的验收目标。
潜在表示余弦相似度中位数仍约 0.9997，因此阶段 2 仍需单独开展。

现在可通过 `algorithm.estimator_update_schedule` 或 CLI 选择更新安排：

| 参数值 | 每轮更新顺序 | 用途 |
| --- | --- | --- |
| `interleaved`（默认） | 每个小批次计算 PPO 前向和 KL → 更新估计器 → PPO 反向更新，重复全部 epoch | A，保留原实现 |
| `after_ppo` | 完成全部 PPO 小批次更新 → 按原小批次顺序完成全部估计器更新 | B，仅后移估计器更新 |

B 每轮仍执行 5 epoch × 4 minibatch = 20 次 PPO 更新与 20 次估计器更新，
重放 PPO 使用的相同样本索引和该批次调度后的学习率；不再次打乱样本、不缓存完整批次副本，
也不是把估计器改成每轮只更新一次。重放的是 **B 自身** 的逐批学习率，不是 A 的历史学习率。
自适应调度规则保持不变，A/B 实际学习率轨迹可以因 KL 不同而不同。
损失、温度、原型路径、门控结构、奖励、初始学习率与更新次数均保持不变。

配置随 `run_info.json` 和 checkpoint 保存。续训/play 不传参数时采用 checkpoint 的安排；
旧 checkpoint 没有该字段时视为 `interleaved`。显式指定与 checkpoint 不同的安排会报错，
不能把 A 的检查点直接续训成正式 B 对照。

提交并同步本次修改后，先复跑单元测试与 B 的真实 GPU 短测：

```bash
source .venv/bin/activate
PYTHONPATH=.:rsl_rl OMP_NUM_THREADS=1 python -m unittest -v \
  legged_gym.tests.test_mine_training_pipeline \
  legged_gym.tests.test_mine_modes \
  legged_gym.tests.test_mine_update_schedule \
  legged_gym.tests.test_mine_diagnostic_summary

python legged_gym/scripts/check_nezha_pipeline.py \
  --mode_gate on --estimator_update_schedule after_ppo --run_name phase1_B_smoke
```

短测还会断言真实优化器 step 的顺序/次数、逐批学习率重放和检查点安排恢复。
正式 B 从头训练，保持与 A 相同的环境数、seed、门控模式、target375 和迭代数：

```bash
python legged_gym/scripts/train_nezha_mine.py \
  --headless --num_envs 4096 --max_iterations 1500 \
  --seed 10 --mode_gate on --no_wandb \
  --estimator_update_schedule after_ppo --run_name phase1_B_seed10

python legged_gym/scripts/summarize_training_diagnostics.py \
  logs/nezha3_mine/<B实际运行目录> --start 501 --end 1500 --check-integrity
```

用同一脚本、同一 501–1500 窗口重新汇总 A，比较触底比例、KL 均值、p95/p99、
极值及超过 0.05/0.1 的次数。小批次分位数以每批的平均 KL 为统计单元；
阶段/分组分位数以每轮的平均 KL 为统计单元，并非单样本 KL 的分布。
`*_kl_sample_max` 单独记录样本级最大值。新增诊断的含义是：

| 指标 | 含义 |
| --- | --- |
| `post_ppo_kl`（B） | 全部 PPO 更新之后、估计器更新之前，相对采样策略的完整 rollout KL |
| `estimator_phase_kl`（B） | Actor/std 不变时，整段估计器更新前后策略的 KL；直接比较分布，不含旧公式的 1e-5 偏置 |
| `post_update_kl`（A/B） | 所有 PPO 与估计器更新完成后，相对采样策略的 KL，是共同的最终漂移指标 |
| `update_ppo_count` / `update_estimator_count` | 每轮两类更新次数，汇总时与小批次数交叉检查 |

以上阶段 KL 均按重置/非重置样本额外分组。`estimator_phase_kl` **不等于**
`post_update_kl - post_ppo_kl`，KL 不能这样相减分解。B 的逐小批次
`kl_estimator_shift` 在对应的后置估计器更新前后测量，不与早先 PPO 的均值比较。

继续使用原验收标准：数据一致性通过、迭代末学习率触底比例 < 10%、KL 约 0.01。
不能只因 B 的 PPO 内 KL 或触底率更小就判定更好，还要检查完整更新后的 KL 尾部，
避免把漂移转移到学习率调度器看不到的估计器阶段。如果 B 没有稳定收益，保留 A。
单个 seed 的短训练只用于选择后续更新安排，不证明门控优劣或跨 seed 的普遍优势。
选定安排后，才进入阶段 2 的无门控表示塌缩消融；本次未实现温度/VICReg/原型平均修复。

## Isaac Gym Play 与导出

Play：

```bash
source .venv/bin/activate
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
source .venv-mujoco/bin/activate
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
source .venv-mujoco/bin/activate
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

训练管线回归测试（rollout 存储与动作输入一致、首个 mini-batch KL 为零、latent/门控诊断、`best_policy.pt` 与配置/git 记录），不需要 Isaac Gym：

```bash
PYTHONPATH=.:rsl_rl python -m unittest legged_gym.tests.test_mine_training_pipeline -v
```

`.gitignore` 忽略虚拟环境（包括 `.venv-mujoco`）、训练日志、checkpoint/导出权重、录像截图、缓存、构建产物和本机账户/代理状态。权重留在本地，通过运行目录和文件名选择。

URDF/Xacro、配套 STL/OBJ 网格、MJCF/XML、配置文件、地形生成脚本，以及仿真场景实际引用的地形 `.bin` / `.json` 继续纳入 Git。不要整体忽略 `resources/` 或机器人资源目录，否则新机器无法加载机器人和地形。
