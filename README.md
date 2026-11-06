# Nezha CTS · 八组 MLP／MoE 消融 · Isaac Gym

本项目使用 CTS 教师—学生联合训练，并对学生编码器、Actor、Critic 的 MLP／MoE 结构进行八组完整组合对照。教师编码器始终为 MLP。教师—学生网络与原有学生 MoE 采用 `legbot_lab` 的 `WF-CTS-MOE` 设计（参考提交 `daff9fbe3d3eed222b9e39e1dcefa1996f690834`），运行平台为 Isaac Gym。

## 八组独立实验

下表中的 MoE 均为 **8 专家、独立 Softmax 门控、全部专家软加权**。三列依次对应学生编码器（S）、Actor（A）和 Critic（C）的结构开关。

| 实验目录 | 学生编码器 S | Actor A | Critic C | 训练任务 |
|---|---|---|---|---|
| [g0_mlp](experiments/g0_mlp/README.md) | MLP | MLP | MLP | `nezha_cts_g0_mlp` |
| [g1_student_moe](experiments/g1_student_moe/README.md) | MoE | MLP | MLP | `nezha_cts_g1_student_moe` |
| [g2_actor_moe](experiments/g2_actor_moe/README.md) | MLP | MoE | MLP | `nezha_cts_g2_actor_moe` |
| [g3_student_actor_moe](experiments/g3_student_actor_moe/README.md) | MoE | MoE | MLP | `nezha_cts_g3_student_actor_moe` |
| [g4_critic_moe](experiments/g4_critic_moe/README.md) | MLP | MLP | MoE | `nezha_cts_g4_critic_moe` |
| [g5_student_critic_moe](experiments/g5_student_critic_moe/README.md) | MoE | MLP | MoE | `nezha_cts_g5_student_critic_moe` |
| [g6_actor_critic_moe](experiments/g6_actor_critic_moe/README.md) | MLP | MoE | MoE | `nezha_cts_g6_actor_critic_moe` |
| [g7_all_moe](experiments/g7_all_moe/README.md) | MoE | MoE | MoE | `nezha_cts_g7_all_moe` |

`g1_student_moe` 是仅学生编码器使用 MoE 的基准组。八组只改变上述模块的结构，观测、奖励、课程、动力学、网络隐藏层宽度配置、PPO 和蒸馏超参数共用同一实现。学生使用 MLP 时仍然蒸馏教师特征；该组没有学生门控，其专家使用均衡项不适用。结构改变会改变参数量，因此这不是参数量匹配对照。

每组都有自己的 `config.py`、`train.py`、`play.py`、`export.py`、`simulate.py` 和 README。日志写入 `logs/nezha_moe_cts/<组名>/`；导出策略写入 `deploy/deploy_mujoco/nezha/policy/<组名>/policy.pt`。默认最新 checkpoint 查找仅搜索本组目录，续训、play 和导出会检查三个模块的类型及专家数量。

旧两组实验目录已被八组目录替换，已有训练日志和策略文件保留原位。旧 `exp01_baseline` 对应新 `g1_student_moe` 的结构，旧 `exp02_actor_moe` 对应新 `g3_student_actor_moe`；兼容的旧 checkpoint 可通过绝对路径在对应结构入口使用。普通 CTS 训练入口 `legged_gym/scripts/train_nezha_moe_cts.py` 默认进入 `g1_student_moe`。

## 网络与训练

- 教师：295 维特权观测 → MLP [512, 256] → 32 维 L2 归一化 latent，八组均相同。
- 学生：228 维本体历史 → MLP 或 8 专家 MoE → 32 维 L2 归一化 latent；隐藏层配置均为 [512, 256, 256]。
- Actor：当前本体观测 57 + latent 32 = 89 维输入，输出 16 维动作均值；使用 MLP 或 MoE，隐藏层配置 [512, 256, 128]。每组内教师与学生共用 Actor。
- Critic：当前特权观测 295 + detached latent 32 = 327 维输入，输出 1 维价值；使用 MLP 或 MoE，隐藏层配置 [512, 256, 128]。每组内教师与学生共用 Critic。
- 默认 75% 环境由教师控制，25% 由学生控制。两类轨迹都用于训练共享 Actor/Critic；PPO 更新教师编码器但不更新学生编码器，价值损失不回传编码器。
- 每轮先完成全部 PPO 更新，再用学生控制环境中保存的**同一时刻**本体／特权观测进行 latent MSE 蒸馏。学生为 MoE 时附加系数 0.01 的专家使用均衡损失；学生为 MLP 时只使用 latent MSE。两个优化器分别保存和恢复。
- Actor 的专家及门控由 PPO 策略损失训练，Critic 的专家及门控由价值损失训练；二者不额外引入专家均衡或门控熵辅助损失。
- PPO 的教师与学生 surrogate loss 分别取均值后相加；value/entropy 对合并批次取均值。默认 rollout 24 步、5 epochs、4 mini-batches、gamma 0.99、lambda 0.95。
- 学生、Actor、Critic 的门控参数互不共享。各自的 `expert_0…expert_7` 没有预设对应关系或轮式／腿式模式标签。门控熵用于诊断，不单独作为新的优化目标。

MoE 延续原有“共享专家特征主干＋独立输出头”设计，各模块的门控使用自己的 MLP：

| MoE 模块 | 专家特征主干 | 每个专家输出头 | 独立门控 |
|---|---|---|---|
| 学生编码器 | 228 → 512 → 256 → 8×256 | 256 → 32 | 228 → 512 → 256 → 256 → 8 → Softmax |
| Actor | 89 → 512 → 256 → 8×128 | 128 → 16 | 89 → 512 → 256 → 128 → 8 → Softmax |
| Critic | 327 → 512 → 256 → 8×128 | 128 → 1 | 327 → 512 → 256 → 128 → 8 → Softmax |

专家输出按照非负、和为 1 的门控权重求和。只有学生 latent 在加权后进行 L2 归一化；动作均值和价值不做该归一化。Actor 仍使用原有单高斯分布，训练时对组合后的均值采样，部署时直接使用均值。Critic 及其门控只用于训练。

公共配置位于 `legged_gym/envs/nezha/nezha_moe_cts_config.py`。各组 `config.py` 显式覆盖 `student_encoder_type`、`expert_num`、`actor_type`、`actor_expert_num`、`critic_type` 和 `critic_expert_num`；MLP 的专家数记为 0，MoE 为 8。每组配置在注册时独立复制。新的完整消融应从零训练，保持配对种子、并行环境数及训练步数一致。

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

Isaac Gym 的高度来自原有 heightfield 邻点采样，接触来自 PhysX net contact tensor；它们与 Isaac Lab 的射线／接触传感器并非完全相同的数值实现。这里对齐输入定义、维数、顺序和预处理；地形采用下述 WF 配置，速度范围按 Nezha 原来的跟踪奖励条件扩展，机器人 URDF、控制与其余奖励／动力学仍使用 Nezha 配置。腿部动作尺度为 0.25，轮速尺度为 **10**，控制频率 50 Hz。

## WF 地形与 Nezha 速度课程

新训练使用 `legbot_wf`，参考 `legbot_lab@daff9fbe` 的 `LegbotEnvCfg` 和 `TERRAIN_CFG`（不是同文件中的 `DEFAULT_TERRAIN_CFG`）。高度场实现对照其依赖的 [Isaac Lab v2.3.2 源码](https://github.com/isaac-sim/IsaacLab/tree/v2.3.2/source/isaaclab/isaaclab/terrains)。

- 10 行 × 20 列，每块 9×9 m，包含四边各 0.5 m 的子地形边界；全图外圈 25 m。
- 水平／垂直分辨率 0.1／0.005 m；初始等级均匀采样 0～5。
- 第 `r` 行难度为 `(r + U[0,1)) / 10`，不是固定 `r/10`；按列、行顺序生成，使用与参考相同的独立难度随机数生成器。
- 地面静／动摩擦系数均为 1，恢复系数为 0；episode 为 25 s。出生 x/y 偏移 ±0.5 m、z 偏移 0～0.05 m、yaw 为 ±3.14。

| 地形名称 | 比例／列数 | 随难度变化的参数 | 坡面修正阈值 |
|---|---|---|---|
| `wave` | 5%／1 | 幅值 0.1～0.28 m，5 个波，叠加 ±0.05 m 粗糙度 | 10 |
| `slope_up` | 10%／2 | 倒置金字塔，坡度 0.1～0.568，中心平台 3 m | 10 |
| `slope_down` | 10%／2 | 正向金字塔，同上 | 10 |
| `rough_slope` | 5%／1 | 正向坡面，叠加 ±0.05 m 粗糙度 | 10 |
| `stairs_up` | 25%／5 | 倒置台阶，高 0.05～0.257 m，宽 0.31 m，平台 3 m | 0.25 |
| `stairs_down` | 10%／2 | 正向台阶，同上 | 0.25 |
| `obstacles` | 20%／4 | 20 个随机障碍，高度范围 0.05～0.275 m，宽度范围 1～2 m，平台 3 m | 0.25 |
| `flat` | 15%／3 | 平地 | 无 |

名称和正／倒置方向直接沿用参考定义。`stepping_stones`、`gap` 的比例均为 0，不生成。高度场边界包含额外采样点，每块为 91×91 格、内部为 79×79 格；台阶宽度按参考离散为 3 格，粗糙度使用双三次插值。三角网格按每类地形自己的阈值修正。Gym 全图保留原坐标约定，较 Lab 整体平移，局部几何一致。

地形课程在 episode reset 前评估：历史最大平面位移大于 4 m 时升级；否则，当最大位移小于 `0.5 × ||累计 xy 指令|| × 5 × (1 - 零指令概率)` 时降级。累计指令按每次重采样的速度向量求和，正反指令会抵消。等级低于 0 时裁剪，超过最高等级 9 时重新随机分配到 0～9。

速度全局范围恢复 Nezha 原设定，使用 `commands.wf_settings["range_curriculum"] = "nezha_reward"`：

| 指令 | 初始范围 | 每次达标扩展量 | 最大范围 |
|---|---|---|---|
| x 速度（m/s） | ±1.0 | 两端各 0.2 | ±1.5 |
| y 速度（m/s） | ±0.6 | 不扩展 | ±0.6 |
| yaw 角速度（rad/s） | ±1.0 | 两端各 0.1 | ±1.5 |

沿用 Nezha 的检查时机：有环境重置，且全局控制步计数是最大回合步数的整数倍时检查（当前每 1250 步）。对这批重置环境，`tracking_lin_vel` 的回合累计奖励均值除以最大回合步数，严格超过该项每步奖励权重的 80% 才扩展 x；`tracking_ang_vel` 独立决定 yaw 是否扩展。侧向范围固定，提前失败的短回合也按最大回合步数归一化。速度不再根据 20000／50000／75000 轮自动增加。

其余 WF 采样保持原样：全局范围仍与各地形限速取交集，平地／波浪／坡面最终 x 上限为 ±1.5 m/s，台阶／障碍 x 上限仍为 ±1 m/s；所有地形 y 保持 ±0.6 m/s，yaw 最高 ±1.5 rad/s。默认每 5 s 重采样，保留动态速度下界、20% 极限指令采样和连续极限指令反向。零指令概率在前 1500 轮从 0 增至 15%，其中 20% 附加极限 yaw。本次只恢复全局速度范围和奖励达标扩展规则，没有改成原 Nezha 的 10 s 均匀采样。

两项奖励课程也同步：`lin_vel_z` 权重在 0～1500 轮从 -2 变到 -1；`base_height` 在 0～5000 轮从 -1 变到 -10。课程时钟按每轮 rollout 步数换算；显式 reset 不额外推进该时钟。其余 Nezha 奖励不变，因此这不是整个 Isaac Lab 任务的动力学或奖励复刻。

**使用新的速度范围与奖励达标课程请从零启动训练。** 旧 checkpoint 续训会恢复它保存的旧环境配置：旧 WF checkpoint 继续使用按轮次扩速，旧地形 checkpoint 继续使用原地形。新 checkpoint 会额外保存已扩展的速度范围，续训后保持进度。

八组任务的新训练都直接进入 `WFTerrain` 和 `WFCurriculum`，速度范围选择其中的 `nezha_reward` 模式，不会执行旧 `wheel_gym_cq` 地形分支。旧分支保留用于恢复旧 checkpoint。Nezha 机器人资产、控制、其余奖励、随机化、终止判据，以及 Isaac Gym 物理后端和高度／接触采样仍沿用本项目实现。

## 训练、续训、play 和导出

本机训练环境为 `.venv`，MuJoCo 环境为 `.venv-mujoco`。入口自动配置仓库包路径、编译缓存和线程数，无需手动设置环境变量。默认使用 `cuda:0`、GPU pipeline、无窗口训练和 TensorBoard 记录。

从零训练只需指定最大轮次、并行环境数和种子（示例先训练 20000 轮）：

以仅学生 MoE 的基准组为例：

```bash
cd /home/bit/LZHMine
.venv/bin/python experiments/g1_student_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10
```

运行其他组时，将路径中的 `g1_student_moe` 替换为表中的组名即可，例如 `g0_mlp` 或 `g7_all_moe`。各组 README 提供对应的完整命令。速度范围是否扩大取决于跟踪奖励；默认最大训练轮数为 100000，可通过 CLI 覆盖，也可后续续训。需要训练窗口时增加 `--viewer`。启动信息会打印实验编号和三个模块的结构。

在另一个终端启动 TensorBoard：

```bash
cd /home/bit/LZHMine
.venv/bin/tensorboard --logdir /home/bit/LZHMine/logs/nezha_moe_cts \
  --host 127.0.0.1 --port 6007
```

浏览器打开 <http://127.0.0.1:6007/#scalars>，在左侧勾选本次运行。远程训练时，在本地执行 `ssh -L 6007:127.0.0.1:6007 用户名@训练服务器`，再访问同一地址。若端口已占用，确认浏览器访问的是本项目的 TensorBoard，或换用空闲端口。

当前 Python 3.8 环境使用 TensorBoard 2.14，需要 `protobuf<5`；项目已在 `rsl_rl/setup.py` 添加对应的条件依赖。若旧环境出现 `MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`，执行 `uv pip install --python /home/bit/LZHMine/.venv/bin/python 'protobuf>=4.25.8,<5'`，然后重启 TensorBoard 并刷新浏览器。无需重新训练，已有事件日志可以继续使用。

TensorBoard 的上述父目录可以同时查看八组及保留的旧日志。新目录为 `logs/nezha_moe_cts/<组名>/<时间>_<运行名>/`，终端会打印实际路径；可通过 `--run_name` 自定义运行名。默认无需 W&B 登录，增加 `--wandb` 后会同时上传 W&B，各组的 W&B group 与组名一致。续训使用当前实验组的默认运行名，也可通过 `--run_name` 指定。

TensorBoard 在首轮采样前创建日志，每轮记录指标直到最后一轮。第一轮 loss 在首次更新完成后出现，回报／episode 长度在对应组有回合结束后出现；无需等待 1000 轮模型保存。首轮指标立即落盘，后续最多约 10 秒批量刷新，正常结束或 Ctrl+C／Python 异常退出时刷新并关闭日志。TensorBoard 服务可在训练前或训练后启动，都能读取已经记录的完整历史。`config.json` 和 checkpoint 同时保存环境配置，可确认实际使用的地形与课程。重点查看：

- `Train/teacher_reward`、`Train/student_reward`：教师／学生回报。
- `Train/teacher_length`、`Train/student_length`：episode 长度。
- `Loss/value`、`Loss/teacher_surrogate`、`Loss/student_surrogate`：PPO 损失。
- `Loss/mean_latent_loss`：所有组的学生特征蒸馏损失；`Loss/mean_load_balance_loss` 仅在学生使用 MoE 时表示专家均衡损失，MLP 学生没有该约束。
- `Experts/gate_entropy`、`Experts/expert_0_usage` 至 `Experts/expert_7_usage`：学生为 MoE 时的门控熵与平均软权重。
- `ActorExperts/gate_entropy`、`ActorExperts/expert_0_usage` 至 `ActorExperts/expert_7_usage`：Actor 为 MoE 时的门控熵与平均软权重，来自 PPO 更新批次。
- `CriticExperts/gate_entropy`、`CriticExperts/expert_0_usage` 至 `CriticExperts/expert_7_usage`：Critic 为 MoE 时的门控熵与平均软权重。
- `ActorExperts/teacher_gate_entropy`、`ActorExperts/student_gate_entropy`，以及 `teacher_expert_0_usage…7`、`student_expert_0_usage…7`：分教师／学生输入统计 Actor 路由；总体指标按样本数量加权。熵上限为 `ln(8)≈2.079`，使用均衡或高熵本身不能证明专家分工有效。
- `Episode/max_command_x`、`Episode/max_command_yaw`：奖励达标后扩展的全局速度上限；实际采样还受各地形限速约束。

训练每 **1000 轮**保存一次 checkpoint，文件名为 `model_1000.pt`、`model_2000.pt` 等，保留各轮版本。训练正常结束时也会保存最后一轮，即使该轮不是 1000 的整数倍。训练需至少两个环境；当前 runner 面向单设备训练。

以下续训、play、导出和 MuJoCo 命令以 `g1_student_moe` 为例。其他组使用自己目录中的同名入口和对应日志／策略路径，完整命令见各组 README。续训示例：

```bash
.venv/bin/python experiments/g1_student_moe/train.py \
  --max_iterations 80000 --num_envs 4096 --seed 10 \
  --resume \
  --load_run "/home/bit/LZHMine/logs/nezha_moe_cts/g1_student_moe/实际运行目录名" \
  --checkpoint -1
```

续训的 `--max_iterations 80000` 表示**追加 80000 轮**，例如从第 20000 轮接着训练到累计 100000 轮；`--load_run` 填运行目录路径，`--checkpoint -1` 选择该目录中最新的 checkpoint，也可传 `1000` 等轮数。续训会创建新的日志目录，TensorBoard 可同时查看各次运行；导出续训结果时需填写新目录下的模型路径。

Checkpoint 保存网络与观测 schema、环境配置、控制参数、两个优化器和训练计数。续训先恢复观测配置再创建环境，可以用 `--num_envs` 调整并行数量；物理状态、历史和每个环境的地形等级重新初始化。新模式的已扩展速度范围保存于 `command_curriculum_state`，恢复后再采样首条指令；WF 零指令概率与奖励课程仍由恢复的训练步数推导。旧 WF checkpoint 保持原来的按轮次速度课程，旧地形 profile 的运行期指令课程状态不恢复。旧架构 checkpoint 会明确报错。

学生 play（直接指定第 1000 轮 checkpoint，默认一个环境）：

```bash
.venv/bin/python experiments/g1_student_moe/play.py \
  --checkpoint_path "/home/bit/LZHMine/logs/nezha_moe_cts/g1_student_moe/实际运行目录名/model_1000.pt" \
  --play_num_envs 1
```

独立导出学生策略（不需要 Isaac Gym/GPU）。八组都导出到 `deploy/deploy_mujoco/nezha/policy/` 下各自的组目录，目录不存在时自动创建。以下导出 `g1_student_moe` 的第 1000 轮 checkpoint，省略 `--output` 也使用同一路径：

```bash
.venv/bin/python experiments/g1_student_moe/export.py \
  --checkpoint "/home/bit/LZHMine/logs/nezha_moe_cts/g1_student_moe/实际运行目录名/model_1000.pt" \
  --output "/home/bit/LZHMine/deploy/deploy_mujoco/nezha/policy/g1_student_moe/policy.pt"

# 或导出该运行最新的 checkpoint
.venv/bin/python experiments/g1_student_moe/export.py \
  --run "/home/bit/LZHMine/logs/nezha_moe_cts/g1_student_moe/实际运行目录名" \
  --output "/home/bit/LZHMine/deploy/deploy_mujoco/nezha/policy/g1_student_moe/policy.pt"
```

`model_<轮数>.pt` 是训练 checkpoint，MuJoCo 使用导出后的 `policy.pt`。导出的文件只包含学生编码器（MLP 或 MoE）、共享 Actor 及必要元数据，输入 [N, 228]、输出 [N, 16]。历史由 runtime 管理；教师、Critic 和优化器不进入部署模型。导出会核对 TorchScript 与 checkpoint 的动作一致性，并写出包含结构及观测信息的 `policy.json`。

八组入口共用导出与 MuJoCo 实现，默认策略文件分别保存。`get_expert_probabilities(history)` 及 MuJoCo 学生专家显示对应学生编码器；MLP 学生没有专家权重。Actor 权重由 `get_actor_expert_probabilities(history)` 查询，MLP Actor 同样没有专家权重。Critic 不参与部署，也不在部署界面显示其门控。

每次导出会覆盖该目录中的 `policy.pt` 和 `policy.json`；play 使用 `--no_export` 可跳过自动导出。如需保留多个策略，可通过 `--output` 在同一目录下指定不同文件名，例如 `policy_1000.pt`，仿真时在 `--policy` 中填写对应文件路径。

MuJoCo 可视化仿真（入口自动选择本组导出的策略，前进速度指令为 0.5 m/s）。八组复用原 YAML 的地形与控制设置，入口自动配置包路径，无需设置 `PYTHONPATH`：

```bash
.venv-mujoco/bin/python experiments/g1_student_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0
```

无窗口短时仿真：

```bash
.venv-mujoco/bin/python experiments/g1_student_moe/simulate.py \
  --headless --duration 30 --vx 0.5 --vy 0.0 --yaw 0.0
```

分组仿真入口默认使用本组 `policy.pt`，不需要再写 `--policy`；可显式指定同结构的另一个导出文件，入口会校验学生、Actor 和训练时 Critic 的结构元数据。直接调用 `deploy_mujoco.py -c nezha_moe_cts.yaml` 使用 YAML 指定的默认策略，当前为 `g1_student_moe/policy.pt`。可视化窗口支持 `W/S` 调前后速度、`A/D` 调侧向速度、`Q/E` 调转向、空格停止、`R` 复位到当前测试道入口。`--headless` 模式必须设置正数 `--duration`（单位为仿真秒）。默认场景为 `scene_course.xml`。

场景包含七条并排测试道，每条横向宽 5 m，保留原来的 10 m 出发区、10 m 障碍区和 5 m 结束区。窗口内按数字键 `1`～`7` 可切换到对应测试道入口并清零速度指令，再按 `W` 前进；默认从第 1 条开始。

| 数字键 | 测试道名称（`--terrain`） | 地形参数 |
|---|---|---|
| 1 | `stairs_15cm` | 单级高 15 cm，踏面深 30 cm |
| 2 | `stairs_20cm` | 单级高 20 cm，踏面深 30 cm |
| 3 | `stairs_25cm` | 单级高 25 cm，踏面深 30 cm |
| 4 | `ramp_15deg` | 15° 光滑上坡—平台—下坡 |
| 5 | `ramp_25deg` | 25° 光滑上坡—平台—下坡 |
| 6 | `ramp_30deg` | 30° 光滑上坡—平台—下坡 |
| 7 | `rubble_speed_bumps` | 不平路面叠加圆滑横向减速带，最高 20 cm |

楼梯每侧上／下各 5 级，顶部平台长 1 m，每级横向长 5 m，使用实体几何保留竖直立面。坡道上下坡的水平长度各为 3.5 m，顶部平台长 1 m；坡面及顶平台滑动摩擦系数为 **0.35**，普通地面、楼梯和减速带为 **1.0**。减速带沿行驶方向宽 80 cm，共 5 条，高度依次为 12、16、20、16、20 cm。可以直接选择起始测试道：

```bash
.venv-mujoco/bin/python experiments/g1_student_moe/simulate.py \
  --terrain ramp_25deg
```

上述命令默认零速度，在入口待命。地形参数集中在 `deploy/deploy_mujoco/tools/generate_terrain_course.py`；修改后执行以下命令即可同步重建高度场、实体地形和元数据：

```bash
.venv-mujoco/bin/python deploy/deploy_mujoco/tools/generate_terrain_course.py
```

传感器观测、裁剪、缩放和历史顺序由模型中的 schema 驱动。部署 runtime 校验模型附带的名义姿态、PD 增益、关节顺序、控制周期和动作尺度等控制参数。详细说明见 [deploy/readme.md](deploy/readme.md)。

## 后续修改入口

| 内容 | 文件 |
|---|---|
| 各组学生／Actor／Critic 类型、日志目录和导出路径 | `experiments/g0_mlp/config.py` 至 `experiments/g7_all_moe/config.py` |
| 观测项、scale、noise、clip、历史排列的统一定义 | `rsl_rl/rsl_rl/utils/cts_observations.py` |
| Isaac Gym 原始观测采集、接触历史、reset | `legged_gym/envs/nezha/nezha_moe_cts_env.py` |
| 历史帧数、专家数、latent、网络层宽、训练参数 | `legged_gym/envs/nezha/nezha_moe_cts_config.py` |
| Nezha 控制、奖励、随机化与资产配置 | `legged_gym/envs/nezha/nezha_robot_config.py` |
| WF 地形参数、离散高度场与网格组装 | `legged_gym/terrains/wf.py` |
| WF 速度采样、地形升级降级、奖励课程 | `rsl_rl/rsl_rl/utils/wf_curriculum.py` |
| MoE 与教师／学生 Actor-Critic | `rsl_rl/rsl_rl/modules/moe.py`、`actor_critic_moe_cts.py` |
| 两阶段训练与 rollout | `rsl_rl/rsl_rl/algorithms/moe_cts.py`、`storage/rollout_storage_cts.py` |

调整历史长度用 `NezhaMoECTSCfg.env.frame_stack`；调整观测项可修改 `default_observation_config()`，或在 `cfg.env.observation_config` 提供完整 schema。维数从 schema 自动推导。新增传感量时，需同时实现环境和部署的原始量采集；不要只修改维数常量。CTS 实际预处理不读取旧基类的 `normalization.obs_scales`。

## 验证

```bash
# 训练环境需已安装 pytest
OMP_NUM_THREADS=1 PYTHONPATH="$PWD/rsl_rl:$PWD" \
  .venv/bin/python -m pytest legged_gym/tests tests -q
```

自动化验证覆盖输入维数、逐项预处理、轮速、历史／reset、梯度隔离、同刻蒸馏、尾批次完整采样、checkpoint 续训、TorchScript 一致性和部署控制参数校验。短时仿真仅验证接口与数值运行，策略运动表现需要正式训练与评估。

结构组合验证检查 MLP／MoE 学生、Actor、Critic 的前向维数、软加权、梯度隔离和导出一致性。实验分组验证检查配置独立性、本组最新模型查找，以及学生、Actor、Critic 类型或专家数量不匹配时的拒绝加载行为。短时运行不作为各组运动性能优劣的证据。

WF 验证还覆盖官方地形高度场 golden、类型比例／难度／边界／原点／种子复现／每类坡面修正阈值，以及旧速度阶段兼容、地形限速、累计指令、最大位移、零指令、奖励课程和 reset 时钟边界。Nezha 速度课程验证严格 80% 阈值、x/yaw 独立扩展、步长与上限、检查时机、采样后范围保持及 checkpoint 续训。

实际 Isaac Gym 短时物理回归（需训练环境，默认 16 个环境和小地形）：

```bash
TORCH_EXTENSIONS_DIR=/tmp/cts_torch_extensions python legged_gym/tests/test_env.py --headless --rl_device cpu --sim_device cpu --pipeline cpu --smoke_terrain trimesh
```

通用入口 `legged_gym/scripts/train.py` 和 `play.py` 也分别转入上述 CTS 训练与学生评估流程。
