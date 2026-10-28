# Nezha MoE-CTS · Isaac Gym

本分支同时维护原始 CTS baseline 和 Actor MoE 对照实验。教师–学生网络与学生 MoE 门控采用 `legbot_lab` 的 `WF-CTS-MOE` 设计（参考提交 `daff9fbe3d3eed222b9e39e1dcefa1996f690834`），运行平台为 Isaac Gym。旧 MINE 任务、模型、训练／play／导出入口和部署 runtime 已删除；已有训练日志和模型文件保留。

## 两组独立实验

| 实验目录 | Actor | Critic | 训练任务 |
|---|---|---|---|
| [exp01_baseline](experiments/exp01_baseline/README.md) | 原始 MLP [512, 256, 128] | MLP | `nezha_moe_cts` |
| [exp02_actor_moe](experiments/exp02_actor_moe/README.md) | 8 专家＋独立 Softmax 门控 | MLP | `nezha_moe_cts_actor_moe` |

第一组保留 `main@1f365bd` 的网络设计；第二组只替换 Actor。两组的学生编码器都是原有 8 专家 MoE，观测、奖励、课程、动力学和 PPO 设置共用同一实现。每组都有自己的 `config.py`、`train.py`、`play.py`、`export.py`、`simulate.py` 和 README，无需手动修改全局 Actor 开关。

新日志分别写入 `logs/nezha_moe_cts/exp01_baseline/` 和 `logs/nezha_moe_cts/exp02_actor_moe/`；导出策略分别写入 `deploy/deploy_mujoco/nezha/policy/exp01_baseline/policy.pt` 和 `deploy/deploy_mujoco/nezha/policy/exp02_actor_moe/policy.pt`。默认查找最新 checkpoint 只搜索本组日志。分组入口在续训、play 和导出时检查 Actor 类型和专家数量，误选另一组会明确报错。

旧日志不搬迁：可以通过绝对路径，在对应架构的实验入口继续使用。原 `legged_gym/scripts/train_nezha_moe_cts.py` 训练入口现已恢复为第一组 baseline；公共模型同时支持两种 Actor，环境和算法代码无需复制。

## 网络与训练

- 教师：295 维特权观测 → MLP [512, 256] → 32 维 L2 归一化 latent。
- 学生：228 维本体历史 → 8 个专家与 Softmax gate → 加权求和 → 32 维 L2 归一化 latent。专家共享特征主干，后接分组线性输出；隐藏层配置为 [512, 256, 256]。
- Actor 输入均为当前本体观测 57 + latent 32 = 89 维，输出 16 维动作均值。第一组使用原始 MLP；第二组使用 8 个动作专家和独立门控，将专家输出加权求和。每组内教师与学生共用 Actor。
- 共享 Critic：当前特权观测 295 + detached latent 32 = 327 维，隐藏层 [512, 256, 128]。
- 默认 75% 环境由教师控制，25% 由学生控制。两组都采样训练共享 Actor/Critic，PPO 更新教师编码器但不更新学生编码器，价值损失不回传编码器。
- 每轮先完成全部 PPO 更新，再用学生控制环境中保存的**同一时刻**本体／特权观测进行 latent MSE 蒸馏，另加系数 0.01 的学生专家使用均衡损失。第二组的 Actor 专家及门控均由 PPO 优化，不额外引入 Actor 均衡／熵辅助损失。两个优化器分别保存和恢复。
- PPO 的教师与学生 surrogate loss 分别取均值后相加；value/entropy 对合并批次取均值。默认 rollout 24 步、5 epochs、4 mini-batches、gamma 0.99、lambda 0.95。
- 第二组的学生 gate 和 Actor gate 分别输出自己的 `expert_0…expert_7` 权重，参数不共享、专家编号没有对应关系，也没有预设的轮式／腿式／混合模式标签。是否形成不同的动作行为需要通过训练与评估验证。

第二组 Actor 沿用学生 MoE 的“共享特征主干＋独立输出头”设计，并非 8 套完全独立的完整 MLP：

| Actor 部件 | 默认结构 |
|---|---|
| 专家特征主干 | 89 → 512 → 256 → 8×128，隐藏层使用 ELU |
| 专家输出头 | 8 个独立的 128 → 16 线性头，通过分组线性运算实现 |
| 独立门控网络 | 89 → 512 → 256 → 128 → 8 → Softmax |
| 动作组合 | `mean = sum(weight[i] * expert_action[i])`，权重非负且和为 1 |
| 训练动作 | 对组合后的 mean 使用原有每动作维共享标准差采样一次高斯动作 |
| 部署动作 | 直接使用组合后的 mean；不对动作做 latent 的 L2 归一化 |

开源参考：[moe-rsl-rl 的 GatedMoENet](https://github.com/iit-DLSLab/moe-rsl-rl/blob/main/moe_rsl_rl/modules/ac_moe_gated.py) 也实现了门控与专家输出的软加权。本项目复用本地 `moe.py`，保留现有 PPO 单高斯分布，不引入该库的额外分布或训练依赖。

公共配置位于 `legged_gym/envs/nezha/nezha_moe_cts_config.py`，默认保持 MLP Actor。各组目录内的 `config.py` 只指定本组 Actor 和日志元数据，注册时复制成互相独立的配置实例。学生专家数仍由公共 `expert_num` 配置。旧 checkpoint 缺少 Actor 类型时按 MLP 恢复；第二组请从零训练或恢复已有 MoE Actor checkpoint。

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

两组任务的新训练都直接进入 `WFTerrain` 和 `WFCurriculum`，速度范围选择其中的 `nezha_reward` 模式，不会执行旧 `wheel_gym_cq` 地形分支。旧分支保留用于恢复旧 checkpoint。Nezha 机器人资产、控制、其余奖励、随机化、终止判据，以及 Isaac Gym 物理后端和高度／接触采样仍沿用本项目实现。

## 训练、续训、play 和导出

本机训练环境为 `.venv`，MuJoCo 环境为 `.venv-mujoco`。入口自动配置仓库包路径、编译缓存和线程数，无需手动设置环境变量。默认使用 `cuda:0`、GPU pipeline、无窗口训练和 TensorBoard 记录。

从零训练只需指定最大轮次、并行环境数和种子（示例先训练 20000 轮）：

```bash
cd /home/bit/LZHMine
# 第一组：原始 CTS baseline
.venv/bin/python experiments/exp01_baseline/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10

# 第二组：Actor MoE
.venv/bin/python experiments/exp02_actor_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10
```

选择要运行的一组命令即可。速度范围是否扩大取决于跟踪奖励，而非预设轮次；后续可根据曲线和策略表现续训，训练预算上限仍为配置中的 100000 轮。如需训练窗口，可额外加 `--viewer`。启动信息会打印实验编号；第一组显示 `Actor=mlp | Actor experts=0`，第二组显示 `Actor=moe | Actor experts=8`。

在另一个终端启动 TensorBoard：

```bash
cd /home/bit/LZHMine
.venv/bin/tensorboard --logdir /home/bit/LZHMine/logs/nezha_moe_cts \
  --host 127.0.0.1 --port 6007
```

浏览器打开 <http://127.0.0.1:6007/#scalars>，在左侧勾选本次运行。远程训练时，在本地执行 `ssh -L 6007:127.0.0.1:6007 用户名@训练服务器`，再访问同一地址。若端口已占用，确认浏览器访问的是本项目的 TensorBoard，或换用空闲端口。

当前 Python 3.8 环境使用 TensorBoard 2.14，需要 `protobuf<5`；项目已在 `rsl_rl/setup.py` 添加对应的条件依赖。若旧环境出现 `MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`，执行 `uv pip install --python /home/bit/LZHMine/.venv/bin/python 'protobuf>=4.25.8,<5'`，然后重启 TensorBoard 并刷新浏览器。无需重新训练，已有事件日志可以继续使用。

TensorBoard 的上述父目录可以同时查看两组及旧日志。新训练默认目录分别为 `logs/nezha_moe_cts/exp01_baseline/<时间>_baseline/` 和 `logs/nezha_moe_cts/exp02_actor_moe/<时间>_actor_moe/`，终端会打印实际路径；如需自定义名称，可额外加 `--run_name`。默认无需 W&B 登录，增加 `--wandb` 才会同时上传 W&B，两组 W&B group 分别为 `exp01_baseline` 和 `exp02_actor_moe`。续训默认沿用 checkpoint 的运行名。

TensorBoard 在首轮采样前创建日志，每轮记录指标直到最后一轮。第一轮 loss 在首次更新完成后出现，回报／episode 长度在对应组有回合结束后出现；无需等待 1000 轮模型保存。首轮指标立即落盘，后续最多约 10 秒批量刷新，正常结束或 Ctrl+C／Python 异常退出时刷新并关闭日志。TensorBoard 服务可在训练前或训练后启动，都能读取已经记录的完整历史。`config.json` 和 checkpoint 同时保存环境配置，可确认实际使用的地形与课程。重点查看：

- `Train/teacher_reward`、`Train/student_reward`：教师／学生回报。
- `Train/teacher_length`、`Train/student_length`：episode 长度。
- `Loss/value`、`Loss/teacher_surrogate`、`Loss/student_surrogate`：PPO 损失。
- `Loss/mean_latent_loss`、`Loss/mean_load_balance_loss`：蒸馏与专家均衡损失。
- `Experts/gate_entropy`、`Experts/expert_0_usage` 至 `Experts/expert_7_usage`：学生编码器的门控熵与平均软权重。
- `ActorExperts/gate_entropy`、`ActorExperts/expert_0_usage` 至 `ActorExperts/expert_7_usage`：仅第二组记录 Actor 的门控熵与平均软权重，来自 PPO 更新批次。
- `ActorExperts/teacher_gate_entropy`、`ActorExperts/student_gate_entropy`，以及 `teacher_expert_0_usage…7`、`student_expert_0_usage…7`：分教师／学生输入统计 Actor 路由；总体指标按样本数量加权。熵上限为 `ln(8)≈2.079`，使用均衡或高熵本身不能证明专家分工有效。
- `Episode/max_command_x`、`Episode/max_command_yaw`：奖励达标后扩展的全局速度上限；实际采样还受各地形限速约束。

训练每 **1000 轮**保存一次 checkpoint，文件名为 `model_1000.pt`、`model_2000.pt` 等，保留各轮版本。训练正常结束时也会保存最后一轮，即使该轮不是 1000 的整数倍。训练需至少两个环境；当前 runner 面向单设备训练。

以下续训、play、导出和 MuJoCo 命令以第二组为例，直接填写实际路径。第一组使用 `experiments/exp01_baseline/` 中的同名入口和对应目录，完整命令也见各组 README。续训示例：

```bash
.venv/bin/python experiments/exp02_actor_moe/train.py \
  --max_iterations 80000 --num_envs 4096 --seed 10 \
  --resume \
  --load_run "/home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录名" \
  --checkpoint -1
```

续训的 `--max_iterations 80000` 表示**追加 80000 轮**，例如从第 20000 轮接着训练到累计 100000 轮；`--load_run` 填运行目录路径，`--checkpoint -1` 选择该目录中最新的 checkpoint，也可传 `1000` 等轮数。续训会创建新的日志目录，TensorBoard 可同时查看各次运行；导出续训结果时需填写新目录下的模型路径。

Checkpoint 保存网络与观测 schema、环境配置、控制参数、两个优化器和训练计数。续训先恢复观测配置再创建环境，可以用 `--num_envs` 调整并行数量；物理状态、历史和每个环境的地形等级重新初始化。新模式的已扩展速度范围保存于 `command_curriculum_state`，恢复后再采样首条指令；WF 零指令概率与奖励课程仍由恢复的训练步数推导。旧 WF checkpoint 保持原来的按轮次速度课程，旧地形 profile 的运行期指令课程状态不恢复。旧架构 checkpoint 会明确报错。

学生 play（直接指定第 1000 轮 checkpoint，默认一个环境）：

```bash
.venv/bin/python experiments/exp02_actor_moe/play.py \
  --checkpoint_path "/home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录名/model_1000.pt" \
  --play_num_envs 1
```

独立导出学生策略（不需要 Isaac Gym/GPU）。两组都放在 `deploy/deploy_mujoco/nezha/policy/` 下各自的实验子目录，目录不存在时自动创建。下面导出第二组第 1000 轮 checkpoint，并直接填写输出路径；省略 `--output` 也使用同一路径：

```bash
.venv/bin/python experiments/exp02_actor_moe/export.py \
  --checkpoint "/home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录名/model_1000.pt" \
  --output "/home/bit/LZHMine/deploy/deploy_mujoco/nezha/policy/exp02_actor_moe/policy.pt"

# 或导出该运行最新的 checkpoint
.venv/bin/python experiments/exp02_actor_moe/export.py \
  --run "/home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录名" \
  --output "/home/bit/LZHMine/deploy/deploy_mujoco/nezha/policy/exp02_actor_moe/policy.pt"
```

`model_<轮数>.pt` 是训练 checkpoint，MuJoCo 使用导出后的 `policy.pt`。导出的文件只包含学生 MoE、共享 Actor 及必要元数据，输入 [N, 228]、输出 [N, 16]。历史由 runtime 管理；教师、Critic 和优化器不进入部署模型。导出会检查 TorchScript 的动作及 Actor 门控与 checkpoint 一致，同时写出便于查看的 `policy.json`，其中 `actor_type`、`actor_expert_num` 标明 Actor 架构。

两组入口共用导出与 MuJoCo 实现，但默认策略文件分开保存。原 `get_expert_probabilities(history)` 及 MuJoCo 专家显示仍表示学生编码器；Actor 权重另通过导出模型的 `get_actor_expert_probabilities(history)` 查询，避免将两套专家混为一谈。

每次导出会覆盖该目录中的 `policy.pt` 和 `policy.json`；play 使用 `--no_export` 可跳过自动导出。如需保留多个策略，可通过 `--output` 在同一目录下指定不同文件名，例如 `policy_1000.pt`，仿真时在 `--policy` 中填写对应文件路径。

MuJoCo 可视化仿真（入口自动选择本组导出的策略，前进速度指令为 0.5 m/s）。两组复用原 YAML 的地形与控制设置，入口自动配置包路径，无需设置 `PYTHONPATH`：

```bash
.venv-mujoco/bin/python experiments/exp02_actor_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0
```

无窗口短时仿真：

```bash
.venv-mujoco/bin/python experiments/exp02_actor_moe/simulate.py \
  --headless --duration 30 --vx 0.5 --vy 0.0 --yaw 0.0
```

分组仿真入口默认使用本组 `policy.pt`，不需要再写 `--policy`；可显式指定同目录下另一个导出文件。直接调用原 `deploy_mujoco.py -c nezha_moe_cts.yaml` 仍读取旧 YAML 中的 `policy/policy.pt`，不会自动切换实验组。可视化窗口支持 `W/S` 调前后速度、`A/D` 调侧向速度、`Q/E` 调转向、空格停止、`R` 复位到当前测试道入口。`--headless` 模式必须设置正数 `--duration`（单位为仿真秒）。默认场景为 `scene_course.xml`。

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
.venv-mujoco/bin/python experiments/exp02_actor_moe/simulate.py \
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
| 实验组的 Actor 类型、日志目录和导出路径 | `experiments/exp01_baseline/config.py`、`experiments/exp02_actor_moe/config.py` |
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

Actor MoE 验证额外覆盖 8 个动作输出的软加权、单高斯采样和 log-prob 一致性、全部 Actor 专家与门控的 PPO 更新、蒸馏不修改 Actor、两类门控日志，以及旧 MLP checkpoint 的恢复训练／导出。实验分组验证覆盖可变配置隔离、本组最新模型查找、两类模型各自续训和导出，以及 Actor 类型或专家数量不匹配时在创建环境／写出策略前拒绝加载。

WF 验证还覆盖官方地形高度场 golden、类型比例／难度／边界／原点／种子复现／每类坡面修正阈值，以及旧速度阶段兼容、地形限速、累计指令、最大位移、零指令、奖励课程和 reset 时钟边界。Nezha 速度课程验证严格 80% 阈值、x/yaw 独立扩展、步长与上限、检查时机、采样后范围保持及 checkpoint 续训。

实际 Isaac Gym 短时物理回归（需训练环境，默认 16 个环境和小地形）：

```bash
TORCH_EXTENSIONS_DIR=/tmp/cts_torch_extensions python legged_gym/tests/test_env.py --headless --rl_device cpu --sim_device cpu --pipeline cpu --smoke_terrain trimesh
```

通用入口 `legged_gym/scripts/train.py` 和 `play.py` 也分别转入上述 CTS 训练与学生评估流程。
