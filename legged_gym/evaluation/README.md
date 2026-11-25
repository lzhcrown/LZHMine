# Isaac Gym 八组 CTS 学生策略原始数据采集

本目录使用 CTS 学生策略、Nezha 控制器和固定评估地形。楼梯与高台评估从接近障碍到在另一侧平稳落地的完整过程。平地、粗糙地面和斜坡沿用 `wheel_gym_CQ` 的四轮投影穿越判据。

## 观看：不采集数据

在带图形桌面的终端进入仓库，使用训练环境 `.venv`：

```bash
cd /home/crown/LZHMine

# 八组依次观看同一块 30 cm 凸形楼梯，每组尝试 3 次
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group all --checkpoint 20000 --viewer --no-record \
  --terrain-types stairs_convex_30cm --envs-per-terrain 1 \
  --episodes-per-env 3 --speed 1.0 --max-seconds 30

# 只看 G1 的 45 cm 高台
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group g1_student_moe --checkpoint 20000 --viewer --no-record \
  --terrain-types platform_45cm --episodes-per-env 3 --speed 1.0

# G12 分别观看单阶高 35 cm 的凸形楼梯和顶面 5×5 m、高 50 cm 的方台
for terrain in stairs_convex_35cm platform_50cm; do
  .venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
    --group g12_high_platform --checkpoint 20000 --viewer --no-record \
    --terrain-types "$terrain" --speed 0.8 --episodes-per-env 1 || break
done

# G12 v3 同时观看六种高台，每种一个机器人
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group g12_v3 --checkpoint -1 --viewer --no-record --speed 0.8 \
  --terrain-types platform_85cm,platform_90cm,platform_95cm,platform_100cm,platform_105cm,platform_110cm

# 单组同时展示十种地形，每种一个机器人
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group g0_mlp --checkpoint 20000 --viewer --no-record --episodes-per-env 3
```

`--viewer --no-record` 打开窗口，终端逐轮显示结果。每种地形默认有一个机器人。每轮从出生开始，成功、跌倒或用完 30 仿真秒就结束。单个模型默认持续观看，每个机器人结束后从自己的起点开始下一轮。**N/B** 切换机器人，**空格**暂停或继续，**Esc、关闭窗口或 Ctrl+C** 退出。暂停期间仿真时钟停止。

指定 `--episodes-per-env 3` 可让每个机器人各跑三轮后结束，也可用 `--episodes-per-terrain` 分配每种地形的总次数。多模型预览（`--group all` 或 `residual`）默认每组每个机器人跑一轮，再轮到下一组，所有组轮转播放。显式指定次数时，各组完成配额后整条命令退出。无窗口的 `--no-record` 默认每个机器人跑一轮。预览不写 CSV。

如果要依次完整观看每种地形上的八组模型，可在 Bash 中运行下面的循环：一次只有一个机器人、一种地形、一组模型，完成后进入下一组，再进入下一种地形。Esc 或 Ctrl+C 中止循环。

```bash
for terrain in flat stairs_concave_{35,25,30}cm stairs_convex_{35,25,30}cm platform_45cm rough slope; do
  .venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
    --group all --checkpoint 20000 --viewer --no-record \
    --terrain-types "$terrain" --envs-per-terrain 1 --episodes-per-env 1 \
    --speed 1.0 --max-seconds 30 || break
done
```

组名可用 `g0_mlp`、`g1_student_moe`、`g2_actor_moe`、`g3_student_actor_moe`、`g4_critic_moe`、`g5_student_critic_moe`、`g6_actor_critic_moe`、`g7_all_moe`，或 `all`。原版 G12 使用 `--group g12_high_platform`。对照 v1、v2、v3 时分别使用 `--group g12_v1`、`--group g12_v2`、`--group g12_v3`。只看一个高台时指定单个名称，例如 `--terrain-types platform_110cm`。

## 采集：观看确认后手动执行

```bash
# 检查模型路径和新协议，不启动仿真
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py --dry-run

# 八组×十种地形，每组每地形 500 次，总共 40000 次完整尝试
# 每种地形 20 个并行槽，每槽自动重复 25 轮
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group all --checkpoint 20000 --envs-per-terrain 20 --episodes-per-terrain 500 \
  --metric-successes-per-terrain 10 \
  --spawn-xy-jitter 0.02 --speed 0.8 --seed 10 --max-seconds 30 \
  --output-dir evaluations/cts_20000_protocol14_500_success10_seed10

# 重新校验某组 CSV
.venv/bin/python legged_gym/scripts/validate_evaluation_capture.py \
  evaluations/cts_20000_protocol14_500_success10_seed10/g1_student_moe
```

去掉 `--no-record` 即为采集模式。默认无窗口，每组每地形 500 次、20 个并行槽。`--episodes-per-terrain` 指定每种地形的总次数，须能被 `--envs-per-terrain` 整除。例如 500 次分给 10 个槽，每槽跑 50 轮。也可用 `--episodes-per-env` 指定每槽次数。输出目录需为新目录。

正式命令速度设为 0.8 m/s、每次最多 30 s，出生位置在 X/Y 方向各加 ±2 cm 的可复现随机偏移；八组使用相同种子与偏移序列。若要复用固定出生条件，将 `--spawn-xy-jitter 0.02` 改为 `0`。重复次数不改变地形形状，不代表重新生成 500 种地形。

每组模型仍运行十种地形各 500 次（共 5000 次），每轮保存尝试结果用于成功率。详细指标仅保留每组、每种地形按完成顺序的前 10 次成功，同一个控制步完成时按环境编号排序；`--metric-successes-per-terrain` 可修改该配额，设为 0 时仅保存结果。某地形不足 10 次成功时保存实际成功次数，不额外增加尝试。成功但计量窗口为零的轮次仍占一个成功名额并保留零样本标记，不补选或把缺失指标填零。

结果尚未确定时，以紧凑数值数组在内存暂存候选窗口。失败、超时、中断和超额成功不写详细轨迹；某地形达到配额后立即释放该地形的其余缓存。每种地形完整完成 500 次后，主成功率为 `success / 500`，包含进区前失败/超时；中断不算完成次数。新模式须重新启动进程并使用新目录；旧模式已写出的 CSV 不会自动转换或删除。

默认读取 `logs/<组名>/model_20000.pt`，也支持 `logs/nezha_moe_cts/<组名>/<运行目录>/model_20000.pt`。`--log-root` 可指定其他根目录；`--checkpoint -1` 选择各组最大数字轮次。相同轮次存在多个运行时拒绝自动选择，需指定 `--checkpoint-path`。加载时检查学生、Actor、Critic 的结构与专家数量，恢复 checkpoint 的观测 schema、环境及控制参数。另一台电脑的机器人资源路径会映射到本仓库的同名 `resources/` 路径，并记录资源 SHA256。

输出目录必须不存在，避免覆盖之前数据。八组依次在独立进程运行，使用相同评估种子、地形、速度和尝试次数；单组出错即停止批处理，保留已有文件和状态。缺少任何一组模型时在启动前报错。所有 CSV 自动逐行校验，结果写入 `data_validation.json`。

## 固定协议

### G7 Student / Actor 专家权重

仅采集 G7 两个部署模块的专家权重，可运行：

```bash
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group g7_all_moe --checkpoint 20000 --expert-weights-only \
  --envs-per-terrain 20 --episodes-per-terrain 500 \
  --metric-successes-per-terrain 10 \
  --spawn-xy-jitter 0.02 --speed 0.8 --seed 10 --max-seconds 30 \
  --output-dir evaluations/g7_student_actor_experts_20000_seed10
```

每组地形仍运行 500 次，仅保留其前 10 次成功的完整计量窗口；不足 10 次则保留实际次数。每种地形分别写入 `experts/<地形>/student_weights.csv`、`experts/<地形>/actor_weights.csv`。G7 每个模块各有 8 个专家，对应 `expert_00_weight`～`expert_07_weight`，每行权重之和约为 1。两个模块的专家编号各自独立。

权重来自生成实际动作的同一次前向计算，时间戳是控制区间起点的观测时刻；每行同时记录区间起止时间、`control_step`、`trial_id`、模型与地形。首次检测到进入测试区的区间仍排除，之后包含终止控制区间。此模式不计算 Critic，也不重复输出功率、COT、速度跟踪等详细采样文件；保留所有轮次结果、索引、通过时间和成功率。采集结束自动校验概率和、地形标注、时钟、样本数和成功轮次筛选。`--expert-weights-only` 仅允许 G7，不能与 `--no-record` 同用。

默认的十种地形含 45 cm 高台。其余高度可通过 `--terrain-types` 显式选择。楼梯名称中的 `concave` 表示先下后上，`convex` 表示先上后下。

| 地形 | `--terrain-types` 名称 |
|---|---|
| 平地 | `flat` |
| 凹形楼梯，单阶 35/25/30 cm | `stairs_concave_35cm` / `stairs_concave_25cm` / `stairs_concave_30cm` |
| 凸形楼梯，单阶 35/25/30 cm | `stairs_convex_35cm` / `stairs_convex_25cm` / `stairs_convex_30cm` |
| 高台，顶面 5×5 m、高 45 cm | `platform_45cm` |
| 可选高台，高 50、55、60、65 cm | `platform_50cm` / `platform_55cm` / `platform_60cm` / `platform_65cm` |
| 可选高台，高 70、75、80 cm | `platform_70cm` / `platform_75cm` / `platform_80cm` |
| 可选高台，高 85、90、95 cm | `platform_85cm` / `platform_90cm` / `platform_95cm` |
| 可选高台，高 100、105、110 cm | `platform_100cm` / `platform_105cm` / `platform_110cm` |
| 粗糙地面 | `rough` |
| 20° 斜坡，水平长 6.5 m、宽 5.6 m | `slope` |

- 除 `slope` 外，测试区为 **7.8×7.8 m**，前后各 1.5 m 支撑平台。阶梯每侧 8 级、踏面 0.3 m、中央平台 3 m；35/25/30 cm 是单级高度，对应中央平台总高度或深度 2.8/2.0/2.4 m。粗糙度幅值参数 0.04 m。
- `slope` 为 **20°、水平长 6.5 m、实际宽 5.6 m** 的斜坡，测试区边界也为 6.5×5.6 m。前后平台各 1.5×5.6 m，含平台的评估走廊总水平长度 9.5 m；高端平台按 5 mm 高度网格量化后为 2.365 m，并向走廊外延伸同高支撑。两侧不再延伸坡面，外侧为零高度地面，侧缘存在落差。坡宽对齐 2 cm 网格，中心最多偏移半个网格；机器人出生和目标位置使用实际坡面中心线。通过判据仍允许四轮清空同一侧向边界，与原协议一致。
- 所有高度的高台均居中放在 7.8×7.8 m 测试区内，台面为 5×5 m，四周是竖直台阶。区内前后各留 1.4 m 平地。
- 默认机体坐标指令为 `[vx, vy, wz]=[1,0,0]`。每轮从出生起最多运行 **30 仿真秒**，包括接近、穿越和落地。可用 `--max-seconds` 缩短每轮时限。负 `--speed` 支持保持机体 yaw=0 的倒车，目标岸也随行进方向反转。
- 关闭观测噪声、课程、推扰、关节初始随机化、动作延迟及力矩随机缩放；训练的默认随机力矩缩放仍保留。
- 第一只轮的碰撞圆柱 XY 投影进入测试区后开始计量，首次检测到进入的那个控制区间不计入。此后一直采样到成功、跌倒或超时，保留后退、再进入及前后轴之间的间隔。
- **高台和楼梯**先验证四轮抵达障碍中央面。凸台和凸形楼梯取最高面，凹形楼梯取谷底平面。随后四轮要沿指令方向越过最后一道台沿，落在对岸平地。接触阶段要求四个轮底距目标平面均不超过 8 cm，至少三个轮子的向上接触力各大于 1 N。
- **平稳落地**还要求机身倾角约 37° 以内，机身在世界坐标系中的垂向速度绝对值不超过 0.3 m/s，滚转和俯仰角速度的合量不超过 1 rad/s。满足落地条件连续 0.20 s 时，本轮记为 `success`（`stable_landing`）。机器人可以继续水平向前行驶。绕过障碍、仅登顶或在空中飞过均不能完成这一过程。
- **平地、粗糙地面和斜坡**沿用几何穿越规则。所有轮子都进入过测试区，再完整越过同一条对侧或侧向边界即可通过，具体边界记录在 `metadata_json.exit_edge`。
- 机体相对竖直倾斜超过 60° 记作跌倒；跌倒优先于同时通过，通过优先于同时超时。接触、后退或暂时离开走廊不单独判失败。NaN/Inf 中止采集，记为中断而非跌倒。
- 边界事件精度为一个控制步。当前模型控制频率约 50 Hz，物理频率约 200 Hz；实际 dt 以 metadata 和 CSV 为准。

无出生扰动时重复槽位的初始条件相同，不能将重复结果当作独立环境样本。正式比较可配对使用多个种子及出生扰动，并另行规划不同速度、坡度及地形难度。每地形单次通过只表示该次尝试成功，不能据此声称稳定通过率为 100%。

## 文件和计算输入

每组目录生成：

| 文件 | 内容及用途 |
|---|---|
| `success_rate.csv` | 全部尝试按地形和总体汇总，含成功/失败/超时计数、成功率和已选成功轮次数 |
| `traversal_time.csv` | 每次尝试的时间记录；所有成功轮次都提供地形内通过耗时和从出生到通过的总耗时，不受前 10 次详细轨迹配额限制；失败/超时/中断的通过耗时留空 |
| `power.csv` | 所选成功轮次：每物理子步 dt、16 个关节实际力矩及步前/步后角速度，用于机械功率分析 |
| `cot.csv` | 所选成功轮次：每控制区间的净功/绝对功/正功梯形积分增量、整机质量、重力、当前/前一位置及目标方向，可独立重建能量、实际距离和 COT |
| `tracking.csv` | 所选成功轮次：每控制步的机体线/角速度、速度指令和 dt，用于跟踪误差分析 |
| `stability.csv` | 所选成功轮次：每控制步世界速度、加速度、姿态和角速度，用于稳定性分析 |
| `episodes.csv` | 每次尝试的地形、状态、原因、计量窗口、进入/退出时刻、总质量、重力、样本数和起终状态；包含尚未进入地形便失败/超时的尝试 |
| `trial_index.csv` | 每轮一行的筛选索引：唯一轮次 ID、模型、地形、结果及原因、计量时长、样本数、出生种子、指令速度、出生/目标坐标及离开边界 |
| `summary.csv` | 按地形和总体汇总事件计数，分别保留已进入/未进入尝试及中断数 |
| `metadata.json` | 模型/资源/源代码 SHA256、环境配置、地形参数、关节名称映射、dt、协议及运行状态 |
| `data_validation.json` | CSV 连续性、子步与控制步对应、终止状态、样本数、指令和有限值校验结果 |

批次根目录还有 `batch.json` 和包含八组计数的 `event_counts.csv`。每次单组采集生成唯一 `run_id`，同时写入 `metadata.json`。轮次表和四类指标 CSV 每行都有 `run_id`、`trial_id`，ID 格式为 `<run_id>/<模型组>/<地形>/env0000/ep0000`；环境编号与环境内轮次从 0 开始。终止结果以 `episodes.csv` / `trial_index.csv` 为准。新模式不再输出 `physics.csv`、`trajectory.csv`，也不在每条采样行重复写大段 metadata JSON；历史全量文件仍可用校验器检查。`j00`～`j15` 与 `metadata.json.joint_names` 顺序一致，含四个轮关节。

后续筛选时，在 `trial_index.csv` 选择 `metrics_selected=True` 的轮次，再按 `trial_id` 提取各指标文件的全部行。`success_rank` 是该地形按完成顺序的成功序号；`recorded_control_samples` / `recorded_physics_samples` 为实际保留样本数，原 `control_sample_count` / `physics_sample_count` 为计量窗口内步数。所有尝试均有索引，只有选中的成功轮次有详细指标。进一步选择最佳单次时保留所选 ID 和筛选依据，成功率仍使用全部 500 次完整尝试。

通过耗时使用仿真时钟。`traversal_time_s = exit_time_s - entry_time_s` 从第一只轮进入测试区开始，楼梯和高台计至对岸稳定落地，其他地形计至四轮通过。`total_time_to_success_s = trial_elapsed_s` 还包含出生后的接近阶段。`time_resolution_s` 为控制步分辨率，当前模型约为 0.02 s。所有成功轮次均记录耗时，包括 `metrics_selected=False` 的轮次。跌倒、超时和中断的轮次只保留已运行时间。

比较历史成功率或通过耗时前，先查看 `metadata.json` 中的 `protocol.version`。本规则自版本 14 起生效。版本 13 的楼梯和高台成功依据四轮离开测试区，当时允许侧向退出。历史数据仍按各自采集时的判据解释。

已完成的旧采集可直接从 `episodes.csv` 补算，无须重跑仿真：

```bash
.venv/bin/python legged_gym/scripts/backfill_traversal_times.py \
  evaluations/cts_20000_terrain_v12_500_success10_seed10
```

每组新增 `traversal_time.csv`，批次根目录新增 `traversal_times_all.csv` 和带源文件 SHA256 的 `traversal_time_backfill.json`。原始 CSV 不修改；补算工具拒绝覆盖已存在的输出。后续新采集自动写出每组的 `traversal_time.csv`。

除成功率、通过耗时及 COT 能量增量外，最终指标仍在后处理计算：

- 每关节机械功率 `p_j=τ_j·ω_j`。每子步有前后两端角速度；可用梯形积分分别计算净功、绝对功和正功，三者不可混用。这里没有电流、电压、驱动效率数据，因此不是电功率或电池能耗。
- `cot.csv` 的 `energy_net_trap_j`、`energy_abs_trap_j`、`energy_positive_trap_j` 是该控制区间内各物理子步、各关节的梯形积分之和：分别对端点的 `p`、`abs(p)`、`max(p,0)` 积分。这些是区间增量而非累计值；保留 `power.csv` 可重新选择其他积分方法。校验器核对能量增量与原始力矩/角速度是否一致。
- `COT=E/(m·g·d)`，其中质量是整机所有刚体质量之和，距离由同一计量窗口的实际位置计算；可分别报告平面路径长度及沿目标方向净位移。距离为零时应标记不可定义，不应代入固定地形长度。
- 速度跟踪可对机体坐标 `vx/vy/wz` 与同坐标指令计算 MAE/RMSE；角速度单位为 rad/s。所选成功轮次保留完整计量窗口，不再额外裁剪热身区间。
- 通过率建议同时报告 `success/(success+failure+timeout)`（所有完整尝试）和 `started_success/started_tests`（已进入测试区的完整尝试），中断不并入失败，进区前失败另外呈现。计数必须按地形及模型分开。

测试地形、穿越、采集和生命周期测试来自参考实现并适配当前 CTS；正常训练不启用记录器。生成数据位于已忽略的 `evaluations/` 下。
