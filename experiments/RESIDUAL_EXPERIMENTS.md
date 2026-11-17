# G7 all_moe 动作残差与时间平滑实验

本轮目标是判断：在现有最优 g7 上增加有界动作补偿，是否带来稳定、可重复的收益。采用残差 R × 时间平滑 T 的 2×2 对照。首轮继承相同 g7 检查点，联合微调原网络及新增残差；不冻结 Actor，也不同时改变门控正则。

## 1. 已配置的四组

| 组 | 简称 | 动作残差 R | 时间平滑系数 λ_T |
|---|---|---|---|
| g8_all_moe_control | B | 关闭 | 0 |
| g9_all_moe_residual | R | 开启 | 0 |
| g10_all_moe_temporal | T | 关闭 | 0.01 |
| g11_all_moe_residual_temporal | RT | 开启 | 0.01 |

四组沿用 g7：Student、Actor、Critic 各 8 个稠密专家和各自独立的门控，Teacher 仍为 MLP。g8 是“给予相同额外训练预算”的基线；旧 g7 的 20000 轮模型可作为起点参照，但不能独自承担残差收益的训练对照。

每组目录都提供 config.py、train.py、play.py、export.py、simulate.py。

## 2. 网络与损失

对 Actor 输入 x_t=[o_t,z_t]，先计算原来的专家加权均值：

    μ_pre,t = Σ_i g_i(x_t) μ_i(x_t)
    u_t = R_ψ([o_t, μ_pre,t])
    δ_t = b ⊙ tanh(u_t)
    μ_final,t = μ_pre,t + δ_t
    π_θ(a_t | x_t) = Normal(μ_final,t, diag(σ²))

o_t 是从历史观测中按 schema 正确提取的 57 维当前本体观测；μ_pre,t 为 16 维动作均值。残差 MLP 为 73→128→64→16，隐藏层 ELU，共新增 18,768 个可训练参数。最后一层权重和偏置都置零，因而初始化时 δ_t=0，四组策略函数一致。仅最后一层置零，不把所有隐藏层置零。

腿、轮动作在 16 维向量中交错排列，轮索引为 3、7、11、15。补偿上限：

    b = [0.10, 0.10, 0.10, 0.05] × 4

它处于环境动作缩放之前；在 action_scale=0.25、wheel_velocity_scale=10 的当前控制配置下，分别对应腿目标角度最多 ±0.025 rad、轮目标速度最多 ±0.5 rad/s。实际物理上限应以来源 checkpoint 的 control_config 为准；环境动作裁剪可能进一步限制最终命令。

补偿发生在高斯分布构造之前。采样、PPO log-prob、KL、rollout 中保存的均值、学生推理、动作历史、TorchScript 导出和 MuJoCo runtime 均使用最终动作路径。残差自身不额外 detach μ_pre；PPO 对学生编码器的原有 stop-gradient 保持，Teacher/Actor/Critic 和残差由 PPO 优化器训练，Student 由独立蒸馏优化器训练。

第一轮不改原有损失项：

    L_PPO,new = L_PPO,g7 + λ_T L_T
    L_student = mean((z_student - stopgrad(z_teacher))²)
                + λ_bal mean_i((mean_batch(g_student,i) - 1/8)²)

原 L_PPO,g7 包含 teacher/student clipped surrogate、价值损失和策略分布熵项。当前实现的显式负载均衡项只作用于 Student；Actor/Critic 的 gate_entropy 和 expert_usage 是日志指标，并未因本轮修改加入负载均衡或门控熵损失。Student 的 gate_entropy 也只是日志。策略高斯熵与门控熵是两个概念。

时间项使用当前参数下的相邻最终动作均值：

    L_T = (1 / (N_valid × 16)) Σ_(e,t) m_(e,t)
          ||μ_final,θ(s_(e,t)) - μ_final,θ(s_(e,t-1))||²

e 是环境编号；t 是控制步；m=1 仅当两条观测来自同一环境同一 episode 的相邻步，N_valid=Σm。两端都重新使用当前策略参数，保留梯度；teacher/student 各自使用对应编码分支。跨 rollout 边界连续的有效相邻对保留，终止或 timeout 后的新 episode 不与旧 episode 配对。此项不会把随机打乱后相邻的 minibatch 行当成时间相邻，也不拿旧采样动作作平滑目标。

λ_T=0.01 是本项目按“有效相邻对 × 动作维数”归一化后的初始试验值，不是对 Bio 论文超参数的复现。原环境中的动作变化惩罚等 reward 保持不变，因此 T 测试的是额外策略均值平滑项的增益与代价。

本实现迁移的是动作补偿思路，并未声称完整复现 Bio。没有添加 Critic 的动作补偿或价值残差，没有更换稀疏路由、共享门控、正交损失、router z-loss 或新的门控熵损失。若后续研究这些因素，应另设对照。

## 3. 公平初始化与训练

新增 --init_from 接受非残差 all-MoE 的训练 checkpoint。该模式：

- 恢复全部共同网络权重、观测 schema、源环境配置、共同 PPO/蒸馏参数及 rollout 长度，只保留目标组的 R/T 设置。
- 两个 Adam 均重新初始化；四组使用同样的源配置初始学习率。不是只给新增残差重建优化器，也不是继承某一组的旧 Adam 动量。
- 恢复训练轮数、总步数、环境公共步计数、已保存的命令课程范围；例如从 20000 继续计到 25000，不重走最初课程时钟。
- 新建仿真状态、逐环境历史和地形初始分布，不恢复源模型采样时的机器人位置或逐环境地形等级。这是统一起点的微调，不是逐位确定性的物理续训。
- 隔离新增残差权重初始化对随机流的影响，并按目标 seed 重设迁移后的采样随机流；避免首次 reset 中的状态依赖地形课程受到残差初始化干扰。
- config.json 和后续 checkpoint 记录来源路径、来源迭代、seed、优化器重建方式。
- 源 checkpoint 中失效的旧机器绝对资源路径可迁移到本仓库对应 resources 文件；已有有效文件保持，找不到对应文件会明确报错。

--resume 仍用于同组中断续训，恢复已有优化器与 R/T 配置；--from_scratch 显式开启从零训练。三种模式互斥；g8–g11 不指定模式会报错，防止不小心把从零训练和微调混在一起。普通 --resume 不应被当成从 g7 向残差组迁移的入口。

新组续训还检查保存的 experiment_key 和时间项开关，防止 g9 模型被误标成 g11，或 g8 模型被误标成 g10。相同组内续训可恢复已保存的非零时间系数，不强行覆盖回 0.01。

先在 GPU 环境进行四组各 1 轮、16 环境的运行链路检查，输出放在独立 smoke run。确认无 NaN、显存可承受、导出正确后再执行正式预算。

下面从同一现有 g7 检查点出发，每组额外训练 5000 轮，三个微调随机种子：

```bash
cd /home/crown/LZHMine
for study_seed in 10 20 30; do
  for study_group in g8_all_moe_control g9_all_moe_residual g10_all_moe_temporal g11_all_moe_residual_temporal; do
    .venv/bin/python "experiments/$study_group/train.py" \
      --init_from logs/g7_all_moe/model_20000.pt \
      --max_iterations 5000 --num_envs 4096 --seed "$study_seed" \
      --run_name "${study_group}_seed${study_seed}_finetune5k" \
      --rl_device cuda:0 --sim_device cuda:0 --no_wandb
  done
done
```

--max_iterations 是此次额外训练轮数，因此上述终点是 model_25000.pt。环境数、rollout 步数、训练轮数、reward、随机化、命令范围、学习率策略等均需在四组相同；GPU 显存不足时统一减少环境数，不只改 RT 一组。上述循环顺序执行，不会同时启动 12 个训练进程。移除 --no_wandb 可使用原有 WandB 配置；TensorBoard 始终保留。

同一 g7 模型加三个微调 seed 能检验“给定该预训练起点”的稳定性；不能代替独立预训练 seed。结论有希望后，至少在多个独立 g7 训练 seed 的 checkpoint 上重复完整 2×2 实验。

新日志位于 logs/nezha_moe_cts/<组>/<时间戳_run_name>/。不要在多个 seed 的目录上含糊地选择“最新”；评估和导出建议指定 checkpoint 绝对路径。正式比较使用固定共同终点或预先规定的验证集选模规则，不用测试集挑最优轮次。

## 4. 评估采样与命令

正式部署性能评估只使用 Student 路径。四组使用相同评估 seed、地形、速度、回合数、终止条件与仿真参数；训练 seed 与评估 seed 分开记录。先测已有局部地形协议，再另设训练分布外地形/速度、推扰、摩擦变化等鲁棒性评估。现有 CLI 并未新增这些鲁棒性条件的自动扫描，需后续单独配置。

示例：选择某个 seed 的明确 checkpoint，500 次/地形，20 个并行环境槽，每槽 25 回合；固定记录每槽编号 0、1、2 的回合明细，共最多 60 条回合/地形，包含成功和失败：

```bash
.venv/bin/python legged_gym/scripts/evaluate_nezha_moe_cts.py \
  --group g9_all_moe_residual \
  --checkpoint-path /绝对路径/对应seed的run/model_25000.pt \
  --seed 101 --speed 1.0 --envs-per-terrain 20 --episodes-per-terrain 500 \
  --detail-selection trial_ids --detail-trial-indices 0,1,2 \
  --record-residual --output-dir /tmp/residual_eval/g9_seed10_eval101
```

分别替换四个 group、各自 checkpoint 和 output-dir。建议使用多个固定评估 seed。若每组只有唯一匹配 run，可 --group residual --checkpoint 25000 一次评估四组；有多个 seed 时该模式会拒绝歧义，需逐个明确 --checkpoint-path。

--detail-trial-indices 指每个环境槽的 episode_index，固定编号不表示不同策略在闭环中经历完全相同的状态轨迹。失败回合若尚未进入测量区域，仍进入成功率统计，但可能没有区域内时序数据。

默认 first_successes 保留了旧评估行为；残差研究推荐 trial_ids，避免只分析成功回合。成功率分母始终包含全部协议尝试。过障时间与完成距离相关指标应同时报告成功率和统计条件；不能把提前摔倒的短时长当作快速通过。

增加 --expert-weights-only 可记录 Student/Actor 的逐步门控，允许与 --record-residual 同用；该模式不写常规功率等流。若同时需要完整性能指标与门控，用相同固定协议分别捕获，并以回合/步时钟匹配，勿假定随机运行天然逐位相同。

残差明细 residual.csv 包含各动作的 μ_pre、δ、μ_final、腿/轮补偿 RMS、接近补偿上限的占比和环境动作裁剪占比。训练日志新增 Residual/rms、Residual/abs_max、Residual/saturation_fraction；时间项记录 Loss/temporal 和 Loss/temporal_valid_pairs。

在输出的具体模型目录执行：

```bash
.venv/bin/python legged_gym/scripts/validate_evaluation_capture.py /实际捕获模型目录
```

它检查记录完整性、时钟、动作分解 μ_final=μ_pre+δ 等。目录位置见评估命令输出；--output-dir 下可能再有模型子目录。

同一训练好的 R/RT 模型另跑 --residual-scale 0、0.5、1，可观察移除/缩小补偿后的闭环变化。每个取值使用独立 output-dir。此开关只用于推理诊断：联合训练后的主体可能与残差协同，关掉残差不是独立训练的 g8/g10，不能替代 2×2 对照。

## 5. 必须完成的对照与判定

| 比较 | 回答的问题 |
|---|---|
| R − B，即 g9 − g8 | 不增加时间项时，残差是否有增益？首要比较。 |
| T − B，即 g10 − g8 | 单独平滑是否已能解释改进？ |
| RT − T，即 g11 − g10 | 已有平滑时，残差是否仍有额外价值？ |
| RT − R，即 g11 − g9 | 对残差策略，时间项是否抑制抖动、改善稳定性，或损害敏捷性？ |
| (RT − T) − (R − B) | 残差与时间项是否存在交互？ |
| 各组终点相对起始 g7 | 增加训练预算后的绝对改进；辅助比较，不单独归因残差。 |

每项使用相同 seed/协议成对比较。至少报告每个训练 seed 的结果、跨 seed 均值及离散度/区间，不把数百回合视为数百个独立训练模型。四组来自同一个 source checkpoint 时，应说明结果条件于该起点。单回合成功率区间和跨训练 seed 的不确定性分开报告。

评估指标至少覆盖：

1. 成功率、跌倒/超时比例、成功过障时间；逐地形和总体都报告。
2. 线速度/角速度跟踪误差、roll/pitch 和机体角速度/加速度；另测命令切换的响应延迟与超调，检查平滑是否牺牲敏捷性。
3. 同一测量窗口内能耗或 COT、扭矩与功率峰值；净/绝对/正功定义保持一致。
4. 动作一阶差分和二阶差分、腿/轮分别统计的补偿幅度及饱和率、最终动作裁剪率。
5. 训练稳定性、样本预算、策略推理耗时与部署控制周期是否满足要求。

当前捕获提供指标原始流，未新增自动计算所有指标、跨 seed 置信区间和显著性报告的完整统计脚本。数值汇总仍需沿用/扩展既有数据处理；应先固定计算定义再比较。

门控分析不能只看各地形平均权重。应补充单步熵、时序方差、转场附近的变化、专家输出差异及门控与残差幅度的对应关系。平均权重相似不足以证明门控失效，也不足以说明应加入更多正则。

建议在正式实验前规定实际可接受的收益/退化阈值，例如“主要困难地形成功率有可重复改善，常规地形、能耗与抖动不超过预设退化容忍度”。阈值取决于任务要求，不能只以训练 reward 或单个 seed 的最好结果决定是否引入。

- R>B 且 RT>T，并且收益跨 seed/地形稳定：支持引入残差。
- RT>B 但 RT≈T：改进可能主要来自时间项，尚无残差贡献证据。
- R 有收益但抖动增加，RT 保留收益并改善平滑：支持带时间约束的残差。
- 残差幅度长期接近零且性能无增益：当前输入/容量/上限下无需复杂化。
- 补偿或动作频繁饱和、能耗或控制周期明显恶化：先定位上限/归一化/优化问题，不急于扩大模块。

若第一轮有效，第二轮再分别做：参数量相近的 Actor 扩容对照；固定主体仅训练残差（必要时更新 Critic）的适配对照；不同残差上限/输入历史/时间系数的敏感性实验；多个独立 g7 预训练种子；独立门控损失或 Bio 网络内部残差对照。上述第二轮因素尚未作为新组自动实现，也不与首轮结果混算。

## 6. 工程验证范围

自动测试覆盖旧 g0–g7、残差边界和零初始化、学生停梯度、相邻时间配对/终止掩码、四组初始化/续训、checkpoint 与 TorchScript、部署观测构造以及捕获/校验。真实 g7 checkpoint 的 CPU 推理一致性也可核验。

仓库完整测试包含 Isaac Gym 模块，需在 torch 之前导入 Isaac Gym，并关闭宿主 ROS pytest 自动插件：

```bash
PYTHONPATH="$PWD:$PWD/rsl_rl" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -c 'from legged_gym.scripts._bootstrap import configure_runtime; configure_runtime(isaac_gym=True); import isaacgym; import pytest; raise SystemExit(pytest.main(["legged_gym/tests", "tests", "-q"]))'
```

代码测试和模拟向量环境更新不证明机器人性能改善；需要在可用 GPU 的 Isaac Gym 中完成上述训练与物理评估，再进行 MuJoCo/真实平台验证。

本次调整验证结果（2026-11-17）：训练/评估完整测试 410 项通过；Python 3.11 的 MuJoCo 部署相关测试 25 项通过。真实 logs/g7_all_moe/model_20000.pt 在四组的初始 Teacher/Student 动作均逐元素一致，两个 Adam 为空；四组均在 CPU 合成向量环境完成 20000→20001 的一轮更新、严格恢复和导出，导出及恢复动作误差均为零。四个导出文件也已在 Python 3.11 runtime 加载并校验最终动作分解。当前会话 CUDA 不可用，尚无正式训练或真实物理性能提升结论。

