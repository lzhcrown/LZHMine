# g3_student_actor_moe：学生与 Actor MoE

本组属于学生编码器（S）、Actor（A）、Critic（C）的八组完整组合消融，开关为 **(S, A, C) = (1, 1, 0)**。教师编码器始终使用 MLP；所有组均保持 CTS 教师—学生联合训练。

| 模块 | 本组结构 | 输入／输出 |
|---|---|---|
| 教师编码器 | MLP | 295 → 32 |
| 学生编码器 | 8 专家 MoE | 228 → 32 |
| Actor | 8 专家 MoE | 89 → 16 |
| Critic | MLP | 327 → 1 |

学生 MoE 使用 8 个专家及独立 Softmax 门控，对专家特征软加权后进行 L2 归一化；蒸馏附加系数 0.01 的批次专家使用均衡损失。

所有 MoE 都使用共享专家特征主干、独立输出头和本模块自己的门控，采用全部专家的稠密软加权。不同模块不共享门控，专家编号没有预设对应关系。Actor 和 Critic 的 MoE 不额外添加均衡或门控熵辅助损失。Critic 只在训练时使用，不进入部署策略。

本组 [config.py](config.py) 只覆盖六个结构开关及日志元数据。观测、网络隐藏层宽度配置、奖励、地形课程、机器人动力学、PPO 和蒸馏超参数共用 [公共配置](../../legged_gym/envs/nezha/nezha_moe_cts_config.py)。基准组为 [g1_student_moe](../g1_student_moe/README.md)，本组不会改变公共超参数。不同结构的参数量可能不同，该组合实验不等同于参数量匹配实验。

## 训练与续训

以下命令在 `/home/bit/LZHMine` 执行。从零训练示例：

```bash
.venv/bin/python experiments/g3_student_actor_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10
```

日志保存到 `logs/nezha_moe_cts/g3_student_actor_moe/<实际运行目录>/`，每 1000 轮保存 checkpoint，正常结束时另存最后一轮。公平对照应使用相同的训练步数、并行环境数和配对种子，并从零开始。

```bash
.venv/bin/tensorboard --logdir /home/bit/LZHMine/logs/nezha_moe_cts \
  --host 127.0.0.1 --port 6007
```

续训示例，`--max_iterations 5000` 表示追加 5000 轮：

```bash
.venv/bin/python experiments/g3_student_actor_moe/train.py \
  --max_iterations 5000 --num_envs 4096 --seed 10 \
  --resume \
  --load_run /home/bit/LZHMine/logs/nezha_moe_cts/g3_student_actor_moe/实际运行目录 \
  --checkpoint 20000
```

本组入口在续训、play 和导出时检查学生、Actor、Critic 类型及各自专家数量。默认最新模型查找仅搜索本组日志；结构不同的 checkpoint 会被拒绝。

## 学生评估、导出与 MuJoCo

```bash
.venv/bin/python experiments/g3_student_actor_moe/play.py \
  --checkpoint_path /home/bit/LZHMine/logs/nezha_moe_cts/g3_student_actor_moe/实际运行目录/model_20000.pt \
  --play_num_envs 1 --command_x 0.5
```

play 默认导出学生编码器及共享 Actor；使用 `--no_export` 可跳过导出。独立导出：

```bash
.venv/bin/python experiments/g3_student_actor_moe/export.py \
  --checkpoint /home/bit/LZHMine/logs/nezha_moe_cts/g3_student_actor_moe/实际运行目录/model_20000.pt
```

输出默认位于 `deploy/deploy_mujoco/nezha/policy/g3_student_actor_moe/policy.pt`，输入为 `[N, 228]`、输出为 `[N, 16]`。教师与 Critic 不进入导出模型。

```bash
.venv-mujoco/bin/python experiments/g3_student_actor_moe/simulate.py \
  --vx 0.5 --vy 0.0 --yaw 0.0
```

无窗口运行可增加 `--headless --duration 30`；使用 `--terrain ramp_25deg` 选择起始测试道。窗口数字键 1～7 切换测试道。各组共用原有地形及控制设置，详细参数见 [仓库 README](../../README.md)。

