# 实验 2：Actor MoE 与门控

在实验 1 的基础上，仅将 Actor 改为 8 专家网络和独立 Softmax 门控。学生编码器仍为原有 8 专家 MoE，Critic 仍为 327 维输入的 MLP；环境、观测、奖励、地形课程和 PPO 设置与 baseline 共用。

本组差异配置集中在 [config.py](config.py)。Actor 的 89 维输入同时进入专家特征主干 `89 → 512 → 256 → 8×128` 和门控 `89 → 512 → 256 → 128 → 8`；8 个独立的 `128 → 16` 输出头产生动作均值，门控权重经过 Softmax 后加权求和。训练时在最终均值上按原有高斯策略采样，部署时使用均值。专家共享特征主干，没有预设动作模式，Actor 不添加额外均衡／熵损失。

以下命令均在 `/home/bit/LZHMine` 下执行，无需设置环境变量。

## 训练和曲线

```bash
.venv/bin/python experiments/exp02_actor_moe/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10
```

日志写入 `logs/nezha_moe_cts/exp02_actor_moe/<实际运行目录>/`；启动信息应为 `Actor=moe | Actor experts=8`。TensorBoard 从训练开始持续记录，每 1000 轮保存 `model_<轮次>.pt`，训练结束也保存 checkpoint。

```bash
.venv/bin/tensorboard --logdir /home/bit/LZHMine/logs/nezha_moe_cts \
  --host 127.0.0.1 --port 6007
```

打开 <http://127.0.0.1:6007/#scalars>，选择两组运行作比较。`Experts/*` 表示学生编码器，`ActorExperts/*` 表示 Actor 门控；两个模块的专家编号没有预设对应关系。若已有服务使用同一日志根目录，可以直接刷新页面。

续训时使用本组 checkpoint，例如从第 20000 轮继续训练 5000 轮（替换为实际路径）：

```bash
.venv/bin/python experiments/exp02_actor_moe/train.py \
  --resume \
  --load_run /home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录 \
  --checkpoint 20000 --max_iterations 5000 --num_envs 4096 --seed 10
```

分组前已训练的 8 专家 MoE Actor checkpoint 可通过绝对运行目录加载。本组会拒绝 MLP Actor checkpoint；不能通过续训将 baseline 自动转换为第二组。新的公平对照建议两组均从零训练，并保持轮次、环境数和种子一致。

## 导出、Isaac Gym 验证和 MuJoCo

```bash
.venv/bin/python experiments/exp02_actor_moe/export.py \
  --checkpoint /home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录/model_20000.pt
```

默认导出到 `deploy/deploy_mujoco/nezha/policy/exp02_actor_moe/policy.pt`，不会覆盖第一组或原来的顶层 `policy.pt`。也可不传 `--checkpoint`，自动选择本组最新运行中的最大轮次模型；导出前先校验 Actor 类型和专家数量。

如需先在 Isaac Gym 检查：

```bash
.venv/bin/python experiments/exp02_actor_moe/play.py \
  --checkpoint_path /home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录/model_20000.pt \
  --play_num_envs 1 --command_x 0.5
```

`play.py` 默认也导出到本组策略目录；增加 `--no_export` 可只验证。

```bash
.venv-mujoco/bin/python experiments/exp02_actor_moe/simulate.py
```

MuJoCo 默认加载本组策略和共用的 `nezha_moe_cts.yaml` 地形场景，不需要再写 `--policy`。可加 `--terrain ramp_25deg` 选择起始地形，窗口中的数字键 1～7 切换地形。模拟器其他参数与原部署入口一致。
