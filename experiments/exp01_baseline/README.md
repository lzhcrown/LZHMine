# 实验 1：原始 CTS baseline

保留原设计：教师 MLP 编码器、学生 8 专家 MoE 编码器、MLP Actor 和 MLP Critic。Actor 输入为 89 维，隐藏层为 `[512, 256, 128]`，输出 16 维动作均值。Critic 输入为 327 维。

本组差异配置集中在 [config.py](config.py)。两组共用环境、观测、奖励、地形课程、PPO 和部署实现，避免复制实现后产生差异；第二组只改变 Actor 结构。旧的 `legged_gym/scripts/train_nezha_moe_cts.py` 默认也进入本组。

以下命令均在 `/home/bit/LZHMine` 下执行，无需设置环境变量。

## 训练和曲线

```bash
.venv/bin/python experiments/exp01_baseline/train.py \
  --max_iterations 20000 --num_envs 4096 --seed 10
```

日志写入 `logs/nezha_moe_cts/exp01_baseline/<实际运行目录>/`；启动信息应为 `Actor=mlp | Actor experts=0`。TensorBoard 从训练开始持续记录，每 1000 轮保存 `model_<轮次>.pt`，训练结束也保存 checkpoint。

```bash
.venv/bin/tensorboard --logdir /home/bit/LZHMine/logs/nezha_moe_cts \
  --host 127.0.0.1 --port 6007
```

打开 <http://127.0.0.1:6007/#scalars>，选择 `exp01_baseline` 和 `exp02_actor_moe` 下的运行作比较；若已有服务使用同一日志根目录，可以直接刷新页面。

续训时使用本组 checkpoint，例如从第 20000 轮继续训练 5000 轮（替换为实际路径）：

```bash
.venv/bin/python experiments/exp01_baseline/train.py \
  --resume \
  --load_run /home/bit/LZHMine/logs/nezha_moe_cts/exp01_baseline/实际运行目录 \
  --checkpoint 20000 --max_iterations 5000 --num_envs 4096 --seed 10
```

原有 MLP Actor 的旧 checkpoint 可通过绝对运行目录加载；续训沿用其保存的环境和训练设置，新的公平对照建议两组均从零训练。本组会拒绝 MoE Actor checkpoint，避免将第二组误加载为 baseline。

## 导出、Isaac Gym 验证和 MuJoCo

```bash
.venv/bin/python experiments/exp01_baseline/export.py \
  --checkpoint /home/bit/LZHMine/logs/nezha_moe_cts/exp01_baseline/实际运行目录/model_20000.pt
```

默认导出到 `deploy/deploy_mujoco/nezha/policy/exp01_baseline/policy.pt`，不会覆盖第二组或原来的顶层 `policy.pt`。也可不传 `--checkpoint`，自动选择本组最新运行中的最大轮次模型；导出前先校验 Actor 类型。

如需先在 Isaac Gym 检查：

```bash
.venv/bin/python experiments/exp01_baseline/play.py \
  --checkpoint_path /home/bit/LZHMine/logs/nezha_moe_cts/exp01_baseline/实际运行目录/model_20000.pt \
  --play_num_envs 1 --command_x 0.5
```

`play.py` 默认也导出到本组策略目录；增加 `--no_export` 可只验证。

```bash
.venv-mujoco/bin/python experiments/exp01_baseline/simulate.py
```

MuJoCo 默认加载本组策略和共用的 `nezha_moe_cts.yaml` 地形场景，不需要再写 `--policy`。可加 `--terrain ramp_25deg` 选择起始地形，窗口中的数字键 1～7 切换地形。模拟器其他参数与原部署入口一致。
