## 训练

```bash
cd /home/bit/LZHMine

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

```bash
cd /home/bit/LZHMine

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
cd /home/bit/LZHMine

# 在另一个终端选择一条命令启动，浏览器访问对应地址。

# 全部实验：http://127.0.0.1:6007/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts \
  --host 127.0.0.1 --port 6008

# G0：http://127.0.0.1:6100/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g0_mlp \
  --host 127.0.0.1 --port 6100

# G1：http://127.0.0.1:6101/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g1_student_moe \
  --host 127.0.0.1 --port 6101

# G2：http://127.0.0.1:6102/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g2_actor_moe \
  --host 127.0.0.1 --port 6102

# G3：http://127.0.0.1:6103/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g3_student_actor_moe \
  --host 127.0.0.1 --port 6103

# G4：http://127.0.0.1:6104/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g4_critic_moe \
  --host 127.0.0.1 --port 6104

# G5：http://127.0.0.1:6105/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g5_student_critic_moe \
  --host 127.0.0.1 --port 6105

# G6：http://127.0.0.1:6106/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g6_actor_critic_moe \
  --host 127.0.0.1 --port 6106

# G7：http://127.0.0.1:6107/#scalars
.venv/bin/tensorboard \
  --logdir /home/bit/LZHMine/logs/nezha_moe_cts/g7_all_moe \
  --host 127.0.0.1 --port 6107
```

## MuJoCo 仿真

```bash
cd /home/bit/LZHMine

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
