# Nezha CTS-MoE MuJoCo 部署

使用 `deploy/deploy_mujoco/deploy_mujoco.py` 启动，默认配置是 `configs/nezha_moe_cts.yaml`，runtime 是 `runtime/cts_policy_runtime.py`。机器人 MJCF、网格和地形仍位于 `nezha/`。

先用新训练入口产生 checkpoint，再执行：

```bash
PYTHONPATH="$PWD/rsl_rl:$PWD" .venv-train/bin/python \
  legged_gym/scripts/export_nezha_moe_cts.py \
  --checkpoint /absolute/path/model_200.pt --output /absolute/path/policy.pt

.venv-sim/bin/python deploy/deploy_mujoco/deploy_mujoco.py \
  -c nezha_moe_cts.yaml --policy /absolute/path/policy.pt \
  --headless --duration 3 --vx 0.5
```

相对场景／策略路径以 `deploy/deploy_mujoco/` 为基准，也支持绝对路径。不传 `--policy` 时读取 `nezha/policy/policy.pt`，该文件需要先替换为新导出的 CTS 模型；既有旧架构策略不会被当成新模型加载。

学生输入为 57 维本体观测的 4 帧历史，共 228 维，输出 16 个动作。观测维数、尺度、裁剪、轮关节位置掩码和 term-major 历史顺序直接读取模型元数据，与训练共用预处理模块；runtime 保留真实轮速。reset 清除旧历史，下一次传感器帧填满历史。

默认 Actor 为 8 专家 MoE。日志显示实际模型中的 `expert_i` 权重；这些编号没有预设运动模式含义。教师和 Critic 不参与部署。

策略每 0.02 s 更新一次，低层 PD 每个物理步计算：12 个腿关节用位置目标，4 个轮关节用速度目标。默认腿部尺度 0.25、轮速尺度 10。runtime 校验 checkpoint 导出的控制元数据与 YAML 一致，包括默认姿态、PD 增益、力矩限制、关节顺序与控制周期。

省略 `--headless` 使用交互窗口，`W/S` 调前后速度，`A/D` 调侧向速度，`Q/E` 调转向，空格停止，`R` 复位。macOS viewer 需使用对应 MuJoCo 环境的 `mjpython`。

`tools/generate_training_terrain.py` 从新的 `NezhaMoECTSCfg` 生成训练地形资产；`tools/generate_terrain_course.py` 管理部署测试路线。短时仿真只验证接线，运动表现需使用训练完成的模型评估。
