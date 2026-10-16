# Nezha CTS-MoE MuJoCo 部署

使用 `deploy/deploy_mujoco/deploy_mujoco.py` 启动，默认配置是 `configs/nezha_moe_cts.yaml`，runtime 是 `runtime/cts_policy_runtime.py`。机器人 MJCF、网格和地形仍位于 `nezha/`。

先按根目录 [README](../README.md) 训练；每 1000 轮生成 `model_<轮数>.pt`。以下命令在仓库根目录执行，直接填写模型的绝对路径；将路径中的 `实际运行目录名` 替换为对应的训练运行目录名：

```bash
.venv/bin/python legged_gym/scripts/export_nezha_moe_cts.py \
  --checkpoint "/home/bit/LZHMine/logs/nezha_moe_cts/实际运行目录名/model_1000.pt" \
  --output "/home/bit/LZHMine/deploy/deploy_mujoco/nezha/policy/policy.pt"

.venv-mujoco/bin/python deploy/deploy_mujoco/deploy_mujoco.py \
  -c nezha_moe_cts.yaml \
  --headless --duration 30 --vx 0.5 --vy 0.0 --yaw 0.0
```

独立导出和 play 自动导出统一默认写入 `deploy/deploy_mujoco/nezha/policy/policy.pt`，并在同目录生成 `policy.json`；目录不存在时自动创建。独立导出可省略 `--output`，每次导出覆盖这两个文件。如需保留多个策略，通过 `--output` 在同目录指定不同文件名，并将仿真的 `--policy` 改为对应路径。

相对场景／策略路径以 `deploy/deploy_mujoco/` 为基准，也支持绝对路径。不传 `--policy` 时读取仓库下的 `deploy/deploy_mujoco/nezha/policy/policy.pt`，与默认导出位置一致；既有旧架构策略不会被当成新模型加载。

导出和 MuJoCo 入口都会自动加入仓库内的 Python 包路径，无需设置 `PYTHONPATH`。可在仓库根目录直接执行 `.venv-mujoco/bin/python deploy/deploy_mujoco/deploy_mujoco.py -c nezha_moe_cts.yaml` 启动仿真。

学生输入为 57 维本体观测的 4 帧历史，共 228 维，输出 16 个动作。观测维数、尺度、裁剪、轮关节位置掩码和 term-major 历史顺序直接读取模型元数据，与训练共用预处理模块；runtime 保留真实轮速。reset 清除旧历史，下一次传感器帧填满历史。

默认 Actor 为 8 专家 MoE。日志显示实际模型中的 `expert_i` 权重；这些编号没有预设运动模式含义。教师和 Critic 不参与部署。

策略每 0.02 s 更新一次，低层 PD 每个物理步计算：12 个腿关节用位置目标，4 个轮关节用速度目标。默认腿部尺度 0.25、轮速尺度 10。runtime 校验 checkpoint 导出的控制元数据与 YAML 一致，包括默认姿态、PD 增益、力矩限制、关节顺序与控制周期。

省略 `--headless` 使用交互窗口，`W/S` 调前后速度，`A/D` 调侧向速度，`Q/E` 调转向，空格停止，`R` 复位。macOS viewer 需使用对应 MuJoCo 环境的 `mjpython`。

`tools/generate_training_terrain.py` 从 `NezhaMoECTSCfg` 生成 WF 训练高度场、元数据，并同步 `scene_training.xml` 的地图尺寸和高度范围；`tools/generate_terrain_course.py` 管理独立的部署测试路线。默认 `scene_course.xml` 仍是三条测试路线。MuJoCo 的训练场景使用高度场，台阶边缘碰撞不等价于 Gym 中经坡面修正的三角网格。短时仿真只验证接线，运动表现需使用训练完成的模型评估。
