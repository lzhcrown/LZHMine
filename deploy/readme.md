# Nezha CTS-MoE MuJoCo 部署

两组实验分别通过 `experiments/exp01_baseline/simulate.py` 和 `experiments/exp02_actor_moe/simulate.py` 启动，共用 `deploy/deploy_mujoco/deploy_mujoco.py`。默认配置是 `configs/nezha_moe_cts.yaml`，runtime 是 `runtime/cts_policy_runtime.py`。机器人 MJCF、网格和地形仍位于 `nezha/`。

先按根目录 [README](../README.md) 训练；每 1000 轮生成 `model_<轮数>.pt`。以下命令以第二组为例，在仓库根目录执行，直接填写模型的绝对路径；将路径中的 `实际运行目录名` 替换为对应的训练运行目录名。第一组将命令和路径中的 `exp02_actor_moe` 替换为 `exp01_baseline`：

```bash
.venv/bin/python experiments/exp02_actor_moe/export.py \
  --checkpoint "/home/bit/LZHMine/logs/nezha_moe_cts/exp02_actor_moe/实际运行目录名/model_1000.pt"

.venv-mujoco/bin/python experiments/exp02_actor_moe/simulate.py \
  --headless --duration 30 --vx 0.5 --vy 0.0 --yaw 0.0
```

分组独立导出和 play 自动导出默认写入 `deploy/deploy_mujoco/nezha/policy/<实验组>/policy.pt`，并在同目录生成 `policy.json`；目录不存在时自动创建。每次导出只覆盖对应目录的这两个文件，分组入口会校验 checkpoint 的 Actor 类型和专家数量。如需保留多个策略，通过 `--output` 在同目录指定不同文件名，并将仿真的 `--policy` 改为对应路径。

相对场景／策略路径以 `deploy/deploy_mujoco/` 为基准，也支持绝对路径。分组仿真入口不传 `--policy` 时读取本组的 `policy.pt`。原来的公共导出／play 入口仍默认导出到 `deploy/deploy_mujoco/nezha/policy/policy.pt`，直接调用原 `deploy_mujoco.py -c nezha_moe_cts.yaml` 仍读取 YAML 指定的这个顶层策略路径。

导出和 MuJoCo 入口都会自动加入仓库内的 Python 包路径，无需设置 `PYTHONPATH`。可在仓库根目录直接执行 `.venv-mujoco/bin/python experiments/exp02_actor_moe/simulate.py` 打开第二组的仿真窗口。

学生输入为 57 维本体观测的 4 帧历史，共 228 维，输出 16 个动作。观测维数、尺度、裁剪、轮关节位置掩码和 term-major 历史顺序直接读取模型元数据，与训练共用预处理模块；runtime 保留真实轮速。reset 清除旧历史，下一次传感器帧填满历史。

第一组 Actor 为原始 MLP，第二组 Actor 为 8 专家 MoE；两组学生编码器均为 8 专家 MoE。当前 MuJoCo 日志和窗口中的 `expert_i` 权重表示学生编码器，Actor 权重可通过导出模型的 `get_actor_expert_probabilities(history)` 单独查询。这两套专家的编号没有预设运动模式或对应关系。教师和 Critic 不参与部署。

策略每 0.02 s 更新一次，低层 PD 每个物理步计算：12 个腿关节用位置目标，4 个轮关节用速度目标。默认腿部尺度 0.25、轮速尺度 10。runtime 校验 checkpoint 导出的控制元数据与 YAML 一致，包括默认姿态、PD 增益、力矩限制、关节顺序与控制周期。

省略 `--headless` 使用交互窗口，`W/S` 调前后速度，`A/D` 调侧向速度，`Q/E` 调转向，空格停止，`R` 复位到当前测试道入口。数字键 `1`～`7` 切换测试道并清零命令，也可以在启动时传入 `--terrain ramp_25deg` 等名称。macOS viewer 需使用对应 MuJoCo 环境的 `mjpython`。

默认 `scene_course.xml` 有七条横向宽 5 m 的测试道，数字键依次对应：`stairs_15cm`、`stairs_20cm`、`stairs_25cm`、`ramp_15deg`、`ramp_25deg`、`ramp_30deg`、`rubble_speed_bumps`。前三组楼梯单级分别高 15／20／25 cm、踏面深 30 cm，各上／下 5 级，顶平台长 1 m。三组光滑坡道分别为 15°／25°／30°，上下坡水平长各 3.5 m、中间平台长 1 m。坡面及平台滑动摩擦系数为 0.35，普通地面和楼梯为 1.0。最后一条保留不平路面并增加 5 条宽 80 cm 的横向减速带，高度为 12、16、20、16、20 cm，路面最高 20 cm。出发区长 10 m，障碍区长 10 m，结束区长 5 m；默认出生在第一条入口，零速度时待命。

`tools/generate_terrain_course.py` 管理这些部署测试地形；运行 `.venv-mujoco/bin/python deploy/deploy_mujoco/tools/generate_terrain_course.py` 会同步生成 `course_geometry.xml`、高度场二进制和元数据。楼梯使用实体 box，坡道使用平整的凸楔形 mesh，坡道提高接触参数优先级以确保实际摩擦为 0.35；仅不平路面使用高度场。真实 MuJoCo 几何、接触及切换测试可运行 `.venv-mujoco/bin/python -m unittest discover -s deploy/deploy_mujoco/tests -v`。

`tools/generate_training_terrain.py` 独立从 `NezhaMoECTSCfg` 生成 WF 训练高度场、元数据，并同步可选 `scene_training.xml` 的地图尺寸和高度范围。该完整训练地图仍使用高度场，台阶边缘碰撞不等价于 Gym 中经坡面修正的三角网格。短时仿真只验证接线，运动表现需使用训练完成的模型评估。
