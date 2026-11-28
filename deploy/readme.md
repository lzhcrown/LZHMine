# Nezha CTS-MoE MuJoCo 部署

八组实验通过各自的 `experiments/<组名>/simulate.py` 启动，共用 `deploy/deploy_mujoco/deploy_mujoco.py`。默认配置是 `configs/nezha_moe_cts.yaml`，runtime 是 `runtime/cts_policy_runtime.py`。机器人 MJCF、网格和地形位于 `nezha/`，完整结构对照表见根目录 [README](../README.md)。

先按根目录 README 训练；每 1000 轮生成 `model_<轮数>.pt`。以下以仅学生采用 MoE 的 `g1_student_moe` 为例，在仓库根目录执行；将 `实际运行目录名` 替换为对应训练目录，其他组替换组名：

```bash
.venv/bin/python experiments/g1_student_moe/export.py \
  --checkpoint "/home/bit/LZHMine/logs/nezha_moe_cts/g1_student_moe/实际运行目录名/model_1000.pt"

.venv-mujoco/bin/python experiments/g1_student_moe/simulate.py \
  --headless --duration 30 --vx 0.5 --vy 0.0 --yaw 0.0
```

分组独立导出和 play 自动导出默认写入 `deploy/deploy_mujoco/nezha/policy/<实验组>/policy.pt`，并在同目录生成 `policy.json`。目录不存在时自动创建。每次导出只覆盖对应目录的这两个文件。导出和仿真入口会校验模型结构。G13 v1/v2 支持回放同版本较早的地图学生检查点，训练时的 Critic 可以是旧的 8 专家 MoE 或新的两个任务价值网络。如需保留多个策略，通过 `--output` 指定不同文件名，并将仿真的 `--policy` 改为对应路径。

相对场景／策略路径以 `deploy/deploy_mujoco/` 为基准，也支持绝对路径。分组仿真入口不传 `--policy` 时读取本组的 `policy.pt`。公共导出／play 入口默认使用 `g1_student_moe`，直接调用 `deploy_mujoco.py -c nezha_moe_cts.yaml` 读取 YAML 指定的策略，当前同样为 `nezha/policy/g1_student_moe/policy.pt`。

导出和 MuJoCo 入口都会自动加入仓库内的 Python 包路径，无需设置 `PYTHONPATH`。可在仓库根目录直接执行 `.venv-mujoco/bin/python experiments/g1_student_moe/simulate.py` 打开仿真窗口。

学生使用 57 维本体与指令的 4 帧历史，共 228 维，输出 16 个动作。观测维数、尺度、裁剪、轮关节位置掩码和 term-major 历史顺序读取模型元数据。runtime 保留真实轮速。reset 清除旧历史，下一次传感器帧填满历史。

**G13 v1 和 v2 还需要当前地形扫描。** MuJoCo 从场景中取机器人周围的 17 × 11 个高程，按机身航向排列。实机若使用 `runtime.step(..., height_scan=...)`，每个控制步提供 187 个以米计的 `机身 z − 0.5 m − 地表 z`，沿 x 方向从 −0.8 至 +0.8 m 最先变化，沿 y 从 −0.5 至 +0.5 m 换行，间距 0.1 m。runtime 执行裁剪和缩放。直接调用 TorchScript 时，接口为 `policy(history, scaled_scan)`，第二个参数应已经完成 `clamp(-1, 1) × 2.5`，形状为 `[N,187]`。冻结编码器包含在导出文件内，学生从中取得 16 维几何特征。

学生和 Actor 各自可以是 MLP 或 8 专家 MoE。MuJoCo 日志中的 `expert_i` 权重表示学生编码器的门控概率。Actor 权重可通过导出模型的 `get_actor_expert_probabilities` 查询，参数与策略推理相同。每个专家的作用由训练学习。部署只运行学生编码器和 Actor。

策略每 0.02 s 更新一次，低层 PD 每个物理步计算：12 个腿关节用位置目标，4 个轮关节用速度目标。默认腿部尺度 0.25、轮速尺度 10。runtime 校验 checkpoint 导出的控制元数据与 YAML 一致，包括默认姿态、PD 增益、力矩限制、关节顺序与控制周期。

省略 `--headless` 使用交互窗口，`W/S` 调前后速度，`A/D` 调侧向速度，`Q/E` 调转向，空格停止，`R` 复位到当前测试道入口。数字键 `1`～`7` 切换测试道并清零命令，也可以在启动时传入 `--terrain ramp_25deg` 等名称。macOS viewer 需使用对应 MuJoCo 环境的 `mjpython`。

默认 `scene_course.xml` 有七条横向宽 5 m 的测试道，数字键依次对应：`stairs_15cm`、`stairs_20cm`、`stairs_25cm`、`ramp_15deg`、`ramp_25deg`、`ramp_30deg`、`rubble_speed_bumps`。前三组楼梯单级分别高 15／20／25 cm、踏面深 30 cm，各上／下 5 级，顶平台长 1 m。三组光滑坡道分别为 15°／25°／30°，上下坡水平长各 3.5 m、中间平台长 1 m。坡面及平台滑动摩擦系数为 0.35，普通地面和楼梯为 1.0。最后一条保留不平路面并增加 5 条宽 80 cm 的横向减速带，高度为 12、16、20、16、20 cm，路面最高 20 cm。出发区长 10 m，障碍区长 10 m，结束区长 5 m；默认出生在第一条入口，零速度时待命。

`tools/generate_terrain_course.py` 管理这些部署测试地形；运行 `.venv-mujoco/bin/python deploy/deploy_mujoco/tools/generate_terrain_course.py` 会同步生成 `course_geometry.xml`、高度场二进制和元数据。楼梯使用实体 box，坡道使用平整的凸楔形 mesh，坡道提高接触参数优先级以确保实际摩擦为 0.35；仅不平路面使用高度场。真实 MuJoCo 几何、接触及切换测试可运行 `.venv-mujoco/bin/python -m unittest discover -s deploy/deploy_mujoco/tests -v`。

`tools/generate_training_terrain.py` 从 `NezhaMoECTSCfg` 生成可选的完整训练地图 `scene_training.xml`，使用 G7 共用的 WF 高度配置。地图有 20 行难度、20 列地形，其中凹台和凸台各占 2 列。上下楼梯的相邻台沿间距均为 35 cm，在导出的 5 cm 水平栅格上对应 7 格。高度场元数据记录每格的实际难度、出生位置，以及原难度区间对应的 10 行参考跨度。MuJoCo 高度场在台沿处用一个栅格连接两级踏面，Gym 训练网格会将陡边修正为竖直面。
