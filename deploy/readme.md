# MuJoCo deployment

本目录按“部署入口、配置、运行时、机器人资产、辅助工具”分层，结构与
Dreamwaq_nezha 的 `deploy/deploy_mujoco` 保持一致，同时只保留 LZHMine
当前实际使用的 Nezha sim-to-sim 功能。

```text
deploy/
├── readme.md
└── deploy_mujoco/
    ├── deploy_mujoco.py          # 唯一 MuJoCo 启动入口
    ├── configs/
    │   └── nezha_mine.yaml       # 策略、场景与控制参数
    ├── runtime/
    │   └── policy_runtime.py     # 观测历史、推理与轮腿控制
    ├── nezha/
    │   ├── meshes/               # 部署自包含的机器人网格
    │   ├── mjcf/                 # 机器人与场景 MJCF、地形资产
    │   └── policy/
    │       └── policy.pt         # 当前发布到部署端的策略快照
    └── tools/
        ├── generate_terrain_course.py
        └── generate_training_terrain.py
```

配置文件中的相对路径均以 `deploy/deploy_mujoco/` 为基准。`logs/` 保存
训练运行、checkpoint/model 和原始导出记录；`nezha/policy/policy.pt` 只保存
当前选定的本地部署策略快照，且策略二进制不会上传 GitHub。

无窗口验证：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py \
  -c nezha_mine.yaml \
  --headless --duration 3 --vx 0.5
```

macOS 交互式 viewer：

```bash
.venv/bin/mjpython deploy/deploy_mujoco/deploy_mujoco.py \
  --duration 60
```

`W/S` 调前后速度，`A/D` 调侧向速度，`Q/E` 调转向，空格停止，`R`
复位。当前结构不包含录像、批处理、日志绘图、截图脚本及其输出目录；需要时
再按独立可选模块加入。
