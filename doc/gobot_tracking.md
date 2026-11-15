# BeyondMimic 动作跟踪

`g1-tracking` 使用 PPO 训练一个策略跟踪一段机器人动作。代码模块为
`gobot.rl.tracking`，场景由 Gobot `.jscn` 提供，物理由 MuJoCo CPU 或 MuJoCo Warp 执行。
训练不依赖 Editor 或其它框架的安装、训练结果和配置。代码来源及许可证见模块 `NOTICE`。

## 准备动作

输入必须是已经重定向到 G1 的机器人动作。LAFAN CSV 的列为 root xyz、root 四元数
xyzw、29个关节角度，单位为米和弧度；默认30Hz。SEED CSV 使用命名列，厘米、角度，默认120Hz。
转换器统一重采样到50Hz，并从 Gobot 场景计算连杆运动学。

```bash
.venv/bin/python -m gobot.rl.tracking.prepare \
  --csv /path/to/dance2_subject1.csv --csv-format lafan \
  --output logs/dance2-assets

# SEED 动作：
.venv/bin/python -m gobot.rl.tracking.prepare \
  --csv /path/to/seed-motion.csv --csv-format seed \
  --output logs/seed-assets
```

输出项目包含 `world.jscn`、`robot.jscn`、`motion.npz`、`project.gobot` 和源数据摘要。
网格引用原机器人资源，需保留源资产。输出目录必须是新目录。

也可用 `--npz motion.npz` 导入命名轨迹：

| 字段 | 形状 / 约定 |
|---|---|
| `joint_pos`, `joint_vel` | `[T,J]`，rad / rad/s |
| `body_pos_w` | `[T,B,3]`，世界坐标，m |
| `body_quat_w` | `[T,B,4]`，wxyz |
| `body_lin_vel_w`, `body_ang_vel_w` | `[T,B,3]`，m/s / rad/s |
| `fps`, `joint_names`, `body_names` | 帧率与源列名称；缺少名称时导入阶段用 `--layout layout.json` 显式提供 |

导入按名称映射，拒绝重复名称、非有限值和不匹配的帧率。`--standing-smoke` 可以生成静止参考做链路测试。

## 一个动作、八卡训练一个策略

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m gobot.rl.train g1-tracking \
  --config examples/g1/train/tracking_dance_8gpu.json \
  --set project_path='"logs/dance2-assets"' \
  --output logs/dance2-train
```

`num_envs` 始终表示每卡环境数：配置默认每卡1,024，共8,192，训练30,000次更新。
各rank以 `seed + rank` 初始化，独立收集失败并更新动作采样分布；PPO梯度、学习率和
观测归一化跨卡同步。每次更新收集24个控制步，使用5个学习epoch、4个minibatch。
单卡可使用普通 Python 启动同一入口。

```python
from gobot.rl.tracking import train_motion

if __name__ == "__main__":
    train_motion("logs/dance2-assets", output="logs/dance2-train",
                 num_envs=1024, updates=30000, seed=42)
```

输出包括 `model.pt`、定期检查点、`config.json`、TensorBoard、`metrics.jsonl`、`status.json`
及源代码归档。`synchronization.json` 检查各卡模型、优化器和归一化状态一致。

恢复使用同一卡数、每卡环境数和任务配置，`updates` 表示本次追加更新数：

```bash
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m gobot.rl.train g1-tracking \
  --config examples/g1/train/tracking_dance_8gpu.json \
  --set project_path='"logs/dance2-assets"' --set updates=1000 \
  --resume logs/dance2-train/model.pt --output logs/dance2-resume
```

恢复网络、优化器、采样分布和随机状态后开始新的物理回合。检查点必须符合当前tracking协议；
没有旧格式转换、旧字段别名或外部框架检查点导入。

## 评估和实时播放

```bash
.venv/bin/python -m gobot.rl.tracking.evaluate logs/dance2-train/model.pt \
  --backend mujoco-warp --device cuda:0 --output logs/dance2-train/evaluation.json

.venv/bin/python -m gobot.rl.tracking.play logs/dance2-train/model.pt \
  --output logs/dance2-editor
.venv/bin/gobot_editor --path logs/dance2-editor
```

评估和播放使用同一个策略加载器，核对场景、动作、关节布局和观测协议。项目移动后可通过
`--project-path` 指定本地项目位置，内容仍须匹配检查点。

点击 Editor 顶部 Play：蓝色为策略实时控制的机器人，绿色为参考动作。推理50Hz，物理200Hz；
顶部支持暂停、停止、单步和重新开始。接触点、力箭头和逐连杆受力均由内置 Physics 面板管理。
显示时平移两台机器人以方便对比，不改动物理状态。

评估关闭随机化、观测噪声和推力，从第0帧状态开始。遇到首次失败或动作末尾即停止，
保留最终物理姿态；训练的10秒回合限制不截断整段评估。`--steps` 可限制评估步数，
`--video video.mp4` 可录制实际物理运动。

## 固定的任务语义

- root为 `pelvis`，anchor为 `torso_link`，跟踪29关节及14刚体。actor为160维、critic为286维。
- 策略输出默认关节角的缩放偏移，经过PD驱动；不裁剪策略动作。缩放为 `0.25 * effort_limit / kp`。
- 六项位置、朝向和速度跟踪奖励，加动作变化、软关节限位和自碰撞惩罚。
- **自碰撞事件固定为任一自碰撞接触力超过10N**，每控制周期累计4个物理子步，权重-10。
  CPU直接检查所有匹配接触，GPU使用最大接触力归约实现相同谓词。没有 `none/maxforce` 配方开关。
  不能用净合力代替最大接触力：相反方向的接触力可能互相抵消。
- 奖励和终止读取最后物理子步积分前的连杆状态，下一步观测读取刷新后的状态。
- 自适应采样按每个rank独立运行：失败计数EMA为0.001，基础均匀权重0.1，控制步更新。
- 启动随机化包括躯干质心、编码器偏差和脚底摩擦；训练期间每1～3秒施加随机速度扰动。
- 物理容量默认128个接触、500个约束；溢出报错，不静默丢弃数据。使用IEEE FP32。

这是Gobot的BeyondMimic式跟踪任务。固定的自碰撞子步惩罚是本实现的接触正则项，
不声称与其它引擎的接触奖励逐项完全相同，也不保证跨GPU运行逐位确定。

## 验证

```bash
.venv/bin/python -m pytest -q tests/python/test_tracking.py \
  tests/python/test_tracking_ppo.py tests/python/test_tracking_distributed.py \
  tests/python/test_articulation_batch.py tests/python/test_tracking_playback.py

GOBOT_RUN_MUJOCO_WARP_GPU_TEST=1 .venv/bin/python -m pytest -q \
  tests/python/test_articulation_batch.py tests/python/test_tracking.py -k gpu
```

测试覆盖数学参考值、接触排列不变性、真实GPU接触求解结果、同步PPO、检查点恢复和Editor生命周期。
PPO同输入测试逐项比较GAE、梯度、优化器和归一化；它不把物理采样的数值分歧当作学习循环差异。
