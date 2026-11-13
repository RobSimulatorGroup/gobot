# G1 SuperMimic

Gobot 的 SuperMimic GenCoord 工作流使用 `.jscn` 场景、MuJoCo CPU/Warp、可微世界模型和 BPTT actor-critic。
训练入口为 `g1-supermimic`，Python 实现在 `gobot.rl.supermimic`。

实现对齐 `HybridRobotics/supermimic` 的 `abc124c1` 提交：单帧 `rep_state`、10帧/stride5目标、
WM窗口8、想象长度32、Muon＋Adam、固定学习率2e-4、ReturnSampler。
这是指定代码版本的实现，不代表论文所有实验配方或最终成绩已复现。
源版本与数值容差在 `reference.lock.json`；独立数学对照与真实场景行为测试在 `tests/python/test_supermimic*.py`。

## 场景

在已安装参考依赖的环境中导出官方有效模型，再通过 Gobot 导入为可编辑场景：

```bash
PYTHONPATH="$PWD/supermimic/src:$PWD/mjlab_research/src" \
  .venv/bin/python -m examples.g1.supermimic.export_model \
  --output logs/supermimic/model_export

.venv/bin/python -m examples.g1.supermimic.prepare_scene \
  --reference-export logs/supermimic/model_export \
  --project examples/g1/assets/supermimic
```

导入后核对质量、惯量、关节、执行器、碰撞体和 solver 设置。训练直接读取 Gobot 场景与 profile。

## 动作数据

训练库包含 `root_pos_w`、`root_quat_w`、`joint_pos`、`root_lin_vel_w`、`root_ang_vel_w`、`joint_vel`
六组 NumPy 数组，以及动作名、结束帧、关节名和带校验值的 manifest。采样率50Hz，四元数wxyz，根速度为世界坐标。
已有库可直接使用，不必重新转换。

将已有71D紧凑动作包按原精度拆分为训练库：

```bash
.venv/bin/python -m examples.g1.supermimic.prepare_bones prepare \
  --source /path/to/library.pt \
  --profile examples/g1/assets/supermimic/assets/supermimic/robot_profile.json \
  --output /path/to/bones_train

.venv/bin/python -m examples.g1.supermimic.prepare_bones verify --data /path/to/bones_train
```

完整评估使用原始FP32 NPZ参考，避免经过FP16转换。输入是已审计的BONES目录，包含
`npz/`、`manifest.json`、`manifest_all.jsonl`、`train_clips.json` 和 `eval_clips.json`：

```bash
.venv/bin/python -m examples.g1.supermimic.prepare_reference \
  --data /path/to/bones_dense \
  --profile examples/g1/assets/supermimic/assets/supermimic/robot_profile.json \
  --output /path/to/bones_fp32 --workers 8
```

转换器核对源文件校验值、关节顺序、动作成员与帧数；失败时不发布不完整库。

## 训练

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m gobot.rl.train g1-supermimic \
  --config examples/g1/train/supermimic_bones_2048.json \
  --project examples/g1/assets/supermimic --library /path/to/bones_train \
  --seed 0 --steps 100000 --output logs/g1-supermimic/bones_2048_s0
```

八卡，每卡2048环境（全局16384），学习率不随卡数放大：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
.venv/bin/torchrun --standalone --nproc_per_node=8 -m gobot.rl.train g1-supermimic \
  --config examples/g1/train/supermimic_bones_2048.json --batch-scope per_rank \
  --project examples/g1/assets/supermimic --library /path/to/bones_train \
  --seed 0 --steps 100000 --output logs/g1-supermimic/bones_2048pergpu_8gpu_s0
```

`--batch-scope global` 是默认值，将配置中的环境数、WM/policy batch均分给各卡；`per_rank` 表示每卡数量。
梯度跨卡平均后裁剪并更新，归一化和采样统计也同步。配置提供1024、2048和4096三档。
`--steps` 为每环境控制步目标，`--max-seconds` 为本次训练循环的时间上限。

每次训练输出 `config.json`、`requested_config.json`、`run.json`、`metrics.jsonl`、心跳 `status.json`、
TensorBoard事件和checkpoint。非零rank有各自的状态文件。

```bash
.venv/bin/tensorboard --logdir logs/g1-supermimic --port 6006
```

`episode_return` 是已结束回合按实际步数归一化的平均回报；`Policy/reward` 来自想象轨迹。
两者均不能代替固定完整动作评估。

## 恢复

同源码、同配置、同卡数使用 `--resume`，并指定新的输出目录和绝对 `--steps` 目标。
恢复网络、优化器、采样器和各rank随机状态；物理回合重新开始、replay重新填充，非逐位连续轨迹恢复。

`--initialize-from` 显式迁移匹配架构、场景、数据的网络和归一化参数，重新建立优化器和训练计数。
两者不能同时使用。清理前checkpoint的严格续训需要原源码分支；这里不提供旧模块路径或源码签名兼容层。

## 评估

固定设置：原始FP32参考、FP32推理/TF32关闭、无噪声或DR、friction1、dt=.005×4、128/500容量。
从frame0重置，以frame1作为首目标，N帧执行N−1步，首次失败或动作结尾结束。高度阈值0.25m、倾斜阈值0.8。
同条轨迹记录骨盆与躯干回报，保留训练时的策略观测布局。

单卡或指定分片：

```bash
.venv/bin/python -m gobot.rl.supermimic.evaluate \
  --checkpoint /path/to/latest.pt --project examples/g1/assets/supermimic \
  --library /path/to/bones_fp32 --output logs/evaluation/shard0.json \
  --num-envs 512 --shard-index 0 --shard-count 1
```

八卡评估一个或多个checkpoint，各模型依次使用同八卡：

```bash
.venv/bin/python -m examples.g1.supermimic.evaluate \
  --checkpoints /path/to/checkpoint_a.pt /path/to/checkpoint_b.pt \
  --project examples/g1/assets/supermimic --library /path/to/bones_fp32 \
  --output logs/evaluation/comparison --gpus 0 1 2 3 4 5 6 7 --num-envs 512
```

可用 `--filter-manifest /path/to/eval_manifest.csv` 补充固定过滤集结果。只有完整成员、同一条件且无截断的分片才可合并。
比较结果含全量回报、失败数、修复/新增失败名单和checkpoint身份。模型选择以评估为准。

## 验证

```bash
.venv/bin/python -m pytest -q tests/python/test_supermimic*.py
```

设置 `GOBOT_SUPERMIMIC_TEST_PROJECT` 和 `GOBOT_SUPERMIMIC_TEST_LIBRARY` 后，会执行真实CPU场景、
训练恢复、双进程同步和完整评估测试。未设置时对应测试跳过，数学和数据格式测试仍运行。
重建独立官方数值fixture的入口为 `examples.g1.supermimic._reference.export`，该进程不导入Gobot。
