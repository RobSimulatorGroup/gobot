# SuperMimic / Reproduced 共用评估基准

目的：冻结现有两个最终checkpoint，不重新训练，用同一模拟环境/数据/时钟评分。
这是同评估条件的策略对照；训练预算、观测信息、目标窗口仍为各自模型的一部分，不是同预算算法消融。

## 固定条件

- 原始已审计 dense FP32 NPZ，逐文件校验历史 `npz_sha256`，只提取root/joints的六组状态；
  不经过fp16、不重采样、不混用已量化训练缓存。两边共用相同manifest和scene FK生成参考link状态。
- 同一官方Gobot `.jscn` 场景。原Reproduced与官方场景的关节顺序、动作缩放、默认姿态、软限位、
  控制周期和solver profile必须一致。本机检查466组编译后MuJoCo数值数组相同；authoring的
  limit_compliance写法不同（0与0.0004），编译后jnt_solref均为(0.02,1)。
- MuJoCo Warp，physics_dt=.005、decimation=4（50Hz），nconmax128/njmax500，512 lanes/worker，
  physics CUDA graph开启，额外Warp deterministic reductions关闭。所有推理明确FP32、TF32关闭。
- reset采用frame0 pose/velocity、前一动作0、历史重复初始sensor；首目标frame1。
  N帧严格运行N−1步或首次失败停止，无10秒训练截断。
- 无obs/action/reset噪声，无DR，足底摩擦1.0，名义CoM、encoder bias0。
- 相同骨盆失败判定：高度误差>0.25米或gravity_z差>0.8。两者都读最后积分前一physics子步的link状态。
- **同条轨迹同时计算pelvis和torso_link两套奖励**，函数、权重、soft limits、采样时刻完全共用。
  归一化为每clip `sum(reward)/(0.1*executed_steps)` 再对clips取均值；另外报告完整长度归一化回报和coverage。

## 策略输入保留

- Reproduced：10×93D term-major proprio history，680D目标（10帧、stride1）。
- 官方：1×68D privileged GenCoord state，640D目标（10帧、stride5）。

输入适配器只将相同state/reference转换成训练时布局，不改变checkpoint、控制器或环境评分。
不能强行给旧网络喂官方观测，否则测试的是输入接口错误。该信息差异必须随对比结果一起报告。

## 实现与校验

- `python/gobot/rl/supermimic/common_evaluation.py`：唯一环境，两个冻结策略适配器，dual-anchor评分、严格分片合并。
  放在official包之外，不改变现有训练源码identity。
- `examples/g1/reproduction/prepare_common_reference.py`：有界并行解压/校验dense NPZ，原子发布FP32库。
- `examples/g1/reproduction/compare_common.py`：两轮独立8卡评估、选集和协议指纹校验、逐片段比较。
- 真实CPU场景测试核对旧adapter的观测/history/goals/actions与原环境逐步一致、torso reward一致；
  官方adapter和pelvis reward与原环境一致。另验证clip-end、capped不算完成、损坏数据/不同协议不能合并。

## 118 命令

FP32库只需生成一次，已存在时不要覆盖：

```bash
.venv/bin/python -m examples.g1.reproduction.prepare_common_reference \
  --data /home/h3c-r5300-g6/wqq/datasets/bones_gobot_e8_50hz \
  --profile examples/g1/assets/official/assets/supermimic/robot_profile.json \
  --output /home/h3c-r5300-g6/wqq/datasets/bones_common_fp32_50hz --workers 8
```

两个模型分两轮使用同八卡、同lanes和同一motion顺序，每轮无训练更新：

```bash
.venv/bin/python -u -m examples.g1.reproduction.compare_common \
  --official logs/g1-supermimic-official/bones_2048pergpu_8gpu_s0/latest.pt \
  --reproduced examples/g1/logs/cluster/g1_bones_nj500_s0/ckpt_latest.pt \
  --project examples/g1/assets/official \
  --reproduced-project examples/g1/logs/reproduced_alignment/project_317 \
  --library /home/h3c-r5300-g6/wqq/datasets/bones_common_fp32_50hz \
  --filter-manifest /home/h3c-r5300-g6/wqq/datasets/bones_gobot_e8_50hz/eval_manifest.csv \
  --output logs/supermimic_common_evaluation/full --num-envs 512 --gpus 0 1 2 3 4 5 6 7
```

每个policy的`full.json`含完整common配置与源码hash。只有common指纹、片段名、源帧数、结束时钟都
一致且无缺失/重复/人为截断，比较才发布。历史各自协议成绩不替代本次重测。


## 2026-11-12 全量重测结果

两个已完成的八卡模型均在上述共用条件重新运行129943动作。两轮总墙钟1816.22秒（30.27分钟），
8个worker每轮正常退出，完整合并和跨模型common identity检查均通过。

| 指标 | Reproduced 8卡 | 官方SuperMimic 8卡 |
|---|---:|---:|
| 全量骨盆奖励 | 0.931829 | 0.950115 |
| 全量躯干奖励 | 0.935604 | 0.945823 |
| 全量失败 | 429 / 129943 | 19 / 129943 |
| 全量失败率 | 0.330145% | 0.014622% |
| 固定子集骨盆奖励 | 0.933205 | 0.950736 |
| 固定子集躯干奖励 | 0.936953 | 0.946537 |
| 固定子集失败 | 91 / 129491 | 13 / 129491 |

本轮官方模型失败数少410个（95.57%）；两套共用奖励均更高。共同失败7个，官方单独失败12个，
Reproduced单独失败422个。按名称关键词，flip/handstand/cartwheel失败为54/160/98→0/0/0，
safety_roll为46→6；关键词有重叠，不是人工动作类别。

相比历史各自协议：Reproduced全量仍429，但6个失败变成功、6个成功变失败；官方20→19，
3个历史失败变成功、2个历史成功变失败。历史与本轮同时存在精度/时序/批次设置等变化，
不能把个别标签变化归因于单个因素，也不宣称Warp逐位确定性。

这已是**同评估条件**的比较，但仍不是同训练预算或同输入信息的算法消融：Reproduced训练
3276800000 transitions、10.828小时；官方训练1638400000 transitions、5.436小时；两者均8卡seed0。
官方输入privileged单帧状态，旧模型输入10帧proprio。全部数据为训练动作集，未评估held-out泛化。

身份：

- common协议SHA256：`8d0d64e56dfd065d38390b8dfaa6deda378a49de7fbcfd16c13d65ea7b681ee1`。
- FP32数据manifest：`e8d347d5adb4546804a57a3d0ca2f3d2586cf680087f4a3a3011b72b13926e95`。
- 官方checkpoint：`c00787c1d59800f8fd5137fc6b92e78548c451a88ceb7c3b902b8571da032b63`。
- Reproduced checkpoint：`89de8a3bd3774b4c407bc92063d5447c76efb49c54bc8a930f81cfed2605e279`。
- 评估代码提交：`f8db13b`；43项相关测试通过。

本机与118的结果目录均为 `logs/supermimic_common_evaluation/full_20261112/`：
`report.md`、`comparison.png`、`comparison.json`、`historical_protocol_changes.json`，以及两模型的
`full.json`、`filtered.json`、`failures.json` 和worker日志。原始分片和检查点保留在118。
