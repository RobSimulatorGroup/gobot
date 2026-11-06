# Gobot 内置 SuperMimic 训练

G1 tracking 的唯一训练实现是固定版本 SuperMimic-Reproduced 的迁入版本。
旧 Uni-SuperMimic learner、旧观测/目标、fork/promote 适应流程及相应测试、工具和配方已删除。
PPO/AMP locomotion 是独立任务，仍然保留。

2026-11-06 已完成一次八卡 BONES 100k 从头训练与完整评测，见[结果报告](g1_bones_100k_results.md)。
完整操作入口：[数据准备/评测/视频](../examples/g1/reproduction/README.md)。

训练入口为 `python -m gobot.rl.train` / `gobot-train`。场景和 robot profile 由 Gobot 拥有，
训练不依赖 mjlab、编辑器或 NodeScript。mjlab 仅用于独立参考对照。

## 固定目标

- SuperMimic-Reproduced `ba6105349069bf040f1fdad60873d13138d505a2`。
- mjlab `8bb2bc9e187797689ec82c78c35019afd2fd0376`。
- G1 29 DoF；50 Hz 控制，200 Hz 物理；93D 观测、term-major 10 帧历史、680D 目标。
- 显式积分世界模型、学习终止头、8 步 replay window、32 步 BPTT、GAE critic。
- 正式 e8 配方每卡 4096 env、WM batch 2048 × 10、policy batch **2048**、UTD 0.4。
- 基础学习率 0.0002；4 卡时乘 sqrt(4) 得到 0.0004。配置文件始终存基础值。
- 320000 控制步调度、250000 衰减结束、100000 停止分开表达。

Gobot 的原生批处理依赖 MuJoCo 3.17。原始发布栈 3.16 复核和同 3.17 栈配对实验分开记录，
不能把 3.17 上的诊断当作原始发布栈复现。[当前证据](supermimic_reproduced_progress.md)列出已验证项目和缺口。

## 本地构建与诊断

使用本仓库重新构建的 Gobot Python 包。开发构建可通过
`cmake --build build --target gobot_python` 更新，再设置
`PYTHONPATH="$PWD/build/python:$PWD"`。不要混用新 Python 包与旧原生库。
当前 checkout 的验证使用这个独立构建目录；主 `.venv` 的原生安装没有被覆盖。
项目最低 Torch 版本现为 2.9，配对实验实际锁定使用 2.11。

## 准备场景与数据

先使用 [reproduction 工具](../examples/g1/reproduction/README.md) 导出固定参考的有效 G1 模型，再执行：

```bash
python -m examples.g1.prepare_tracking --reference-export /data/reference_model --project-path /data/g1_project
python -m gobot.rl.motion_tracking.convert --project /data/g1_project \
  --input /data/lafan1/*.csv --input-fps 30 --output /data/lafan1_npz --compact
```

转换遵循参考的端点、插值和速度定义，以 Gobot 场景 FK 输出 50 Hz NPZ；可同时输出 fp16 `library.pt`。
`manifest.json` 记录 CSV/NPZ/packed library 哈希、帧数和场景 FK 身份。
BONES CSV 支持厘米、欧拉角度和关节度数的输入格式，必须显式填写真实输入 FPS。
不能用公开评估片段清单替代原始训练数据清单。

全量 BONES 使用 `examples.g1.reproduction.export_bones_reference` 和 `prepare_bones`，
分别导出固定参考输入、分批转换并按历史规则独立筛选；详细命令见工具 README。
训练库和评估子集清单分开保存，已完成的数据包含 129943 条训练动作与 129491 条评估子集动作。
`library.pt` 是 71D fp16 动作数据，`ckpt_latest.pt` 才是模型及优化器 checkpoint；LAFAN1 单独存储。

## 训练和恢复

```bash
# 完整网络、缩小批次和 rollout 的功能检查；不是成绩实验。
python -m gobot.rl.train --project /data/g1_project --data /data/lafan1_npz \
  --config examples/g1/train/reproduced_smoke.json --output logs/smoke

# 单节点同步四卡，目标 e8 配方；必须有实际 BONES 训练库。
torchrun --standalone --nproc-per-node 4 -m gobot.rl.train \
  --project /data/g1_project --library /data/bones/library.pt \
  --config examples/g1/train/reproduced_bones.json --output logs/g1_e8_s0

# 按相同 rank 数恢复；先重新填充 replay，再继续绝对控制步调度。
torchrun --standalone --nproc-per-node 4 -m gobot.rl.train \
  --project /data/g1_project --library /data/bones/library.pt \
  --config examples/g1/train/reproduced_bones.json --output logs/g1_e8_s0_resume \
  --resume logs/g1_e8_s0/ckpt_latest.pt
```

`--stop-iteration` 是绝对停止点，不改变 LR 调度。恢复要求相同场景、数据、rank 数和算法参数。
每个 rank 保存自己的 sampler 与 RNG；世界模型、actor、critic 及优化器同步。
与参考恢复方式相同，episode/replay 重新开始，warmup 40 步，UTD 累加器从零开始；不承诺逐位连续恢复。
`config.json` 分别保存 base/effective config、软件版本、数据哈希及模型身份；checkpoint 也校验训练源码哈希。
发布 e8 有 92k 恢复历史，更新数 39968；默认不中断 100k 重建配方为约 39984 次更新，两者必须分别记录。
`metrics.jsonl`、TensorBoard 和 checkpoint 都使用控制步计数。

八卡 4090 实测使用独立的 `examples/g1/train/reproduced_bones_nj500.json`，只将约束容量
从 250 改为 500。原配方仍保留。不能在 `--resume` 时修改容量；也不能把八卡增加的全局
样本量与学习率当成和四卡等价。每 rank 保存一份 checkpoint，八卡续训须保留全部八份。
每 500 控制步打印一次；第一次日志前可能包含初始化、编译和 warmup。

## 完整片段评估

```bash
python -m examples.g1.evaluate_tracking --project /data/g1_project --data /data/lafan1_npz \
  --checkpoint logs/g1_e8_s0/ckpt_latest.pt --num-envs 40 --output logs/g1_e8_s0/evaluation.json
```

所有片段从首帧开始，关闭观测/动作/reset 噪声和随机化，足部摩擦固定 1。
报告逐片段失败、实际执行步数、归一化 return 和各奖励项。
`--num-envs` 不超过选定片段数。单世界诊断若触发容量溢出，可对**双方**显式使用
`--nconmax 128 --njmax 500`；报告会记录容量，不能将它视为正式 35/250 配方。
公开 teacher 评估使用 `--release` 并强制验证固定 SHA256；
teacher 评估结果与 Gobot 从头训练结果分别标记。

上述入口适用于 LAFAN1 或有限片段。全量 BONES 使用
`python -m examples.g1.reproduction.evaluate_bones prepare|worker|merge` 分批加载 dense NPZ，
不要把全库一次塞进显存。最终报告保留全量和过滤子集、逐片段失败、实际步数和 coverage。
视频使用 `python -m examples.g1.reproduction.record_tracking --outcome success|failure ...`；
成功视频必须完成整条片段，失败视频在终止处结束。工具 README 提供可执行命令。

## 代码与验证

| 职责 | 位置 |
|---|---|
| 配置、replay、sampler、WM、BPTT、critic、分布式梯度 | `python/gobot/rl/supermimic/` |
| 场景训练环境、数据转换、训练时钟、恢复、评估 | `python/gobot/rl/motion_tracking/` |
| 任务注册和 CLI | `python/gobot/rl/tasks/`、`train.py` |
| 独立参考工具与来源锁定 | `examples/g1/reproduction/` |
| 数学和完整更新 fixture 回归 | `test_supermimic_reproduced_math.py`、`test_reproduced_training.py` |

回归覆盖独立参考的 replay wrap/terminal rows、WM 更新、actor BPTT、critic GAE、梯度和 Adam 状态。
原生场景/物理回归继续执行。最终成绩对齐需要相同训练库、训练量和独立 seeds 的配对训练；
冒烟测试、teacher 移植和数学测试都不能代替这一步。

恢复 checkpoint 包含每个 rank 的 startup COM；恢复时不依赖环境重新构造所消耗的随机数。
多 rank 恢复会比较控制步、learner 更新和三套优化器计数，拒绝混合不同进度的 shards。
配置的正整数维度、采样模式、学习率和 replay/window 要求会在创建仿真前检查。
