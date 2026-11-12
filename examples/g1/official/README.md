# 官方 SuperMimic / BONES-SEED

`g1-supermimic` 使用 Gobot `.jscn` 场景、`ArticulationBatch` 和官方 GenCoord
数学/学习器。训练进程不导入 mjlab、官方 checkout 或 W&B。
`g1-tracking` 仍为旧 Reproduced 配方，checkpoint 互不兼容。

首版支持 **单 GPU / 单进程**。不要对 `g1-supermimic` 使用 torchrun；多 seed 用不同
进程、不同 GPU、不同输出目录运行。完整 BONES 数据转为 float32 后约占 13.5 GB
设备内存，正式 4096 环境/2048 WM batch/4096 policy batch 的峰值还取决于图编译。
本机验证覆盖真实数据的小 batch；没有宣称完整 4096 batch 已在 24 GB 卡上验收。

## 1. 场景

已有经验证的官方 G1 场景时，复制整个场景目录（包括 assets 与 audit），例如放到
`examples/g1/assets/official`。不能把旧 `project_317` profile 改名当作官方场景。

从源码重建时，独立参考解释器需要可导入固定 checkout 的依赖，且导出/导入都使用
MuJoCo 3.17.0。`reference.lock.json` 固定源码，`export_model` 会校验来源：

```bash
PYTHONPATH="$PWD/supermimic/src:$PWD/mjlab_research/src" \
"$REFERENCE_PYTHON" -m examples.g1.official.export_model \
  --output logs/supermimic_official/model_export

.venv/bin/python -m examples.g1.official.prepare_scene \
  --reference-export logs/supermimic_official/model_export \
  --project examples/g1/assets/official
```

输出目录必须不存在。官方旧 `uv.lock` 与配套源码当前依赖声明不同，不能把直接
`uv sync --frozen` 的安装结果视为已经验证过的同栈运行环境。本轮实际导出与测试软件栈
记录在 [对齐状态](../../../doc/supermimic_official_alignment.md)。

## 2. BONES-SEED 数据转换（只需一次）

2026-11-12：118 下述目标目录已完成全量转换和校验（129943 clips / 47475680 frames），
且已通过完整库加载与 CPU 单步检查。现有 118 环境可直接跳到训练步骤，不要覆盖重建。

在 118 上，从原先已审计的 129943 clips 紧凑库转换：

```bash
cd /home/h3c-r5300-g6/wqq/gobot
.venv/bin/python -m examples.g1.official.prepare_bones prepare \
  --source /home/h3c-r5300-g6/wqq/datasets/bones_gobot_e8_50hz/library.pt \
  --profile examples/g1/logs/reproduced_alignment/project_317/assets/supermimic/robot_profile.json \
  --output /home/h3c-r5300-g6/wqq/datasets/bones_supermimic_official_50hz
```

转换只拆分字段，不重新采样或 FK，保留原 fp16 数值、clip 顺序和边界。
文件是 `root_pos_w/root_quat_w/root_lin_vel_w/root_ang_vel_w/joint_pos/joint_vel.npy`、
`motion_ends.npy`、关节/clip 名称 JSON 和 SHA256 manifest；两边加载时转 float32。
这沿用的是现有 Reproduced 数据集成员与量化，不能称为官方训练原始数据的逐字节复现。

`--limit 4` 可在**另一个目录**做转换检查。已有完成目录不会被覆盖，校验命令为：

```bash
.venv/bin/python -m examples.g1.official.prepare_bones verify \
  --data /home/h3c-r5300-g6/wqq/datasets/bones_supermimic_official_50hz
```

独立参考解释器可运行 `examples.g1.official.check_data --data ... --output ...`；
该工具不导入 Gobot，直接使用固定 `mjlab_research.MotionLoader` 核对字段。

## 3. 训练

在 118 使用一张空闲 GPU；输出目录必须不存在：

```bash
cd /home/h3c-r5300-g6/wqq/gobot
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m gobot.rl.train g1-supermimic \
  --config examples/g1/train/official_bones.json \
  --project examples/g1/assets/official \
  --library /home/h3c-r5300-g6/wqq/datasets/bones_supermimic_official_50hz \
  --backend mujoco-warp --device cuda:0 --seed 0 \
  --output logs/g1-supermimic-official/bones_s0
```

先加 `--dry-run` 可核对配置，不构建物理世界。
按训练循环墙钟时间限制，例如每次 1 小时：追加 `--max-seconds 3600`。
计时包含编译、采样、更新、日志和中间 checkpoint，排除数据载入/环境构建和最后保存；
到期在完整 learner 更新结束后保存，因此可能略超预算。`--steps` 是绝对控制步目标，
默认 100000。日志分别记录控制步、环境 transitions、WM/actor/critic 更新次数。
`Ctrl+C` / SIGTERM 请求在当前控制步完成后保存，不用强制 kill 模拟正常结束。

显存不足时，两边对照必须同时调整相同的 batch/env 配方。例如：
`--num-envs 1024 --set wm.batch_size=512 --set policy.batch_size=1024`。
不要只改变一边后仍称为相同配置实验。使用 `--no-compile --set capture_updates=false`
可诊断 eager 路径；这会改变吞吐。`official_smoke.json` 的小网络只验证链路，不用于质量比较。

原版官方训练器也可直接读取**同一个转换目录**：

```bash
cd /path/to/supermimic
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline \
python src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file /home/h3c-r5300-g6/wqq/datasets/bones_supermimic_official_50hz \
  --num-envs 4096 --env.seed 0 --env.sim.njmax 500 \
  --runner.run-name bones_s0 --runner.max-iterations 100000
```

该命令需要安装好官方参考环境；原 runner 不提供墙钟停止保存参数，100000 iterations
是控制步预算，不是等时比较。此次交付不把两边训练质量或等时结果标为已通过。

## 4. 恢复和评估

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m gobot.rl.train g1-supermimic \
  --config examples/g1/train/official_bones.json \
  --project examples/g1/assets/official \
  --library /home/h3c-r5300-g6/wqq/datasets/bones_supermimic_official_50hz \
  --backend mujoco-warp --seed 0 --steps 200000 \
  --resume logs/g1-supermimic-official/bones_s0/latest.pt \
  --output logs/g1-supermimic-official/bones_s0_continue

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m gobot.rl.supermimic.official.evaluate \
  --checkpoint logs/g1-supermimic-official/bones_s0/latest.pt \
  --device cuda:0 --num-envs 64 \
  --output logs/g1-supermimic-official/bones_s0/evaluation.json
```

恢复保存网络、优化器、RNG、启动随机化、sampler 的长期统计和计数器。物理回合重置、
replay 重填，旧回合的 sampler 窗口被清除；不是物理轨迹逐位连续。
严格检查源代码/配置/场景/数据身份；实验中修改这些内容应新建运行，不能强行续训。
最终 checkpoint 为 `latest.pt`，每 1000 控制步另保存编号 checkpoint。

评估从每个 clip 起点按官方 reset 后参考前进一帧的时序开始，确定性 actor、关闭 DR 和噪声，
以有限数量 lane 排队评估。输出每片段回报、失败和 coverage。`--clips name...` 限定片段；
`--max-steps 100` 仅用于调试，到达上限会标记 incomplete 并返回失败退出码。
该评估是训练集重放，不能替代 held-out 或正式多 seed 质量验收。
