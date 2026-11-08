# G1 行走模型与 mjlab 执行对照

2026-11-07。Gobot 的 G1 行走模型已换成 mjlab `dcdf6ce519abfd2ea45e9503ae6fb650b75bb869`
使用的 29 关节、固定手掌模型。质量、惯量、关节、驱动和碰撞的 1,125 项静态检查通过。
行走、AMP 和回放共用 `examples/g1/g1.jscn`，可视网格随仓库提供；原来的手指 PD 分支和碰撞补丁文件已删除。

此次用 **同一个已训练的 Gobot checkpoint** 比较三个执行环境，未重新训练：

- 旧 Gobot 模型：`11948e5`，43 个活动关节，其中 14 个手指由 PD 保持目标。
- 统一后的 Gobot：29 个身体关节、固定手掌。
- mjlab：独立构建参考 G1，读取同一策略、相同地形几何和初始状态。

checkpoint 为 `g1_ppo_8gpu/model_final.pt`，iteration 17799，原训练为 8 GPU / 8192 环境。
因此后两列都是旧策略的模型迁移结果，**不是 Gobot 与 mjlab 分别训练后的收敛成绩**。
三个种子是评测种子，不是三个训练种子。

## 评测条件

三组种子 71 / 72 / 73。本机 RTX 4070；MuJoCo / MuJoCo Warp 3.17.0，
Torch 2.11.0+cu128，RSL-RL 5.5.2。两边接触容量 70、约束容量 1500、EPA 边容量 64。确定性 actor，关闭观测噪声、推力与启动随机化，
保留 ±0.1 m 出生位置扰动和原配置的出生高度扰动。物理 200 Hz，控制 50 Hz。

平地：每个速度每个种子 32 次、20 秒，共 384 次运行。
粗糙地形：7 种类型 × 10 个难度 × 3 个种子，共 210 次，前进指令 0.8 m/s，最多 20 秒。
每个地形块从 -3.25 m 行走到 +3.25 m，偏离横向 ±3.5 m 或退出起点后方算路线失败。
平地/粗糙地形评测保留同一粗糙地形策略的观测结构和求解器参数。

速度误差为 `||实测水平速度 - 指令水平速度||₂` 的时间平均，单位 m/s，排除前 2 秒，
在首次结束后不再累计。它不是只测 vx 的误差。跌倒和数值异常分别统计；
通过率只计算活着到达对侧终点的次数，偏离路线、跌倒或超时均不通过。
速度误差会受存活时长影响，必须和跌倒率、通过率一起看。

15 个 mjlab 场景的初始 actor 观测与 Gobot 导出数据的最大差为 **0**，
CUDA 动作相对 CPU 导出的动作最大差为 **1.4×10⁻⁶**，通过 10⁻⁴ 核对阈值。
地形、起始状态及策略由文件导出；mjlab 评测进程不导入 Gobot。
这保证了对照输入一致，不保证长时间接触动力学逐位一致。

## 结果

平地速度误差：

| 指令 m/s | 旧 Gobot | 统一后的 Gobot | mjlab |
| --- | ---: | ---: | ---: |
| -0.3 | 0.0895 | 0.0729 | 0.0730 |
| 0.8 | 0.0593 | 0.0757 | 0.0757 |
| 1.5 | 0.0368 | 0.0425 | 0.0413 |
| 2.5 | 0.0560 | 0.0549 | 0.0561 |

三者平地各 384 次运行均无跌倒。统一后的 Gobot 与 mjlab 的平地误差接近。

粗糙地形：

| 指标 | 旧 Gobot | 统一后的 Gobot | mjlab |
| --- | ---: | ---: | ---: |
| 速度误差 m/s | 0.0877 | 0.1089 | 0.1079 |
| 跌倒次数 / 210 | 10 | 15 | 19 |
| 跌倒率 | 4.76% | 7.14% | 9.05% |
| 通过次数 / 210 | 126 | 117 | 114 |
| 地形通过率 | 60.00% | 55.71% | 54.29% |

各地形通过次数，每类 30 次：

| 地形 | 旧 Gobot | 统一后的 Gobot | mjlab |
| --- | ---: | ---: | ---: |
| 平地 | 30 | 30 | 30 |
| 凸楼梯 | 20 | 23 | 19 |
| 凹楼梯 | 20 | 17 | 17 |
| 凸坡 | 4 | 3 | 3 |
| 凹坡 | 9 | 7 | 8 |
| 随机粗糙 | 13 | 12 | 14 |
| 波浪 | 30 | 25 | 23 |

旧策略在统一模型上没有保持原粗糙地形水平，主要短板仍是坡面和路线跟踪。
Gobot 与 mjlab 粗糙地形结果仍有差距；现有数据不足以判定某个训练框架更好，
也不证明自然步态、抗扰动能力或训练收敛一致。下一轮训练应从统一模型开始，
再按相同样本预算和训练种子比较独立训练结果。

## 修复与复现

原评测起点 ±3 m 恰好位于楼梯拼接边界，所有 20 个楼梯格的中心射线未命中。
现起点位于边缘平台内部；CPU 和真实 CUDA 均验证全部 70 格、四个方向的出生高度有效。
同时修复了“更换评测地形会更换策略观测配置”的问题，单列跌倒、数值异常、偏离路线和穿越结果。

在 Gobot 环境运行（对旧模型 checkpoint 显式允许迁移）：

```bash
for seed in 71 72 73; do
  python -m examples.locomotion.assessment \
    --checkpoint logs/g1_ppo_8gpu/model_final.pt --allow-model-change \
    --terrain flat --speeds -.3 .8 1.5 2.5 --steps 1000 --num-envs 32 \
    --seed "$seed" --output "logs/g1_comparison/flat_${seed}.json"
  python -m examples.locomotion.assessment \
    --checkpoint logs/g1_ppo_8gpu/model_final.pt --allow-model-change \
    --terrain rough --speeds .8 --steps 1000 --num-envs 70 \
    --seed "$seed" --output "logs/g1_comparison/rough_${seed}.json"
done
PYTHONPATH=. python tests/support/export_locomotion_comparison.py \
  --checkpoint logs/g1_ppo_8gpu/model_final.pt --allow-model-change \
  --output logs/g1_comparison/common
```

在同版本数值依赖、已安装固定版本 mjlab 的环境运行：

```bash
python tests/support/evaluate_mjlab_locomotion.py \
  --input logs/g1_comparison/common --output logs/g1_comparison/mjlab.json
```

新模型训练得到的 checkpoint 无需 `--allow-model-change`。旧模型 checkpoint 的正常恢复与回放
仍会拒绝模型不匹配，不会静默忽略。AMP 的动作特征数据也应按新模型重新生成。

本次原始报告、checkpoint 哈希、逐次结果和视频位于被 Git 忽略的 `logs/g1_alignment/`，
汇总为 `comparison.json`。模型参考固定在 `tests/fixtures/mjlab_g1.xml`，
对应的来源哈希与配置在 `tests/fixtures/mjlab_velocity.json`。


## mjlab 独立训练入口

`tests/support/train_mjlab_locomotion.py` 直接调用固定提交的 mjlab G1 配置和原始 PPO runner，
不导入 Gobot。依赖与本次对照固定一致；运行 `--check` 可先核对。118 的独立环境安装清单位于
`logs/mjlab_comparison_setup/requirements.txt`，这是实验环境文件，不参与 Gobot 安装。

在 mjlab 源码目录，通过其 Python 环境启动：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  ../tests/support/train_mjlab_locomotion.py \
  --terrain-reference ../logs/g1_training_terrain.json \
  --num-envs 1024 --iterations 18000 --seed 42 --save-interval 500 \
  --output logs/g1_comparison_seed42
```

每卡 1024 环境、每次采样 24 步，共同训练一个策略。先用 `--num-envs 64 --iterations 10` 和单独
输出目录做短测试。`--resume path/to/model_N.pt` 恢复同配置训练；这里的 `--iterations` 是目标总更新数，
区别于 Gobot 入口的追加更新数。物理 episode 重新开始，不是完整物理轨迹恢复。

求解器保持 10 次迭代、20 次线搜索。启动前只关闭这两类逐内核重复打印，在每轮训练日志中汇总
`Solver/iterations_limit_fraction` / `Solver/linesearch_limit_fraction`。比例按全体 GPU 上所有环境的
物理子步和 forward 求解调用加权，reset 清理前已收集结果，其余容量警告保留。
这改变日志形式，不证明求解器已经收敛。需要提高迭代参数时应作为单独实验记录。

该训练入口现已要求共同地形文件，并统一全卡统计。先按 [g1_shared_training.md](g1_shared_training.md) 导出地形，再使用新输出目录从头建立下一轮基线。
