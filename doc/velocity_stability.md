# 速度训练积分稳定性

2026-11-10，118 上 `g1_velocity_v2_seed42_retry1` 在完成 16487 次 PPO 更新后中止。
rank 1、环境 392、控制步 395704 的 `numerical_failure_rank1.pt` 保存了积分前后的状态。
关节速度最大值从 462.39 rad/s 增至 3760.49 rad/s，超过数值检查的 1000 阈值；
状态仍是有限数。环境分配在 9 级楼梯，快照已有自碰撞穿透，不能据此确定最初失稳的触发点。

## 已验证的改动

通用 `VelocityTaskCfg` 及随附 G1、Go1 速度任务使用完整 `Implicit`。它将速度相关的
惯性力导数纳入隐式积分；`ImplicitFast` 省略关节链的这部分导数。MuJoCo CPU 和
MuJoCo Warp 3.17 均原生支持完整隐式，Gobot 无需添加机器人特例或修改物理引擎。
完整隐式会引入数值耗散，并需要更多计算；它不是精确能量守恒算法。

原有动作缩放、力矩上限、奖励、步长、控制周期和数值失败中止规则不变。
旧 Gobot 曾重置数值异常世界、将该步奖励清零后继续；该处理不代表仿真没有失稳。
固定 mjlab 的 NaN guard 默认关闭，即使开启也只检查 NaN/Inf，没有当前的有限速度爆炸检查。

## 同一故障状态回放

使用保存的模型、随机化参数、积分前状态和实际控制目标，运行 4 个 5 ms 物理步。
以下为最后一步的最大速度，回放数值与原多世界运行会有小量差异：

| 积分方式 | CPU | Warp |
| --- | ---: | ---: |
| ImplicitFast | 3787 rad/s | 3791 rad/s |
| Implicit | 320 rad/s | 320 rad/s |

将求解迭代数从 10 增到 100 没有改善原积分器的结果。从这个已失稳状态关闭电机或
将最后一步动作限幅到 5 也不能阻止发散；这不能用于断言更早的动作限制没有作用。

把快照注入完整 `WarpVelocityEnv` 后，完整隐式能完成该控制步，由普通跌倒条件结束 episode，
奖励有限，没有触发数值恢复或跳过物理步骤。

可以在原始运行目录保留快照，用通用诊断工具独立回放：

```bash
.venv/bin/python tests/support/replay_locomotion_failure.py \
  logs/g1_velocity_v2_seed42_retry1/numerical_failure_rank1.pt \
  --backend warp --integrator implicit
```

换成 `--integrator implicitfast` 对照；`--backend cpu` 使用 CPU MuJoCo。
回放工具不加载策略、不更新 checkpoint，也不重启训练。

## 回归与性能

`tests/python/test_implicit_stability.py` 使用无重力、无接触、无执行器的两节正交关节机构，
独立于 G1/Go1。0.5 秒回放中，ImplicitFast 的能量超过初始值 6 倍，而完整隐式没有能量增长。
CUDA graph 下的 Warp provider 与 CPU 完整隐式轨迹一致。

在 118 上，旧 checkpoint 仅用于固定策略诊断：1024 个环境、最高难度楼梯、随机动作采样，
完整隐式跑 5000 个控制步（512 万个环境步），最大速度 52.45 rad/s，无数值失败。
相同脚本的 ImplicitFast 对照跑 3000 步也未失败，因此不能把短回放成功当作长期训练稳定的证明。
完整隐式耗时约 27.1 ms/控制步，原积分器约 18.4 ms/控制步，采样耗时约增加 47%；
该测量包含诊断同步，不是完整 PPO 的吞吐基准。

新配置必须使用新实验目录。默认八卡配置输出到 `logs/g1_velocity_implicit_seed42`。
旧 checkpoint 的求解器契约不同，不能直接 `--resume` 或 `--initialize`。
故障回放、通用回归和短训练能验证本次修复，但还需由完整训练确认长期表现。
