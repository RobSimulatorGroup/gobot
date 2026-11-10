# 速度任务的数值异常处理

标准速度任务对齐固定 mjlab 参考 `dcdf6ce519abfd2ea45e9503ae6fb650b75bb869` 的默认行为，
使用 `ImplicitFast`、0.005 秒物理步长、4 个子步、Newton 10 次迭代和 20 次线搜索。
自定义 `VelocityTaskCfg` 的部分求解器配置也默认使用 `ImplicitFast`，不会继承引擎的 Euler。

## 与参考一致的处理

- 不因有限关节速度超过 1000 rad/s 或位置超过自定义阈值而中止训练，也不裁剪物理状态。
- 标准终止条件仍为跌倒、超时和离开地形边界；不增加“数值异常自动重置”。
- 每个奖励项先计算权重和时间缩放，再把该项的 NaN、正无穷和负无穷置零，最后汇总。
  正常的其他奖励项保留，日志中的逐项 rate 等于处理后的奖励除以 `step_dt`。
- 标准 actor/critic 观测不清零 NaN/Inf，与 mjlab 的 `nan_policy="disabled"` 一致。
- 返回观测、奖励或 done 含 NaN/Inf 时，由相同版本 RSL-RL 5.5.2 的 `check_nan` 报错，
  在进入 PPO storage 前结束采样；多卡传播该错误并退出。
- 标准 PPO 使用参考的采样时价值作 timeout bootstrap，不增加终态 critic 计算。
- MuJoCo Warp 的容量超限由运行时警告报告，不在速度任务中额外定期调用容量断言。
  底层 provider 的显式容量检查 API 仍可用于诊断。

mjlab 的 `nan_detection` 是可选终止函数，标准速度任务没有启用；`NanGuard` 也是默认关闭的
调试记录器，即使打开也只记录、不抛异常或重置。当前标准 Gobot 路径不启用物理快照守卫。
过去加入的每步快照拷贝、有限速度阈值和相应全局中止已删除。

这些约定对齐异常处理，不代表物理过程不会失稳。有限但很大的数仍可能影响学习；
它们如何影响完整训练，应通过训练和评测衡量。

## 验证与运行

`tests/python/test_locomotion_numerics.py` 检查有限大速度不被额外终止、观测中的 NaN/Inf
不被清洗，以及 RSL-RL 在存储样本前拒绝非有限输出。
`tests/python/test_mjlab_numerics.py` 使用固定参考的实际奖励与观测处理代码验证数值语义。
CPU 与 Warp 共享同一规则，没有机器人名称分支。

默认八卡配置为 `examples/locomotion/configs/g1_velocity_8gpu.json`，输出到
`logs/g1_velocity_v3_gobot_seed42`。异常处理已改变，训练签名会拒绝旧运行的直接 `--resume`；
新实验使用新目录，不会覆盖已有 checkpoint。

本次保存的故障状态已在完整 Warp 环境重放：ImplicitFast 下最大速度约 3791 rad/s，
随后由普通跌倒条件结束该 episode 并重置，返回观测和奖励均通过 RSL-RL 输出检查。
这验证的是当前异常处理流程，物理轨迹本身仍存在发散。
