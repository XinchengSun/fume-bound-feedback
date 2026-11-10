# FUME bound-feedback

本仓库用于集中保存 FUME 下界反馈研究代码、接入说明、回归测试和版本变更，替代反复传递 ZIP。

## 2026-11-10 · 连续反馈训练入口

新增四组、5 轮、3 个共同批次顺序种子的连续末层训练，Adam 状态跨步保留，全 8,000 条训练行保留；固定第 5 轮为主终点，训练与验证选轮结束后才计算测试指标。独立认证数据不足不阻止这项经验训练。[训练方案与一条命令入口](fume_bounds/feedback_training/README.md)

代码已通过 9 项针对性测试和 96 步合成链路核验。**本次当前执行环境未接通原服务器，真实 NUS 新训练为 0／12 组，尚无新的检索指标。** [实际验证及阻塞记录](fume_bounds/feedback_training/validation.json)

## 2026-11-10 · NUS 实验与认证审计

**八批结果已从原第 6 轮检查点完整重放。** 178 行指标与 16 行梯度摘要全部复现；仅使用仓库 CSV 也能重算 24 行均值和 140 行配对统计。[源码、复跑命令与状态核验](fume_bounds/certification/replay/README.md) · [实体历史清单与随机观测机制说明](fume_bounds/certification/source_intake.md)

**独立认证停在数据审计阶段（P0a）。** 已登记的 10,000 条记录中，8,000 条用于编码器训练，1,000 条用于验证选轮，1,000 条用于历史测试。能确认独立的认证实体为 **0**，认证对为 **0**。仅截距金标准完成 222 个端点配置与 192 个精确覆盖网格点的核验。真实 NUS 概率区间、覆盖率及认证反馈的检索收益均未测。继续认证需要可按底层来源核验、事前隔离的实体池。[审计报告](fume_bounds/certification/records/20261110/REPORT.md) · [复核结论与交付缺口](fume_bounds/docs/updates/2026-11-10-p0a-review-240eb33.md) · [来源清单](fume_bounds/certification/records/20261110/provenance.json) · [审计代码与复跑命令](fume_bounds/certification/README.md)

**此前完成的 8 批反馈对照**使用同一第 6 轮检查点，每批独立从该起点更新一次。原 FUME 损失加完整下界反馈的平均测试双向 mAP 为 **58.89727%**，加后验均值监督为 **58.89874%**，配对差为 **−0.00147 个百分点**，完整下界在 8 批中有 3 批领先。[逐批指标与梯度摘要](fume_bounds/certification/records/20261110/mechanism/) · [各组平均](fume_bounds/certification/records/20261110/historical_averages.csv) · [配对差值](fume_bounds/certification/records/20261110/historical_paired.csv)

## 代码与版本

主要目录为 [fume_bounds/](fume_bounds/)，包括历史更新控制、参考族、v5 实验与认证审计。原始图文数据和大型权重保存在受控存储中。

```bash
git clone https://github.com/XinchengSun/fume-bound-feedback.git
cd fume-bound-feedback
# 后续更新
git pull --ff-only origin main
```

每次变更见 [`fume_bounds/CHANGELOG.md`](fume_bounds/CHANGELOG.md)。请按接入说明检查参考目标版本，不能把不同版本的下界数值直接比较成训练收益。

本仓库不声称已经完成真实 INRIA 上的性能提升；合成测试与真实实验必须区分记录。
