# FUME bound-feedback

本仓库用于集中保存 FUME 下界反馈研究代码、接入说明、回归测试和版本变更，替代反复传递 ZIP。

## 2026-11-10 · 下界、EU 与实际排序的理论核验

**已补齐一条假设明确的理论链条：有效观测模型 → 全时间参数置信集合 → 嵌套概率区间 → 最坏对数损失遗憾不增。** 对原后验 EU，仅在一致 Bayesian 更新下证明条件期望不增；不把它改名成逐路径保证。检索部分保留实际 cosine 排序，对同一概率机制下的新旧条件期望 AP 差值构造下界，只有证书通过才有改进保证。[完整推导与证明](fume_bounds/theory/20261110_evidence_bounds/DERIVATION.md)

推导给出预序似然比、可计算 logistic 集合、信息半径闭式和类别概率盒 AP 差值线性规划；同时给出直接下界损失不保证 EU 下降的反例。统计与信息论工具已有文献基础，不作为新定理宣称。**现有 NUS 配对实体、历史用途、教师表征变化及高维集合宽度尚未满足这一认证接口，未新增真实覆盖或 mAP 保证。**

六组数学核验已通过，包含独立计算路径、相关性反例、有限排序枚举和有限时域混合置信序列；这些检查不替代理论证明，也不是 NUS 训练。[核验脚本](fume_bounds/theory/20261110_evidence_bounds/verify_claims.py) · [机器可读结果](fume_bounds/theory/20261110_evidence_bounds/verification.json)

## 2026-11-10 · 连续反馈训练的负结果

**12 组真实 NUS 训练已完成，共 1,620 次更新。** 固定第 5 轮，完整下界双向 mAP 为 58.9663%，比同条件原 FUME 低 0.0642 个百分点，比后验均值监督低 0.0964 个百分点；两项均为 0 胜 3 负。完整下界组的平均区间宽度比起点增加 4.02%，EU 增加 10.22%，未实现整体不确定性收紧。[完整实测报告](fume_bounds/feedback_training/records/20261110_nus/README.md) · [原始记录独立复算](fume_bounds/feedback_training/review_4c7f97f.json)

**此前安排的六组 train_prior 权重对照暂停作为研究主线，真实训练仍为 0／6。** 改变类别权重不能修复概率对象、覆盖条件和检索目标之间的缺口。保留已有代码、预算与负结果，下一次接入按上述理论核对固定教师、有效证据及实际排序差值；不继续把直接下界损失的调参当作理论闭合。[历史方案及暂停说明](fume_bounds/feedback_training/WEIGHTING_CONTROL.md)

权重对照代码此前已通过 20 项测试、48 步两臂合成链路验证；旧 504 项配对汇总逐字段保持一致。这些属于上次代码验证，不是本轮新增训练。[历史代码验证](fume_bounds/feedback_training/weighting_control_validation.json)

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
