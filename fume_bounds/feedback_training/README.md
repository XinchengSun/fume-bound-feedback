# NUS 连续下界反馈训练

**真实运行已完成。** 12 组测试指标、固定终点与验证选轮对比、原始曲线及复跑核验见 [NUS 实测记录](records/20261110_nus/README.md)。

本阶段回答一个经验问题：在同一第 6 轮检查点上连续更新时，完整下界反馈是否比后验均值监督和停止标准差梯度带来更好的检索结果。独立概率认证继续按 P0a 管理；缺少认证池不阻止本阶段训练，但实验不产生逐对 95% 覆盖保证。

## 固定协议

| 项目 | 本轮设置 |
|---|---|
| 输入 | 原 NUS 10K HDF5、原版第 6 轮 best.pt；输入哈希沿用已核验重放 |
| 模型 | 冻结两塔前层，重新提取隐层；连续训练两塔末层，固定 W |
| 训练数据 | 全部 8,000 行，每轮每行恰好一次；不按置信度筛除样本 |
| 验证／测试 | 各 1,000 个查询及全部对应图库，双向余弦 mAP |
| 四组 | 原 FUME；FUME＋后验预测均值；FUME＋停止标准差梯度下界；FUME＋完整下界 |
| 预算 | 5 轮，batch 300；每组 27×5＝135 次连续更新 |
| Adam | lr=5e-5，betas=(0.5,0.999)；每组起点为空状态，后续步骤连续保留状态 |
| 反馈权重 | 固定为 1；原 FUME calc_loss 的 alpha=0.5，对比温度沿用默认值 1 |
| 配对次序 | 110104、110105、110106 三个批次顺序种子；四组共享每个种子的完整批次 |
| 支持 | 种子 41531 的固定 2,048 行组成 1,024 对；按当前末层重新拟合 Laplace 后验 |
| 选择 | 固定第 5 轮是主终点；仅按验证 mAP 选择的最佳轮次是次终点 |

总计 4 组×3 个顺序种子×135 步＝1,620 次真实 Adam 更新。三个种子改变的是批次顺序，不是三个独立编码器初始化或独立数据集。学习率和反馈权重不根据测试结果调整。

支持记录也参与本次编码器训练，因此全部 8,000 行均被保留。这与旧单步实验的反馈批次和支持记录不相交条件不同。本阶段按相同的新协议比较四组；不能把新旧实验的差异全部归因于训练步数。

## 一条命令从检查点运行

从仓库根目录执行，输出目录必须尚不存在：

```bash
python fume_bounds/feedback_training/run.py all \
  --checkpoint /absolute/path/to/best.pt \
  --dataset /absolute/path/to/nus_wide-10k_deep_doc2vec-corr-ae.h5py \
  --out /absolute/path/to/new_multistep_run
```

入口先验证受控输入及依赖源码，从检查点提取隐层，再运行四组训练及配对汇总。依赖、原模型和加载器见 [已验证重放说明](../certification/replay/README.md)。新提取入口保留相同前向，只核对初始验证 mAP，将旧提取脚本中的测试指标计算延后到全部训练与选择完成；两个入口的哈希分别登记。本阶段训练使用 CPU，不需要占用 GPU。没有输入时入口会明确失败，不会以随机数据代替 NUS。

已有本入口生成的特征包时，可使用 prepare/train 分步命令；参数细节以 `python fume_bounds/feedback_training/run.py --help` 及各子命令 `--help` 为准。

```bash
python fume_bounds/feedback_training/run.py prepare \
  --checkpoint /absolute/path/to/best.pt \
  --dataset /absolute/path/to/nus_wide-10k_deep_doc2vec-corr-ae.h5py \
  --out /absolute/path/to/new_extraction

python fume_bounds/feedback_training/run.py train \
  --features /absolute/path/to/verified_features.pt \
  --out /absolute/path/to/new_training

python fume_bounds/feedback_training/summarize.py \
  --run /absolute/path/to/new_training
```

## 关键数值与判断

先看固定第 5 轮，不通过挑选测试最高轮次报告收益。训练期间只使用验证集选轮；初始、固定终点和验证最佳模型的测试指标在各自训练与选择完成后计算。

| 比较 | 要回答的问题 |
|---|---|
| full − predictive_mean | 完整下界是否超出普通后验均值监督 |
| full − stop_sd | 标准差梯度在累计更新中是否有额外价值 |
| stop_sd − predictive_mean | 下界改变梯度权重是否有价值 |
| 三个反馈组分别 − original | 新反馈是否超出继续训练原 FUME 的收益 |

每项按相同种子逐一配对，报告双向、I→T、T→I mAP 的百分点差、均值、最小／最大值及胜负次数。原始 mAP 存为 0–1，汇总差值乘 100 得百分点；不得再乘 100 或将其当相对百分比。

Brier、NLL 和十箱 ECE 在测试全自然配对上计算，越小越好。正例匹配下界、负例不匹配下界、平均区间宽度、AU、EU 用于机制诊断；不按这些量变小直接判断方法成功。所有区间与 AU/EU 来自当前 Laplace 模型，未经真实逐对覆盖认证。

如果 mAP 没有改善而概率评分改善，结论仍应分别报告。如果效果依赖某一个顺序种子，保留全部种子的结果，不挑选最佳运行。三次顺序变化只支持固定检查点的敏感性分析，不据此声称跨模型稳健性或总体显著性。

本轮没有新增独立支持观测。下界上移或 EU 降低只能解释为训练后的模型读出变化，不能直接写成“新增证据减少了认知不确定性”。

## 输出与核验

- 每步训练损失、逐轮验证轨迹、完整批次索引和参数设置。
- 原始终点评测 `final.csv`，含原始两方向 mAP、概率评分、端点和熵分解。
- 权重及 Adam 状态、随机状态和协议哈希，便于核对连续更新。
- `summary/per_seed.csv`、`summary/paired_summary.csv`、`summary/summary.json`。
- 合成接线验证明确标注 synthetic，真实 NUS 未运行时不得填写替代指标。

```bash
python fume_bounds/feedback_training/run.py smoke --out /absolute/path/to/new_smoke_run
python -m unittest discover -s fume_bounds/tests -p 'test_feedback_training_*.py' -v
```

smoke 只验证多步优化、各组隔离、保存和汇总链路，不能用其 mAP 作为 NUS 实验结果。本目录 [validation.json](validation.json) 记录本次实际验证及真实执行入口的状态。
