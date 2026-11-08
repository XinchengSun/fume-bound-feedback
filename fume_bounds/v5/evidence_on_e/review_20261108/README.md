# 匹配证据：先修监督构造，再验证 AU/EU（2026-11-08）

## 当前决定

不因为“MLP 层数多”就删掉匹配头，不改已验证的双塔、球面归一化、固定 W、ReLU 隶属度类别监督，不换激活，不加 KL。正式目标继续是：

    L_total = 1*L_CL + 1*L_Cls + Ed(1*L_E + 1*L_Mix + 1*L_Noise), KL=0

Ed 的 E 是原始正负 Pair 的证据期望 CE，不是旧 E 基线。原始 Pair 监督不得再次关闭。Pair/Mix/Noise 共用原证据头，输出仍是 Softplus+1 的 Beta 参数。

本目录只新增经过 CPU 测试的 Mixup 候选模块、数学反例和实验要求。**没有修改现有 core.py/run.py，没有替换主训练默认配置，也没有运行真实 INRIA/NUS 训练。**被核对源码：6fc7fd56baef92ce43f1a32a52c1fbc100035c80。

## 已复现的具体风险：双边 Pair Mixup 的端点标签冲突

现有 core.py::mixup 同时线性混合图像、文本，并线性混合这两个原始 Pair 的二元关系标签。

假设猫图 I_c、狗图 I_d、猫文本 T_c、狗文本 T_d，猫与狗标签无交集。

| 虚拟样本的来源 | lambda=0.5 的输入 | 旧目标 |
|---|---|---:|
| (I_c,T_c,1) 和 (I_d,T_d,1) | ((I_c+I_d)/2,(T_c+T_d)/2) | 1 |
| (I_c,T_d,0) 和 (I_d,T_c,0) | ((I_c+I_d)/2,(T_c+T_d)/2) | 0 |

相同虚拟输入的两个目标完全相反。这里不是模型“没有足够复杂的 MLP”，而是关系标签的来源无法只由混合后的观测决定。

**边界**：连续 Beta 分布恰好抽中0.5的概率为0。此反例证明端点插值规则可以产生相冲突的虚拟监督，并不说明训练中严格相同输入频繁出现，更不能量化它造成的 mAP 损失。接近中点时也可以构造任意接近的输入和相反的端点标签。同样，带随机标签的训练并非数学上非法；但这种人为制造的冲突，不能不经验证地宣称代表真实数据的偶然不确定性。

## 最小候选改法：固定查询，只混合候选

定义原始关系 r_ij = 1[labels_i 与 labels_j 有交集]。

图搜文时：

    image = I_i
    mixed_text = lambda*T_j + (1-lambda)*T_k
    target = lambda*r_ij + (1-lambda)*r_ik

文搜图时：

    mixed_image = lambda*I_i + (1-lambda)*I_k
    text = T_j
    target = lambda*r_ij + (1-lambda)*r_kj

例如固定猫图，猫文本和狗文本各混一半，目标0.5。反过来排列这两段文本并取互补系数，得到同一输入和同一目标。每个目标都引用**同一个查询**下的两项可核查原始关系，不把另一张图与另一段文本的关系当作该查询的监督。

这仍是邻域风险最小化的建模假设：输入线性插值对应条件关系目标的线性插值。可推导的目标是“随机选取两个已知候选时，关系标签的期望”；**不等于真实混合特征的条件概率已被测量出来，也不保证所有可能的特征碰撞都消失。**它解决上述双边来源切换的具体问题，而不宣称解决所有跨模态增强问题。

### 接口与明确的接入位置

query_mixup.py 接收原始 xi/xt/y，以及已经采样好的 i、j，不改变原始 Pair 的构造数量、正负比例。默认按 Pair 索引交替使用图像查询和文本查询，全部 Pair 都保留；Beta 混合参数仍为10。两条梯度路径都可反向传播，标签只用来构造目标，不输入证据头。

在现有 objective 中，在 i,j,target = sample_pairs(y,cfg) 之后，将 **仅 matching 模式的 vicinal 分支**替换为：

```python
from query_mixup import query_conditioned_mixup
# xi, xt, y 是原始 batch；i, j 是已由旧代码采样好的图文索引。
data = query_conditioned_mixup(xi, xt, y, i, j, beta=cfg.mix_beta)
# 后面的 pair 分块前向、expected_ce、backward 与 Adam.step 保持不变。
```

不是把这段修改直接应用到 category 模式。原始 E、Noise、CL、Cls、权重和 KL 设置全部保持。加入明确的 mix-mode 开关、新源码哈希、新输出目录，旧版作为对照；不能覆盖旧结果或称它“完全没有改设计”。

NUS 的执行端脚本尚未归档到当前 remote，本目录没有猜测其路径或改写服务器；集成时需要使用实际运行版，而不是误用仍限定单正标签 INRIA 的旧启动器。

## 第二个数学检查：软目标不足以识别认知不确定性

在现用 Beta 证据期望交叉熵中，固定 t=0.5、alpha=beta=S/2，只增大 S：

| S | 期望 CE | 模型 AU | 模型 EU |
|---:|---:|---:|---:|
| 4 | 0.833333 | 0.583333 | 0.109814 |
| 20 | 0.718771 | 0.668771 | 0.024376 |
| 200 | 0.695653 | 0.690653 | 0.002494 |
| 2000 | 0.693397 | 0.692897 | 0.000250 |

没有增加样本和知识、均值也未变，loss 仍下降，EU 同时趋近0。所有参数都符合“证据+1”（alpha,beta>=1）。因此“soft target + Beta 输出”本身不是正确 EU 的充分条件。此例不否定 VLS 的实验；原文 Theorem 3 是在特定浓度排序条件下，对浓度参数局部更新的分析，不能当作整个迁移模型全局识别正确 AU/EU 的保证。

## 下一轮实验：不再把层数变化、维度变化和新监督绑在一起

先固定当前实际使用的 NUS/INRIA 表示维度，不同时扫10/128维，不删40维输入、不改证据头、不改激活、不加 KL。主运行的五项系数仍全部为1。

1. 保留当前 full-Ed 作为 R0：旧双边 Mixup，CL+Cls+E+Mix+Noise（全1）。
2. R1：只将 Mix 改成上述固定查询规则，其余与 R0 完全相同。
3. 如已有可比原始 E-only 证据监督记录，可用作“有没有邻域监督”的消融；没有则不能把别的对比权重、别的头的结果混进来。关闭 Mix/Noise 的运行必须显式标为消融，不冒充全1主配置。

先复核相同初始化/原始样本排列、目标端点和梯度，再训练。R1 只能检验这个 Mixup 适配是否更合理，不能直接宣称整个认知/偶然不确定性问题被解决。

### 可信学习的验收必须对准动机

- **匹配概率**：在未改变正负先验的真实候选分布上报告 Brier、NLL、校准曲线，至少同时报告 raw 结果。若作校准，校准器只能拟合于独立 calibration split，两方法同规则处理。训练平衡抽对与图库分布不同，不把1:1样本上的校准当作图库概率校准。
- **偶然不确定性 AU**：用独立的、有已知条件匹配概率 eta 的可控双模态基准，对照预测 AU 与 H(eta)。真实数据没有多标注或真实条件概率时，错误检测/扰动实验只是代理证据，不能写成“恢复真实AU”。不能只在与训练相同的 Mixup 生成器上评价它自己制造的标签。
- **认知不确定性 EU**：固定数据生成过程和歧义水平，改变独立训练集的覆盖/规模，观察对新增知识的响应；是否减少模型与已知 eta 的偏离，EU 是否随知识增加而合理变化。不能把高斯加噪、低密度或高错误率自动当作纯 EU 真值。需要重复独立训练集或种子，否则无法把知识量与一次优化偶然性分开。
- **任务能力**：全部查询的双塔余弦 mAP，原始样本全保留。风险覆盖图可以作为额外诊断，但不能通过筛掉低置信样本来替代正式 mAP。

决策：若只改善 mAP，应如实称为检索辅助监督；若概率更准但 EU 对知识量不敏感，不声称改善了认知不确定性；只有相应对照支持才采用“更可靠地估计 AU/EU”的研究结论。不要通过删 MLP 或不断调参逃过这项验收。

## 代码与执行结果

- query_mixup.py：独立候选模块；不改生产默认值。
- tests/test_query_mixup.py：18项实际 CPU 测试，全部通过。
- check_mechanisms.py / CHECK_RESULTS.json：本端实际执行的两个数学检查。
- TEST_REPORT.md：精确环境和检查边界。

运行：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_query_mixup.py
    python check_mechanisms.py --output NEW_CHECK_RESULTS.json

没有新的真实数据训练结果。

## 依据

- 当前代码（端点混合与证据损失）：https://github.com/XinchengSun/fume-bound-feedback/blob/6fc7fd56baef92ce43f1a32a52c1fbc100035c80/fume_bounds/v5/evidence_on_e/core.py
- VLS 原论文，Sections 3.1–3.3 与 Theorem 3：https://proceedings.neurips.cc/paper_files/paper/2025/file/97b73904e88cc1dc0a3485595eda3753-Paper-Conference.pdf
- DECH 官方 EvidenceNet（确实含带 ReLU 的两层负证据子网；层数本身不是错误证据）：https://github.com/blackant-dev/DECH/blob/269568159760da45ade1d2ecc88ddec721a97f4c/models/Evidencenet.py
- 固定查询的公式是本次任务适配候选，不宣称是 VLS/DECH 原文给出的跨模态算法，也不宣称已证明新颖性。
