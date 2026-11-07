# 类别证据 / 图文匹配证据：互斥对照

基于 `e3a45df9ee075a015a6202d928948870ba29a4ae`。只新增实验入口，不覆盖此前 `train.py`、`run_experiment.py` 或任何已有模型/损失/评测代码。旧命令和旧 checkpoint 的语义不变。

## 1. 两种方案，到底替换什么

共同保留：FUME 同尺寸双 MLP、共享固定 W、隐藏 ReLU、L2 球面检索、原对比损失、完整 multi-hot 标签、Mixup 和噪声输入、证据+1、全量不筛选 mAP。

| 内容 | category | matching |
|---|---|---|
| 普通类别监督 | 完整 multi-hot | 完整 multi-hot，绝不删除 |
| 证据对象 | 每个模态的各语义标签 | 一张图与一段文本是否相关 |
| 证据头 | 既有方向+证据总量 | 4d+1关系向量 -> Linear128 -> ReLU -> Linear2 |
| 正负定义 | 类别存在/不存在 | 标签有交集/无交集 |
| Mixup软标签 | 类别向量逐位置插值 | 两个图文对的关系标量插值 |
| 噪声目标 | 各标签向0.5靠近 | 关系概率向0.5靠近 |
| 不确定性输出 | label-level Beta | pair-level Beta |

matching 不实例化 `img_evidence`/`text_evidence` 类别证据层；category 不实例化 `pair_head`。不是两个证据模块都加进去。两者都没有 FUME 必要性/c/fuzzy 损失。

## 2. 类别监督口径必须公开

旧 v5 的 clean BCE 监督类别 Beta 后验均值 `(1+s*r)/(2+s)`。删除类别证据层后，matching 不再有这个后验。

因此提供显式选项，不暗中改变旧版：

- `category --classification-source posterior`：直接委托旧 `experiment_objective()`，损失和梯度与旧版一致；新入口 category 默认是它。
- `category --classification-source direction`：仅 clean BCE 改为普通 `sigmoid(h@W)`，类别 VLS 与错误证据 KL 保持；这是明确命名的对照配置，不冒充旧版原样结果。
- `matching --classification-source direction`：普通 multi-hot BCE + 原CL + 匹配VLS/可选匹配错误证据KL；无类别证据损失。matching 默认自动采用 direction，拒绝 posterior。

`compare_evidence.sh` 的两个方案都显式使用 direction，使共同类别监督的函数、输入和目标相同。原来的 posterior 方案仍可作为第三个原样参照。两方案证据模型、监督单元和参数量本身仍不同，不能声称严格等容量或等计算量。

## 3. 匹配监督的实际计算

训练标签矩阵 Y 为 [B,K]，完整标签仍用于两路类别 BCE。匹配分支另外构造：

    R_ij = 1[ sum_k Y_ik * Y_jk > 0 ]

默认每条记录作为一个锚点，从有交集候选中随机采一个正例，从无交集候选中采 `pair_negatives=1` 个负例；锚点的图像/文本方向随机选择。支持非原始对应但语义相关的正例。`--pair-positive-mode aligned` 可只用原配正例作采样消融。

同标签候选绝不作为负例。没有可用负例的锚点仍保留正例，记录 `anchors_without_negative`；全部原始训练记录始终参与 clean 类别与对比训练。允许重复抽到同一关系，不将这些配对宣称为独立标注。

关系特征和参数：

    u = [zI, zT, abs(zI-zT), zI*zT, sum(zI*zT)]
    logits = Linear2(ReLU(Linear128(u)))
    alpha_match = 1 + Softplus(logits[:,0])
    beta_nonmatch = 1 + Softplus(logits[:,1])
    q_match = alpha_match / (alpha_match + beta_nonmatch)

方向：第0维=匹配，第1维=不匹配。没有把标签拼进头的输入，也没有把标签数值直接当作证据。

两个图文对 a,b 的 Mixup：

    xI_mix = lambda*xI_a + (1-lambda)*xI_b
    xT_mix = lambda*xT_a + (1-lambda)*xT_b
    r_mix = lambda*r_a + (1-lambda)*r_b
    lambda ~ Beta(10,10)

这是在各自原始特征空间混合，然后重新编码到球面；不是混合球面点。随机配对可出现正+正/负+负，不强制全部正+负。

噪声路径独立于 Mixup：

    xI_noise = rho*xI + (1-rho)*epsilonI
    xT_noise = rho*xT + (1-rho)*epsilonT
    r_noise = rho*r + (1-rho)*0.5
    rho ~ Beta(1,1)

噪声尺度复用训练集RMS和已有倍率。两路使用同一个 rho，独立高斯噪声。两种虚拟输入都回传图像编码器、文本编码器、匹配头。证据损失逐对采用：

    r*(digamma(alpha+beta)-digamma(alpha))
    +(1-r)*(digamma(alpha+beta)-digamma(beta))

关系软标签仍是构造的邻域监督，不宣称等于虚拟图文的真实相关概率。

## 4. KL、对比与计算量

错误证据KL复用 `wrong_evidence_kl()`，matching 使用原始0/1关系标签，category 使用原始multi-hot。两边都不对Mixup软目标套硬标签掩码。

matching 的 clean pair KL/可选 clean EDL 直接复用原始编码器输出；不增加此处编码器前向。类别信念一致性没有同义的匹配实现，所以 matching 显式拒绝非零 consistency，而不是悄悄关掉。

匹配池默认最多2B对。编码器样本量：category约3B/模态，matching约B+2B+2B=5B/模态。缺负例时减少。使用 `--pair-chunk 128` 分块反传；每块损失乘块大小/总对数，最终每batch只执行一次Adam.step。不能把“同epoch”写成“相同FLOPs”。

默认架构：category为122,396,674可训练参数，matching为122,440,194；新增匹配头51,714参数并移除两个类别证据层8,194参数。实际参数数和每轮时间写入记录。两方案用同seed时，编码器和固定W的初始哈希相同；独立DataLoader生成器防止配对采样改变下一轮原始数据顺序。

## 5. 运行

```bash
git pull --ff-only origin main
cd fume_bounds/v5
python -m pytest -q tests/test_evidence_ab.py
bash compare_evidence.sh /path/INRIA-Websearch.mat /path/new_comparison --device cuda
```

脚本串行运行两个方案，各30轮、seed2、batch300、同损失权重，输出必须新建。编号标签额外传 `--classes 100 --label-base 0` 或1；multi-hot矩阵不需转换。

只运行匹配方案：

```bash
python run_evidence_ab.py --data /path/INRIA-Websearch.mat --output /path/new_match \
  --evidence-task matching --epochs 30 --device cuda
```

原样类别目标参照（与旧版目标逐值逐梯度对齐，但运行器和新checkpoint格式不同）：

```bash
python run_evidence_ab.py --data /path/INRIA-Websearch.mat --output /path/new_category_legacy \
  --evidence-task category --classification-source posterior --epochs 30 --device cuda
```

续训：

```bash
python run_evidence_ab.py --data /path/INRIA-Websearch.mat --resume /path/new_match/last.pt \
  --epochs 100 --output /path/new_match_resumed --device cuda
```

续训恢复任务、全部参数、Adam、随机状态及独立DataLoader状态；禁止一边续训一边切换证据任务。旧v5 checkpoint继续用旧入口；只有明确的 `--warmstart` 接受满足格式的FUME编码器初始化，并记录非从头实验。

调参只使用val。配置确定后：

```bash
python run_evidence_ab.py --data /path/INRIA-Websearch.mat --evaluate test \
  --checkpoint /path/new_match/best_map.pt --output /path/final_match_test --device cuda
```

新权重只用新入口评测，不交给旧 `evaluate.py`。没有自动测试集筛选或调参。

## 6. 新增开关

- `--evidence-task category|matching`：互斥结构。
- `--classification-source posterior|direction`：类别BCE的明确口径；匹配只能direction。
- `--pair-hidden 128`：匹配MLP隐藏维。
- `--pair-negatives 1`：每个锚点采多少无交集负例。
- `--pair-positive-mode overlap|aligned`：语义相关正例还是仅原配正例。
- `--pair-chunk 128`：训练虚拟对分块，不改变全体均值损失。
- `--pair-clean-weight 0`：可选额外干净匹配EDL；不冒称VLS必需项。
- `--uq-pairs 20000`、`--uq-seed 20261107`：验证/测试配对的固定均匀抽样，0表示全部笛卡尔积。
- `--eval-pair-chunk 1024`：配对评测分块。

其余lr、batch、epochs、分类/对比/VLS/KL权重、KL退火、噪声倍率等在 `--help` 中；此新入口使用常数学习率以减少第一轮变量，没有悄悄套TCL的证据温度。

## 7. 怎样比较结果

共同指标：完全相同相关性定义、全查询/全候选的余弦mAP；两模态相同 `sigmoid(h@W)` 分类器的Brier/NLL/macro-AP。

各自原生可信性指标分开报告：category按“标签决策”，matching按“图文关系”。它们的Brier/NLL/AURC基准事件不同，不把数值大小直接当胜负。matching评测从完整笛卡尔池均匀无放回抽样，不复用训练时的正负平衡采样；保存索引和哈希。其概率受到训练采样先验影响，不能假定天然在自然检索分布上校准。

主mAP始终使用球面余弦排序，不切换到匹配头重排后再与类别方案比较。配对UQ抽样不改变检索分母。不宣称类别UQ等于匹配UQ，也不将负图文对当OOD。

`comparison.csv` 每组选择验证mAP最优的同一轮，列出该轮所有指标；不拼接各指标不同轮的最大值。测试集仅在配置固定后报告。
