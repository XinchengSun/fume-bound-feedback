# E 基础上的类别证据与匹配证据对照

## 研究定位与历史结果

这套代码固定使用已验证的 E 组普通监督作为公共基础：原版 FUME A 第150轮测试双向 mAP 56.9021%，E 56.1806%，B 的未归一化分类读出 BCE(h@W) 只有4.5839%。后续单因素实验 F 将 BCE 换为 Brier 后为2.8196%，G 保留 BCE 但改用归一化 z@W 后恢复至49.1959%（比 B 高44.6121个百分点）。这些数字是**历史运行结果，不是本目录的新实验**。

E 的成功说明需要优先保留原双塔、球面归一化、ReLU 类别隶属度监督和原对比目标。本目录不再将 BCE(h@W) 当作公共训练目标；两个证据机制在 E 上互斥选择，避免重新出现 B/C/D 的基础退化后，无法判断证据机制本身是否有用。

## 公共基础：原 E 目标严格不动

使用固定的归档原版模型文件：

    fume_bounds/v5/diagnostics/20261107_ab150/training_source/model.py

调用已经完成150轮验证的最小对照函数：

    fume_bounds/v5/minimal_supervision/minimal_control.py : minimal_objective

两路隐藏层：Linear -> ReLU；末层是无偏置 Linear；之后 L2 归一化得 z。固定正交映射 W 按原模型方式逐行归一化。类别隶属度 m=ReLU(z@W)。原始监督对两路分别计算逐样本欧氏误差，取均值后相加。保留原 FUME 两项对称对比损失、温度1、权重0.05，全部使用原始 z。没有必要性计算、可信度 c、fuzzy loss、BCE、余弦检索重排或置信度筛除。

数学表达：

    L_E = mean_i || ReLU(z_I[i] @ W) - y[i] ||_2
        + mean_i || ReLU(z_T[i] @ W) - y[i] ||_2
        + 0.05 * L_FUME_contrastive(z_I,z_T)

本轮接入承接 E 的实际 INRIA 配置，限制每条记录恰有一个正标签。其他 multi-hot 数据会直接拒绝，不会调用 argmax、归一化标签或静默丢弃数据。

## 三种互斥模式

| --task | 原始监督 | 辅助机制 |
|---|---|---|
| none | 完整 E | 无证据头、无虚拟样本，用于入口自检 |
| category | 完整 E | 类别存在/不存在的 Beta 证据 |
| matching（默认） | 完整 E | 图文相关/不相关的 Beta 证据 |

增加头在隔离 RNG 作用域内初始化；双塔、W、原始样本顺序不变。不能同时开启类别头和匹配头。类别头不会创建 pair MLP；匹配头不会创建类别浓度层。

### 匹配证据：沿用既有 D 方案

标签交集定义二元关系：r_ij = 1[y_i @ y_j > 0]。每个图文 anchor 默认采1个有交集正对和1个无交集负对；正对允许非原始对应图文。无可用负例时保留其正例，原始 E 训练样本不减少。

每个图文对的关系输入：

    u = concat(z_I,z_T,abs(z_I-z_T),z_I*z_T,sum(z_I*z_T))

证据头为 Linear(4d+1,128) -> ReLU -> Linear(128,2) -> Softplus + 1。输出 alpha,beta；匹配平均概率 alpha/(alpha+beta)。保持旧方案的参数化、激活、先验；默认 clean pair 期望交叉熵权重为0，可显式消融。

### 类别证据：沿用既有 v5 类别辅助定义

只用 hook 获取原网络的最后隐藏层及末层线性结果 h，不替换前向。

    r_k = sigmoid((h@W)_k)
    s = Softplus(Linear(hidden,1))
    alpha_k = 1+s*r_k
    beta_k  = 1+s*(1-r_k)

这里 Sigmoid 仅用于辅助证据，不再额外监督原始样本的 BCE。共享 s 使不同类别的证据强度耦合，是旧版 v5 的参数约束。由于辅助仍使用 h，可能影响 h 的幅度，必须监控范数和梯度，不预先保证安全。

### Mixup、噪声与损失

两种虚拟输入并行构造，始终先在预提取图像特征和文本特征中各自运算，再经相同双塔。两模态共用对应的混合系数：

    Mixup: lambda ~ Beta(10,10)
           x_mix=lambda*x_a+(1-lambda)*x_b
           t_mix=lambda*t_a+(1-lambda)*t_b

    Noise: rho ~ Beta(1,1)
           x_noise=rho*x+(1-rho)*Gaussian(0,sigma_modality^2 I)
           t_noise=rho*t+(1-rho)*0.5

category 的 t 表示每个类别是否存在；matching 的 t 表示图文是否相关。sigma_modality 只由训练集各模态特征 RMS 确定，不读取测试集统计。

Softplus+1 先验不变。证据期望 CE：

    EDL = mean[t*(digamma(alpha+beta)-digamma(alpha))
           +(1-t)*(digamma(alpha+beta)-digamma(beta))]

错误证据掩码 KL 只约束原始硬标签，不将软目标中的合理证据判成错误：

    q_wrong = beta if hard_label==1 else alpha
    KL_wrong = mean[log(q_wrong)-1+1/q_wrong]

默认 KL 权重最高0.01，10轮线性退火（第1轮为0，第10轮达到0.01）。总目标是：

    L = L_E + vicinal*EDL_mix + noise*EDL_noise
      + annealed_error_kl*KL_wrong + pair_clean_weight*EDL_clean_pair

最后一项仅 matching 可以开启，默认0。每个原始 batch 只进行一次 Adam.step()。pair 分块按真实样本数加权。没有梯度截断、冻结或旧 FUME 的可信度筛样本。

## 拉取与测试

在仓库根目录执行：

    git pull --ff-only origin main
    cd fume_bounds/v5
    python -m pip install -r requirements.txt
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q evidence_on_e/tests/test_on_e.py

## GPU1 实验（仅显式启动才会运行）

先核验实际 GPU1 UUID、仓库 checkout、设备显存和数据。服务器原数据：

    DATA=/inspire/ssd/project/video-generation/public/sunxincheng/FUME/local_ops/inria_bound_20261101/frozen_fume/datasets/INRIA-Websearch.mat

匹配证据单组：

    CUDA_VISIBLE_DEVICES=GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35 \
    python evidence_on_e/run.py --data "$DATA" \
      --output /inspire/ssd/project/video-generation/public/sunxincheng/FUME/runs/E_matching_seed2_150 \
      --task matching --classes 100 --label-base 0 --epochs 150 \
      --device cuda --require-ab150 --final-test

类别组只改 --task category 和输出目录。仅 E 对照使用 --task none。三组串行：

    CUDA_VISIBLE_DEVICES=GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35 \
    bash evidence_on_e/compare.sh "$DATA" /path/new_E_comparison \
      --classes 100 --label-base 0 --device cuda \
      --require-ab150 --epochs 150 --final-test

--require-ab150 校验数据 SHA、原模型文件 SHA、双塔/W 初值及150轮原始样本顺序。核验失败时不能绕开开关并仍声称严格同预算/同初值。已有实验的 GPU UUID 不代表当前设备映射一定未变。

## 参数清单

| 选项 | 默认 | 说明 |
|---|---|---|
| --task | matching | none / category / matching，互斥 |
| --vicinal、--noise | 1、1 | Mixup、加噪证据权重，0关闭 |
| --error-kl、--kl-anneal-epochs | 0.01、10 | 错误证据 KL 与退火 |
| --mix-beta、--noise-beta | 10、1 | 增强比例的 Beta 分布 |
| --noise-multiplier | 1 | 高斯噪声训练集 RMS 倍率 |
| --pair-negatives | 1 | 每 anchor 负例数 |
| --pair-positive-mode | overlap | overlap / aligned |
| --pair-hidden、--pair-chunk | 128、128 | pair MLP 和内存分块 |
| --pair-clean-weight | 0 | 原始抽样 pair CE，仅 matching |
| --initial-evidence | 2 | category 共享证据初值 |
| --seed、--batch-size、--lr、--layers | 2、300、5e-5、4 | 固定 E 训练设置 |
| --epochs、--save-every | 150、5 | 更新轮数和完整 checkpoint 间隔 |
| --uq-every、--uq-pairs、--uq-seed | 10、20000、20261107 | 辅助 UQ 评测频率、样本数及随机种子 |

原 E 的监督权重1、CL权重0.05、温度1、Adam betas(0.5,0.999)、weight_decay0和学习率固定不变。匹配版比类别版处理更多虚拟对：同轮数不等于同墙钟时间；记录实际秒数。

## 输出、评测与恢复

history.jsonl：监督、原始和加权对比、辅助损失、h 范数、epoch 时间、学习率。
gradient_probe.jsonl：逐轮首批的两路输出层梯度范数与夹角；不是整个模型或整个 epoch 的梯度。
validation.jsonl：每轮不筛选的 embedding 余弦 mAP，按设置定期计算模型报告的 AU/EU 辅助指标。
order.jsonl 与 provenance.json：数据、源码、初始化哈希、随机种子与原始样本顺序。
last.pt：模型、Adam、DataLoader 和全局 RNG 状态；拒绝覆盖已有输出目录。
test_metrics.json：仅 --final-test 时，在固定末轮计算全量测试结果。

续训只接受本入口的独立 checkpoint，并固定已保存的配置，拒绝静默热启动：

    python evidence_on_e/run.py --data "$DATA" --output /path/new_resumed \
      --resume /path/old_run/last.pt --epochs 150 --device cuda --final-test

两个证据任务的事件不同：类别证据评价每个类别是否存在，匹配证据评价抽样图文是否相关。抽样平衡正负对不代表真实图库先验。不应直接比较两类事件的 UQ 数值。AU/EU 是模型推断的熵分解，并非真实不确定性的直接测量。最终主指标始终是完整图库上的余弦 mAP，不允许用匹配头重排或丢弃样本。

## 验证范围

新增19项本地 CPU 合成测试通过：关闭证据后与 E 的损失、全部梯度和一次 Adam 更新逐张量一致，pair 采样边界、Mixup、噪声、+1 先验、Digamma CE、掩码 KL、分块加权、随机状态、训练续训和独立末轮评测。完整4096/1000输入、4隐藏层、100维、122,440,194参数的 matching 模型也完成合成 batch=2 的一步 CPU Adam。**尚未运行本版真实 INRIA/GPU 训练，也未证明新方案优于 E。**
