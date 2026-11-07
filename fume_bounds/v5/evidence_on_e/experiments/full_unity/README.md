# 全部五项权重为 1：对比学习＋分类监督＋Ed（E＋Mix＋Noise）

## 目标与损失

**这是用户明确指定的固定权重实验，绝对不能使用旧版默认的对比0.05、PairCE0。**

L_total =
  1 * L_CL
+ 1 * L_Cls
+ (1 * L_E + 1 * L_Mix + 1 * L_Noise)

| 项 | 严格定义 | 权重 |
|---|---|---:|
| L_CL | 原 FUME 双向对比损失，归一化图文向量，tau=1 | **1.0** |
| L_Cls | E 基线的 ReLU 隶属度：两路 mean(||ReLU(zW)-y||_2) 之和 | **1.0** |
| Ed / L_E | 未经 Mixup/Noise 的原始正负图文 Pair 证据期望 CE | **1.0** |
| Ed / L_Mix | Mixup 图文对的证据期望 CE | **1.0** |
| Ed / L_Noise | Noise 图文对的证据期望 CE | **1.0** |
| KL | 错误证据 KL，完全关闭 | **0** |

**区别**：Ed 内的 L_E 代表原始 Pair 的证据期望 CE，不是历史 E 基线。后者的监督项是 L_Cls。

本次把历史 E 中的对比学习系数从0.05改成1，扩大20倍。同时把原始 Pair CE 从0改成1，KL从此前的0.01关掉。因此历史 E (测试 mAP 56.1806%) 和历史 E＋匹配 (54.7287%) 只能作为参照，**与本次对比不能只把差值归因于证据学习**。如果要分离证据影响，需要下一轮有 CL=1 的纯 E 对照；这轮不要自动启动其他组。

## 网络与监督

保留原 FUME 双塔、固定 W、最后层归一化得到 z、ReLU 隶属度监督与原双向对比损失，图像4096、文本1000、隐藏4096、4隐藏层、输出100维。

**不使用 BCE(hW)、必要性、可信度 c、类别证据头或 KL。**

匹配证据头保持旧 D 结构：拼接图像 z、文本 z、绝对差、逐元素乘积、内积，经过 Linear(401,128)→ReLU→Linear(128,2)→Softplus+1，输出匹配 Beta 浓度。原始正负标签由完整类别交集确定，默认每 anchor 采1个正对及1个负对；允许非原始对应正例。Mixup 采用 Beta(10,10)，Noise 采用 Beta(1,1)，噪声幅度以训练集 RMS 定标；后两项复用同一个证据期望 CE。原始 Pair CE 本次直接监督硬关系标签，明确启用，不再是0。

每个原始 batch 只调用一次 Adam.step，辅助头与图文双塔共同接收梯度，不冻结双塔，不删样本、不筛置信度、不按匹配头重排。正式检索仍按 L2 单位 z 的余弦排序。

## GPU1 的固定运行条件

- seed2，batch300；训练样本9000，150轮，30步/轮，共4500步。
- 同原历史 A/E 的双塔及 W 初值、每轮原始样本排列。使用 require-ab150 自动比对训练集 SHA、原始模型文件 SHA、初始化哈希及逐轮顺序。
- lr=5e-5，Adam betas(0.5,0.999)，weight_decay0，无 scheduler、无裁剪、CL温度1。
- 每轮都写训练分项损失、实际学习率、两塔 h 范数、验证双向 mAP、时长；每轮首批记录两塔输出层各损失的加权梯度范数和相对类别监督的夹角。
- 第150轮模型一次性评测全部4366个图像与文本查询及候选。不能拿验证最好轮代替。
- 保留证据 Brier、NLL、模型定义的认知/偶然熵及错误检测 AUROC，注明事件是抽样图文配对。不能说不确定性自然等于真实认知/偶然。

## 拉取与启动

仓库根目录：

    git pull --ff-only origin main
    cd fume_bounds/v5
    python -m pytest -q evidence_on_e/tests/test_on_e.py evidence_on_e/tests/test_full_unity.py

将 GPU1 物理 UUID 核实为原记录的4090后：

    CUDA_VISIBLE_DEVICES=GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35 \
      bash evidence_on_e/experiments/full_unity/run_full_unity.sh

此脚本显式传入 --cl-weight 1、--pair-clean-weight 1、--vicinal 1、--noise 1、--error-kl 0。直接运行未显式设置这些参数的旧默认入口**不是本轮配置**。

默认数据位置：
/inspire/ssd/project/video-generation/public/sunxincheng/FUME/local_ops/inria_bound_20261101/frozen_fume/datasets/INRIA-Websearch.mat

默认新的独立输出目录：
/inspire/ssd/project/video-generation/public/sunxincheng/FUME/runs/E_matching_full_unity_seed2_150_20261107

如果需要，自定义数据和全新输出目录可作为脚本两个位置参数。拒绝覆盖任何旧目录和日志。

## 完成时交付

- 每轮原始 history.jsonl、validation.jsonl、gradient_probe.jsonl、order.jsonl。
- provenance.json：解析后超参、源码哈希、初始双塔/W哈希、数据SHA；complete.json / last.pt。
- 固定第150轮 test_metrics.json，含双向 mAP 和匹配证据诊断。
- full_unity_raw_epochs.csv：150行实际记录（不平滑）。
- full_unity_verification.json：逐轮核验五项权重等于1、KL=0、总损失确实是五项之和、每轮9000样本、150轮4500步、顺序相同，并给相对历史 E、E＋匹配两组差值。
- 完整终端输出日志。没有记录的数值绝不补造。

## 源码变更边界

原 E minimal_objective 不修改。仅在 core.objective 中新增 cl_weight 可选参数，默认仍为0.05；run.py 增加 --cl-weight，默认也为0.05；只有本脚本显式传1时，组合成 L_Cls + 1*L_CL。**不会在原0.05的 E 总损失基础上再叠加1份对比学习**。原 M1 和旧 v5 的默认超参保持不变；历史 checkpoint 的源文件 SHA 校验可能拒绝在新版本下原样续训，应固定历史提交进行恢复。

当前对话没有连接用户 GPU1 的直接 SSH/shell 工具，**GitHub 代码提交不代表训练已经启动**；执行端应实际运行并验收后才报告真实性能。
