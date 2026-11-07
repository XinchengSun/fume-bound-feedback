# INRIA 四组 150 轮实验的原始材料

本目录归档了 2026-11-07 的实际训练驱动、恢复过程、全部逐轮记录、第 150 轮测试结果和同序验证前向。训练版本固定为 [2a8980a](https://github.com/XinchengSun/fume-bound-feedback/tree/2a8980a3ab85d35da68660db79813387016047c1/fume_bounds/v5)。四组均已完成，seed 2；9,000 条训练样本、1,332 条验证样本、4,366 条测试样本。

## 公共基线的实际执行路径

| 组 | 实际驱动 | 分类或证据监督 | 正式检索 |
|---|---|---|---|
| A 原版 FUME | compare150.py 的 run_original | 冻结源码的 FUME.forward 与 processor.calc_loss；模糊监督及权重 0.05 的对比项 | 两塔输出的单位向量 |
| B 普通 BCE | recover150.py 的 run_v5 → run_evidence_ab.py | category 架构保留；direction BCE 用 hW；vicinal、noise、KL 均置 0，consistency 默认 0，clean 对比项权重 0.05 | z = normalize(h) |
| C 类别证据 | 同 B 的训练入口 | 同一 direction BCE，加类别 Mixup、噪声和错误证据 KL | 同 B |
| D 匹配证据 | 同 B 的训练入口 | 同一 direction BCE，证据头改为预测图文对关系 | 同 B |

[首次执行的完整驱动](compare150.py)跑完 A；首次 B 因共享环境 NumPy 版本不足停在初始验证，完成的 B 参数更新数为 0。补齐项目依赖、切换到[实际恢复驱动](recover150.py)后，B、C、D 各从 seed 2 的初始参数训练到第 150 轮。A 的已有结果留存。四组共同双塔与 W 初始哈希、首批样本哈希见[验收回执](acceptance150.json)，每轮顺序见各组的 order.jsonl。[启动命令](launch/COMMANDS.md) · [两版驱动逐行差异](provenance/compare150_to_recover150.patch) · [实际使用的原版源码](training_source)

全部使用 batch 300，150 轮，每组 4,500 步 Adam，固定配置 lr=5e-5、betas=(0.5,0.999)。A 的完整调用和参数在 run_original、legacy_model、audited_loader 中；B/C/D 的完整解析后参数在各组 provenance.json 的 run_config 和 model_config。runtime 部分的 null 表示 CLI 使用默认项；解析后的值以 run_config 为准。B/C/D 的 BCE 对两模态各自的样本×类别取均值，再取两模态均值；对比项使用单位向量。C/D 的证据损失及权重以原始逐轮分项为据。[预定方案](PLAN150.md) · [B 运行配置](runs/B_bce/provenance.json) · [C 运行配置](runs/C_category/provenance.json) · [D 运行配置](runs/D_matching/provenance.json)

## 原始记录及缺项

- 各组 history.jsonl 保存实际训练轮均值，validation.jsonl 保存全查询双向验证 mAP，initial_validation.json 是更新前验证。B/C/D 保留实际写出的原始项、weighted_* 项及 KL 系数等字段；没有激活或没有写出的分项以该行原文为准。[四组原始日志](runs) · [四组合并的 600 行曲线](four_arms_raw_epochs.csv)
- A 模糊监督和对比损失的逐轮原始分项、加权分项 **未记录**，只有总损失。四组逐步或逐轮实时学习率、逐步梯度范数、逐批损失 **未记录**。配置中的固定学习率及最终 checkpoint 内的学习率分别留存；未用它们填造逐轮观测。
- 正式测试使用第 150 轮权重，按原始 embedding 余弦排序，每方向 4,366 个查询、4,366 个候选。A、B、C、D 双向均值分别为 56.9021%、4.5839%、4.3261%、4.8417%。A、AB 历史快照和最终四组结果均保留。[最终实测 JSON](test_map.json) · [B/C/D 独立复算](verification.json)
- 源码核对使用训练时的 A 验收哈希和 B/C/D provenance 中的 8 项源码哈希。另保存归档当日服务器 checkout 的 HEAD、status 和 tracked diff。训练时完整 git diff 快照以及仓库外驱动的启动瞬间哈希 **未记录**；本次已从服务器取回两份驱动实物，核对执行端对应副本字节相同。外部驱动不会出现在仓库 tracked diff 中。[核对回执](provenance/source_verification.json) · [服务器采集文件 SHA256](provenance/server_files.sha256)

## 第 150 轮验证集前向

[下载四组 NPZ](https://github.com/XinchengSun/fume-bound-feedback/releases/tag/inria-ab150-validation-20261107) · [各组字段、shape、checkpoint 和文件 SHA256](validation/manifest.json) · [CPU 导出脚本](validation/export_validation.py)

这些文件从同一验证 split 的全部 1,332 条样本重新前向得到，每组一个 NPZ。进程隐藏 CUDA，在 CPU 上使用 eval 和 inference_mode。sample_ids 是原验证集行号 0 到 1331；源文件没有提供单独的媒体 ID。四组 labels、sample_ids、W 逐元素相同，W 为 100×100。四组均保留 h_image/text、z_image/text，形状为 1332×100；h 是末层线性输出，z 是正式检索向量。标签是完整的 100 维 one-hot。

- B/C/D 的 logits 为 h@W。class_probability_image/text 是共同分类评测实际读取的 sigmoid(logits)。C 的类别 Beta 均值单列为 beta_posterior_mean_image/text。B 的类别证据支路未经监督；D 的匹配证据是图文对层级输出。
- A 用原模型前向，从末层 linear 的 hook 捕获 h。图像/文本输出沿用 view1/view2 的 membershipDegree、uncertainty；cred 是用原模型 get_test_category_credibility 对同一隶属度复算的类别可信度。
- 导出结果与实录逐方向核对，CPU 与原 GPU 的验证 mAP 最大绝对差低于 0.00071 个百分点。正式测试继续引用原运行结果。现有 test 预测文件对应不同的样本集合，不能充作此次验证前向。

## 用这些数组可以复核的现象

[原审计和复算脚本](audit) · [验证集读出与分布诊断](validation/readout_audit.json) · [生成脚本](validation/analyze_validation.py)

B 的验证集图像/文本分类 macro-AP 为 37.40% / 67.96%，真类排名中位数为第 3 / 第 1（并列取中间名次）。同一套 B 权重，正式单位向量的验证检索 mAP 为 5.74%，预测类别概率的余弦诊断为 26.65%。W 为固定方形近似正交矩阵；B 全验证集任意图文组合在 h 与 hW 空间的余弦最大绝对差约 9.5e-8。两种读出差异需要考虑 sigmoid 的非线性和已训练的坐标结构；向量整体长度无法单独解释余弦排序差异。

readout_audit.json 还包含同类非原配和异类的相似度分位数、池化的配对 AUROC、每模态 top-1、macro-AP 及真类排名。这些是验证集事后复算，逐轮训练时各目标对编码器梯度的占比没有记录，当前材料不能给出各项损失独立造成多少 mAP 变化。早期审计报告中的进度保留了当时的时间点；四组最终状态以本目录 status.json 为准。
