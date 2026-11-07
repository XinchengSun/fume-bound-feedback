# AB150 诊断与最小监督对照：统一交付入口

**这次只整理诊断并新增独立 E 组代码，不覆盖 A/B/C/D、旧 v5 或已归档结果。没有启动服务器训练。**

训练版本：`2a8980a3ab85d35da68660db79813387016047c1`。证据归档：`6de422a322eb5b47db7b5f5be2b81c40b55c1d16`。

## 先读什么

- [REVIEW.md](REVIEW.md)：本轮结论、证据位置、数学机制与尚未判明的问题。
- [minimal_control.py](minimal_control.py)：可直接运行的“只去掉必要性”对照。
- [check_mechanism.py](check_mechanism.py)：无需数据、checkpoint 或 GPU 的解析／合成检查。
- [mechanism_check.json](mechanism_check.json)：上一轮检查记录，含明确标为引用的 B 第3轮日志数值。
- [mechanism_check_recheck.json](mechanism_check_recheck.json)：本次重新执行的检查结果；不冒称重放了 Release NPZ。
- [TEST_REPORT.md](TEST_REPORT.md)：本次真实执行范围、测试结果与未验证事项。
- [原始四组归档](../diagnostics/20261107_ab150/README.md)：600轮日志、驱动、恢复过程、验证集清单。

## E 组究竟改什么

使用归档的原 FUME `training_source/model.py` 创建双塔与 W。保留隐藏层 ReLU、原始末层线性输出、原 L2 归一化位置、固定 W 的初始化和行归一化、原对比项的计算与 0.05 权重。

```text
图像／文本 → 原双塔 → 单位向量 z → ReLU(zW) = m
                                   ├─ 直接用完整标签监督 m
                                   └─ 原对比损失（权重0.05）
```

目标为：

$$L_E = \operatorname{mean}_i\|m_i^I-y_i\|_2 + \operatorname{mean}_i\|m_i^T-y_i\|_2 + 0.05L_{CL}.$$

不调用原 FUME.forward 的必要性／c／模糊不确定性计算；只调用其原始两个塔。没有 BCE、Sigmoid 分类、证据头、KL、Mixup、噪声、筛样本、改变检索读出。原 A/B/C/D 的 Mixup 和噪声代码没有删除；E 是另一个明确关闭它们的机制定位实验。

**唯一数值防护差别**：欧氏误差范数用 `torch.linalg.vector_norm` 计算，前向等价于原 `sqrt(sum(square))`；在误差恰好为0时采用有限的零子梯度，避免 sqrt(0) 反传 NaN。其余不增加裁剪、正则或归一化。

本次 INRIA 已核实每行恰有一个正标签。E 入口据此限制单正标签；发现 multi-hot 会明确报错，不进行 argmax 或标签归一化。此入口不是泛化的多标签替代方案，更不是已经验证有效的新方法。

## 拉取、检查与执行

在仓库根目录执行：

```bash
git pull --ff-only origin main
cd fume_bounds/v5
python -m pip install -r requirements.txt
python -m pytest -q minimal_supervision/tests/test_minimal.py
python minimal_supervision/check_mechanism.py
```

解析检查默认只打印；需要另存时：

```bash
python minimal_supervision/check_mechanism.py --output /path/to/NEW_mechanism_check.json
```

文件已存在会拒绝覆盖，不会改写已提交的检查记录。

### 在原 GPU1 上跑固定150轮 E 组

下面是**交给执行端的命令，不代表已运行**。GPU UUID 来自原验收记录，避免进程内 `cuda:0` 与物理编号混淆。

```bash
CUDA_VISIBLE_DEVICES=GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35 \
python minimal_supervision/minimal_control.py \
  --data /inspire/ssd/project/video-generation/public/sunxincheng/FUME/local_ops/inria_bound_20261101/frozen_fume/datasets/INRIA-Websearch.mat \
  --output /inspire/ssd/project/video-generation/public/sunxincheng/FUME/runs/E_minimal_no_necessity_seed2_20261107 \
  --classes 100 --label-base 0 --epochs 150 --batch-size 300 \
  --lr 5e-5 --layers 4 --seed 2 --device cuda --threads 2 \
  --require-ab150 --final-test
```

`--require-ab150` 在更新前核对原数据 SHA、原模型源码 SHA、双塔与 W 初始哈希，并在每轮迭代开始时核对该轮全部9000条记录的顺序和首批哈希。任何不一致都中止，不能静默改成近似相同。不同 PyTorch 环境若造成初始哈希不一致，应先恢复原实验环境，不要为了跑通关闭核验后仍声称同初值。

对照固定 Adam `(0.5,0.999)`、无 weight decay、无梯度裁剪、固定学习率，150轮4500步；末批不丢弃。每轮全量验证，**不选最佳 checkpoint**。`--final-test` 仅在最后一轮评测一次完整测试集；不带此参数则不测测试集。

输出目录必须是新目录。旧 A/B/C/D、checkpoint、测试结果均不覆盖。

## 新记录补了哪些缺项

- `history.jsonl`：总损失、直接隶属度监督、原对比损失、加权对比损失、每轮实际学习率、训练样本数与训练耗时。
- `gradient_probe.jsonl`：每轮首批，两塔最后线性层权重上的监督梯度范数、**加权**对比梯度范数和夹角余弦；是局部探针，不冒称全部参数梯度，也不是补造旧四组历史。
- `order.jsonl`：逐轮原始训练样本顺序及首批哈希。
- `validation.jsonl`：逐轮全查询余弦检索 mAP，无平滑、无筛选。
- `last.pt`：每5轮及结束保存模型、固定 W、Adam 和随机／loader状态。
- `test_metrics.json`：仅显式 `--final-test` 后生成，固定末轮。
- `provenance.json`：源码哈希、数据回执、初始权重哈希、设备、环境和完整训练设置。

续训使用新输出目录，仍须传入与原运行相同的训练设置，例如：

```bash
python minimal_supervision/minimal_control.py \
  --data /path/INRIA-Websearch.mat --classes 100 --label-base 0 \
  --resume /path/E_run/last.pt --epochs 150 \
  --output /path/E_run_resumed --device cuda --require-ab150 --final-test
```

不允许把 A/B/C/D 或旧 v5 checkpoint 当作 E 的续训状态。测试集不用于决定训练轮数、调参或选择读出。

## 执行端交付要求

只启动 E，不重跑或改写 A/B/C/D。先核对 `provenance.json` 的初值与数据核验；每轮分项、梯度探针与原始曲线保留。结束后按与 A 相同的固定末轮和全查询协议报告 E。若发生非有限数或校验失败，保留日志并停止，不偷偷放宽条件。

E 若维持明显高于 B 的检索，支持继续分解额外 BCE/无界读出的影响；若同样退化，则继续研究原必要性作用及 ReLU 监督本身。无论哪种结果，都不能预先许诺恢复 A 的 mAP，也不能自动改写 C/D。
