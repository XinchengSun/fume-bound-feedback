# v5：多标签证据学习，替换 fuzzy 分支

这是 `fume_bounds/v5/` 的独立入口。旧 `vls_transfer/`、v3/v4 均保留，不覆盖历史实验。
**不是“完整 FUME + 配对证据头”：本版不计算必要性、可信度 c、fuzzy 损失或配对证据。**

## 固定下来的设计

- 复用 FUME 两路 MLP 的层数、宽度、ReLU、无偏置最终层、固定方形正交 W；复用原对比目标的准确倍数。
- 检索用归一化 `z`；多标签类别读出用归一化前 `h @ W`，避免单位球面幅度限制多个类别同时接近1。
- `r_k = sigmoid(a_k)` 是证据方向，不是最终类别概率。
- 每个模态新增 `Linear(4096,1)` 输出共享证据总量 `s = softplus(b)`。
- **保留单位先验**：`alpha_k = 1+s*r_k`，`beta_k = 1+s*(1-r_k)`。
- 最终类别概率是 `p_k = alpha_k/(alpha_k+beta_k)`。多个类别可同时高，不跨类别归一化。
- 原始 multi-hot 直接监督最终概率；Mixup 与噪声输入采用逐标签 Beta 期望交叉熵。
- 同一样本各标签共享 `s` 是这次已讨论的轻量参数约束，并非原 VLS 的默认结构；它限制了独立逐类证据强度，不能宣称表达能力不受影响。

完整激活、公式、梯度、复用/修改边界见 [METHOD.md](METHOD.md)。改动与验证见 [CHANGELOG.md](CHANGELOG.md)、[TEST_REPORT.md](TEST_REPORT.md)。

## 安装与测试

```bash
git pull --ff-only origin main
cd fume_bounds/v5
pip install -r requirements.txt
python -m pytest -q
```

PyTorch 的 CPU/GPU 构建请沿用服务器已有环境；这里不强制重装特定 CUDA 版本。
已实际测试环境见 TEST_REPORT，不保证所有满足最低版本声明的组合都经过测试。

## 输入数据：不改划分，不对多标签取 argmax

支持两种文件：

1. 原 INRIA MATLAB 文件字段：
   - train：`tr_img`, `tr_txt`, `tr_img_lab`
   - val：`val_img`, `val_txt`, `val_img_lab`
   - test：`te_img`, `te_txt`, `te_img_lab`
2. NPZ，使用 `train_image/train_text/train_labels`，以及 `val_*`、`test_*`。

特征必须是 `[N,D]`。标签为 `[N,K]` 的0/1矩阵时直接保留，可有多个1；不除以每行标签数。
若标签是类别编号，必须显式给 `--classes K --label-base 0` 或 `--label-base 1`，不猜测编号基准。
没有正标签的行会报错，请修正/说明标注协议，程序不会偷偷丢样本。
所有划分必须存在，不能把 test 冒充 val。MATLAB v7.3 仅在 HDF5 带 MATLAB 标记时明确转置。

### 从头训练（主实验默认）

```bash
python train.py \
  --data /path/to/INRIA-Websearch.mat \
  --classes 100 \
  --epochs 100 \
  --batch-size 300 \
  --lr 5e-5 \
  --device cuda \
  --output /path/to/runs/v5_inria_seed2
```

若文件存的是1起始编号而不是标签矩阵，再加 `--label-base 1`。
读取的是你提供文件中的真实数据；本仓库不附带数据集或训练权重。
**训练脚本已改为这个入口，不要继续执行旧 `train_fume_vls.py` 来运行 v5。**

默认：隐藏宽度4096、4个ReLU隐藏层、d=K、seed2、Adam betas=(0.5,0.999)、无weight decay、lr=5e-5、对比温度1、对比权重0.05。
这些来自 FUME INRIA 设置/主训练脚本，是复用起点，不是已针对新损失调好的最优配置。
BCE、VLS混合、VLS噪声的默认权重各1；Mixup Beta(10,10)，噪声比例 Beta(1,1)。
噪声标准差按各模态训练特征的RMS固定，不改变特征本身。`--noise-multiplier` 控制额外倍率。

### 调参/消融

```bash
# 无虚拟证据监督：仍保留同结构的原始类别监督与对比
python train.py ... --vicinal-weight 0 --noise-weight 0
# 只移除噪声或只移除Mixup
python train.py ... --noise-weight 0
python train.py ... --vicinal-weight 0
# 权重、混合与扰动尺度
python train.py ... --vicinal-weight 0.5 --noise-weight 0.5 --noise-multiplier 0.3
```

仅在验证集上选参数；测试集不要用于反复选配置。初版固定末轮checkpoint，训练内不按test选模型。
`--no-test` 跳过末轮测试，适合调参。测试默认只在训练完成后执行一次。

### 真正的断点恢复

```bash
python train.py --data /path/to/INRIA-Websearch.mat \
  --resume /path/to/old_run/last.pt --epochs 120 \
  --device cuda --output /path/to/new_resume_run
```

恢复模型、W、两个证据层、Adam、随机状态及原训练超参。`--epochs` 是总轮数。
恢复时不允许同时重设训练超参，且校验数据文件SHA。输出必须是新目录，不覆盖旧日志。
CPU固定环境下已测试：1轮+恢复第2轮与连续2轮逐参数一致；跨硬件不承诺逐bit一致。

### 原 FUME 权重作为明确的辅助实验

`--warmstart /path/to/encoder.pt` 仅载入两路编码器和 `fixed_W`，新证据层重新初始化。
要求 `state_dict` 或 `fume_state` 及完整 `fixed_W`，不兼容时明确报错。
**若权重已受 fuzzy 损失训练，这不是“没有继承 fuzzy 知识”的从头实验。**主结果默认不使用它。
`--warmstart` 和 `--resume` 含义不同、不可同时使用。旧配对证据头不迁移。

## 评测

```bash
python evaluate.py --data /path/to/INRIA-Websearch.mat \
  --checkpoint /path/to/run/last.pt --split val \
  --device cuda --output /path/to/new_val_report
```

最终测试改 `--split test`。必须为v5 checkpoint，旧配对版本和仅encoder文件不会被当成已训练v5。

输出：
- `last.pt`：模型/固定W/证据层/Adam/RNG/配置/数据指纹。
- `history.jsonl`：各损失、两路平均证据、h范数、每轮实际样本数。
- `provenance.json`：代码哈希、上游版本、数据哈希、训练设置、warmstart来源。
- `test/metrics.json`：全量I→T/T→I余弦mAP；两路逐标签Brier、NLL、ECE、AP、错误检测、AURC。
- `image_predictions.npz`、`text_predictions.npz`：全部标签、z、alpha/beta、最终概率、TU/AU/EU/微分熵/vacuity。
- 风险—覆盖CSV：仅作曲线，不改变主mAP的查询与候选集合。

没有置信筛选，没有关系标签的正负配对扩展；mAP的相关性仍按真实多标签交集定义。
不确定性指标是模型定义的边缘量，不是实验已经测量出的真实AU/EU。没有导入旧配对OOD报告。

## 范围与局限

共享证据预算是一个明确的建模取舍，不是VLS完整复现。人工软标签不自动等于真实歧义。
VLS期望交叉熵可能继续提高浓度；单位先验并不保证认知估计正确。
保留的原对比项仍把未对齐样本作为负例，可能包含共享标签假负例。
本次没有运行真实INRIA训练、GPU训练或真实OOD，也没有性能提升声明。

## 可选的大网络实现检查

```bash
python smoke_full.py --output /path/to/new_full_smoke.json
```

该脚本用真实4096宽度/4层结构和合成输入完成一个Adam步，需要数GB内存，**不是数据集基准**。
