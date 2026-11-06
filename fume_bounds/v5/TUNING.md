# v5 可配置实验：先粗跑，再调参

基线：`8d9cb43`。新增入口 `run_experiment.py`；原 `train.py`、模型、损失、数据与指标模块不修改。
旧 v5 命令/权重仍使用原入口。新实验 checkpoint 有独立格式，禁止混着恢复。

## 1. 不变的部分

- 保留 FUME 双 MLP 的四层隐藏 ReLU、宽4096、最后无bias线性输出、固定正交W。
- 检索/对比使用归一化z，类别读出使用h@W；保留原对比计算及其倍数。
- multi-hot标签原样保留；Sigmoid方向、Softplus证据总量、两个单位先验(+1)保持。
- 原始监督对最终Beta均值做BCE；VLS两项使用逐标签期望CE。
- 训练噪声尺度只取训练集RMS；无样本筛除、无测试集调参、无标签argmax。
- 没有新增pair头、模糊必要性、可信度c、回滚或完整模型集成。

## 2. 新机制与来源

### 错误证据KL：仅原始硬multi-hot标签

参考 QinYang79/TCL 的 `2025-TIP-TCL/model.py::ce_loss`，固定源码 revision
`f9bb530db50bdd1f21c4a2aba0b8e5b059a5006b`；这里只借用带标签掩码的证据正则，
不搬用其候选Dirichlet、两个完整检索模型或多次优化器更新。

对单标签a=alpha、b=beta，y为0/1：

    a_mask = y + (1-y)*a
    b_mask = (1-y) + y*b
    L_KL = mean KL(Beta(a_mask,b_mask) || Beta(1,1))

y=1时只保留b接受KL；y=0时只保留a。因为其中一个参数恰好为1，
精确等价于 mean[log(q)-1+1/q]，q为错误一侧浓度。
代码使用 log1p(q-1)-(q-1)/q，避免大浓度下lgamma相减；已与PyTorch解析KL和梯度核对。
Soft目标直接拒绝：不把Mixup中本来合理的两侧支持都当作错误。
共享s时参数仍会相互耦合；“屏蔽正确证据的直接KL梯度”不等于正确类别完全不受共享参数更新影响。

退火：A>=2时，weight(epoch)=kl_weight*min(1,(epoch-1)/(A-1))。
第一轮为0，第A轮达到最大权重；A=0或1时从第一轮使用全权重。
这个日程和0.01粗跑权重是任务适配起点，不是照搬TCL最优参数。

### 证据共享方式

- shared（默认）：原来每个模态一个hidden->1预算层，同一输入所有标签共享s。
- per_label（可选）：预算层一次输出K个s_k；仍是两个塔，不是训练K套网络。

两种都使用 alpha=1+s*r、beta=1+s*(1-r)。不改激活，不删除+1。
100类/隐藏4096时，per_label比shared多811,206参数；不增加编码器前向次数。
这是一项结构消融，不宣称逐类一定优于共享。

### 意见一致性（默认关闭）

只比较同一条原始图文记录的同一个标签，b+=(alpha-1)/(alpha+beta)，
b-=(beta-1)/(alpha+beta)，使用两模态[b+,b-]的平均L1距离。
双方都反传，复用clean前向，一次总优化器更新。它不是TCL原双模型交替优化的复现。
两模态共同出错也可能一致；缺失模态、标签不对应的任务不能直接沿用。粗跑不启用。

### 总目标

    cls_weight * L_BCE
    + contrastive_weight * L_FUME_CL
    + vicinal_weight * L_mix_EDL
    + noise_weight * L_noise_EDL
    + annealed_kl_weight * L_wrong_evidence_KL
    + consistency_weight * L_belief_L1

各项均按当前代码规定平均，原CL的双对称项倍数保留。记录raw与weighted值。
KL和一致性复用原始输入前向，仍是原始/Mixup/噪声三个输入批；每批只调用一次Adam.step。

## 3. 第一遍：三组同预算粗跑

```bash
# 在仓库根目录先 git pull --ff-only origin main
cd fume_bounds/v5
python -m pip install -r requirements.txt
python -m pytest -q tests/test_experiment.py

# 各30轮、同seed=2、同batch；脚本依次运行，不后台调度
bash coarse_runs.sh /path/INRIA-Websearch.mat /path/runs/v5_coarse --device cuda
```

标签已为[N,K]矩阵不加label-base。编号标签必须额外明确 `--classes 100 --label-base 0` 或1，不能猜。
脚本的三组是：

| preset | 内容 | 用途 |
|---|---|---|
| control | multi-hot BCE + 原CL | 无VLS的同容量控制；不是完整原FUME |
| v5 | control + Mixup + noise | 原v5四项目标，新增模块全部关闭 |
| coarse | v5 + KL最大权重0.01，10轮退火 | 第一组带错误证据约束的候选 |

只运行一组也可以：

```bash
python run_experiment.py --data /path/INRIA-Websearch.mat --output /path/new_run \
  --preset coarse --epochs 30 --device cuda
```

不把30轮验证结果与历史100轮测试mAP直接比较。先看同预算val结果和训练趋势，再继续到100轮。
原FUME完整方法的对照需另行以相同数据/评测协议运行，不能给control改名叫FUME。

## 4. 参数清单

表中为新入口默认/粗跑值；均不是已在真实数据上调优的结论。

| 参数 | 默认/粗跑 | 第一阶段建议候选 |
|---|---:|---|
| --lr | 5e-5 | 2e-5,5e-5,1e-4 |
| --cls-weight | 1 | 先固定；之后0.5,1,2 |
| --vicinal-weight | 1 | 0.1,0.3,1 |
| --noise-weight | 1 | 0.1,0.3,1；消融0 |
| --contrastive-weight | 0.05 | 0.02,0.05,0.1 |
| --kl-weight | coarse=0.01；v5/control=0 | 0,0.001,0.01,0.03 |
| --kl-anneal-epochs | 10 | 5,10,20；0/1不退火 |
| --consistency-weight | 0 | 先固定0；后续0.001,0.01,0.05 |
| --mix-beta | 10 | 2,5,10 |
| --noise-beta | 1 | 0.5,1,2 |
| --noise-multiplier | 1 | 0.25,0.5,1 |
| --image-noise-multiplier | 1 | 先1，必要时0.5,1,2 |
| --text-noise-multiplier | 1 | 先1，必要时0.5,1,2 |
| --temperature | 1 | 对比温度，不是TCL证据温度；0.5,1,2 |
| --evidence-mode | shared | shared,per_label |
| --initial-evidence | 2 | 先2；后续1,2,5 |
| --batch-size | 300 | 150,300；显存不够再减，并重跑同batch对照 |
| --epochs | 30 | 粗跑30，完整100 |
| --weight-decay | 0 | 0,1e-5,1e-4 |
| --clip-grad | 0 | 0关闭；数值不稳定再比较2,5 |
| --scheduler | none | none,cosine |
| --schedule-epochs | 100 | cosine总日程；开始时确定，resume不可改 |
| --min-lr-ratio | 0.1 | cosine末端比率；0.01,0.1 |
| --seed | 2 | 先2，候选确定后2,3,4 |
| --hidden / --layers | 4096 / 4 | 第一阶段固定，避免与网络容量混淆 |

运行/选择参数：`--val-every 1`、`--save-every 5`、`--eval-batch-size 128`、
`--eval-chunk 128`、`--threads 2`、`--selection map`、`--patience 0`、`--min-delta 0`。
patience计数单位是验证次数；0不早停。map最大化验证平均mAP，nll最小化两模态平均NLL。
始终另存best_map和best_nll（只含推理所需状态），full last.pt保存optimizer/RNG用于续训。

不开放任意更换+1先验、跨类归一化、标签argmax或测试集调参；不把学习率和温度当同一参数。

## 5. 看哪些结果，而不是只盯mAP

- history.jsonl：分项raw/weighted loss、当前KL权重、证据均值/最大值、h范数、有效训练样本数。
- validation.jsonl：全量双向mAP、两路Brier/NLL/ECE、macro-AP、AU/EU/vacuity的错误检测与AURC。
- best_map.pt / best_nll.pt：由验证集分别选；比较可信性时必须报告同一checkpoint的检索指标。
- last.pt：每save-every轮及结束时保存；中断最多损失这一间隔内的进度。
- summary.json、provenance.json：选择记录、全部参数、输入和源码哈希。

高label_accuracy可能仅来自大量负标签，不能替代macro-AP/mAP或概率质量。
先排除NaN、损失不动、预测全部0.5、证据异常膨胀，再评价粗跑是否值得扩大搜索。
不预设固定的“下降几个点必定可救”阈值；需看同预算对照与验证趋势。

```bash
python summarize_experiments.py --root /path/runs/v5_coarse \
  --output /path/runs/v5_coarse/another_val_summary.csv --selection map
```

汇总只读取validation.jsonl，所有指标取被选中的同一轮；不拿每个指标各自最优值拼成一行。

## 6. 粗跑通过后：生成有预算上限的调参计划

```bash
python make_sweep.py --data /path/INRIA-Websearch.mat --root /path/runs/v5_search \
  --trials 16 --epochs 100 --seeds 2 --device cuda
# 只生成commands.sh与trials.json，不自动启动任何训练
bash /path/runs/v5_search/commands.sh
```

`tuning_grid.json`含5832个离散组合，但默认只随机抽16组，不穷举。先固定seed筛配置，
再对少量候选使用2/3/4验证稳定性。修改grid可分阶段控制搜索维度。
输出目录必须新建；任何已有结果都不覆盖。

## 7. 续训和最终测试

```bash
python run_experiment.py --data /path/INRIA-Websearch.mat --resume /path/run/last.pt \
  --epochs 100 --output /path/resumed_run --device cuda
```

续训恢复全部训练设置，不允许同时传lr/seed/结构等覆盖参数。它不是改超参后热启动。
cosine默认按预先声明的100轮日程；即使先跑30轮，再续到100轮，日程不重新开始。
验证前后保存/恢复随机状态，不让验证DataLoader改变训练采样序列。

配置确定后单独运行测试：

```bash
python run_experiment.py --data /path/INRIA-Websearch.mat --evaluate test \
  --checkpoint /path/run/best_map.pt --output /path/final_test --device cuda
```

新实验权重不能交给旧evaluate.py；旧v5权重仍用旧入口。本轮没有真实数据调参结果，
也没有证明AU/EU是真值。测试集只用于配置锁定后的报告，不能用它选择下一组参数。
