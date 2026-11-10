# 有效证据区间与实际排序验收原型

日期：2026-11-10（北京时间）。理论基线：[e8fd8e7 的完整推导](../theory/20261110_evidence_bounds/DERIVATION.md)。

**本模块把固定教师的置信集合、向外概率投影、信息半径和实际 cosine AP 差值接成可运行链路。正式受控运行已完成 32 个独立世界、96 条学生训练路径、14,784 次 Adam 更新。** 认证选择接受 218 次非零更新，真实条件期望 AP 下降 0 次；但最终期望 mAP 比 MAP 点估计选择低 0.09779 个百分点。因此结果支持证书可计算且不必拒绝所有更新，未显示优于点估计选择的平均性能。

完整数值、曲线与批次口径见 [正式运行报告](records/20261110_controlled/README.md)。这是已知生成模型的合成任务，不是 NUS 续训，也不是下界梯度反馈方法。

## 实现边界

- 教师：固定三维特征上的二元 logistic 模型，已知真参数范数上界，Gaussian-mixture 置信阈值，历史子水平集合交集。
- 数值投影：对每个线性 logit 方向求凸约束优化，再用非负乘子与球支持函数形成解析对偶下界。数值求解器的可行目标值不会被当成下界。显式浮点余量与最终概率的一 ULP 向外处理均有记录；这不是形式化区间算术。
- 不确定性：信息半径是候选概率区间的最坏对数损失遗憾；没有把它重命名为原后验 EU。
- 检索：两塔仍以实际 cosine 排序，全图库 AP、稳定并列顺序、全查询固定均匀权重。用同一概率机制下的新旧 AP 差值计算证书。
- 学生：二分类、二维单位嵌入、固定正交类别矩阵 W=I 的线性两塔。NumPy 损失是归档 FUME fuzzy 与 consistency 目标在该特例下的代数化简，已对照逐式前向及中央差分梯度。不是原 4096 维网络。
- 当前置信集合实现为二元教师；AP 概率盒函数支持 K 类输入，但尚未实现并验证十类 softmax 教师的完整置信集合。不能仅把 K 设为 10 就宣称 NUS 接入完成。

## 本轮比较的确切含义

每个学生只有一条持续的 Adam 候选生成轨迹。每个支持规模训练三个 epoch，保存三个候选快照，同一候选池交给三种规则：

1. ordinary_latest：直接采用第三个 epoch。
2. point_guard：用教师 MAP plug-in 概率，选择预测期望 AP 改进最大的候选；没有正改进则保留原快照。
3. certified_guard：用有效概率区间，选择最坏期望 AP 差值下界最大的候选；下界未超过固定数值门槛则保留原快照。

候选生成器不因验收而回滚，也不从所选旧快照恢复 Adam。它继续普通 FUME 损失训练。因此这是**共享候选轨迹上的部署快照选择**，不是三套反馈梯度训练，更不是上轮的后验预测均值监督训练臂。96 条轨迹对应 288 组策略视角，不能把后者计成独立训练。

## 固定数据协议

等先验类别 C∈{0,1}，canonical 特征满足 X|C ~ N((2C-1)μ,I)，μ=(1,0)，故真实条件类别概率为 sigmoid(2X₁)。两模态是同一 X 的事前固定可逆变换；教师使用的 canonical 特征包含学生能够获得的全部查询信息。

每个世界的 warm-up、支持流、图库、查询池独立生成。支持流按 128/512/2048 个实体前缀增加，每个实体的标签只计作一次证据；学生多轮训练不增加认证观测数。每库 64 个固定查询、128 个图库实体，查询与图库身份不相交。

学生初始化及批次随机流使用三个固定种子；所有策略共用。查询真实标签、真实类别概率和真参数仅在该世界全部支持规模、学生种子与策略选择冻结后进入评估函数。真实参数不参与区间求解或候选选择。

配置在生成正式数据前写入并哈希，见 [pilot_config.json](pilot_config.json) 与运行目录中的 config.json/run.json。该配置未经正式运行结果调参；另有两个不同随机世界的链路 smoke，不计入正式统计。

## 运行

验证环境为 Python 3.12.14、NumPy 2.3.5、SciPy 1.17.0、Matplotlib 3.10.8。这个模块不依赖 torch、h5py 或 pytest；版本清单在 [requirements.txt](requirements.txt)。

在仓库根目录执行，输出目录必须不存在：

~~~bash
python -m unittest discover -s fume_bounds/evidence_feedback/tests -v
python -m fume_bounds.evidence_feedback.run run --out /tmp/fume_evidence_new_run
python -m fume_bounds.evidence_feedback.replay --out /tmp/fume_evidence_new_run
python -m fume_bounds.evidence_feedback.analyze --out /tmp/fume_evidence_new_run
python -m fume_bounds.evidence_feedback.plot_results --out /tmp/fume_evidence_new_run
~~~

仅从已保存记录独立复核，无需训练：

~~~bash
python -m fume_bounds.evidence_feedback.replay --out fume_bounds/evidence_feedback/records/20261110_controlled
~~~

replay 从保存的模型权重独立重算 cosine 排名，使用累计相关数的向量公式重算 AP，使用二类仿射端点闭式重算差值证书，不调用 ranking.py 的 AP 或证书函数。它不重跑置信集合投影求解器，也不把数值复算替代统计定理。

## 真实 NUS 的接入状态

具体接口、当前数据与服务器访问事实、原配对测试的信息条件见 [REAL_DATA_INTAKE.md](REAL_DATA_INTAKE.md)。

原 12 组 NUS 训练负结果保留。当前受控实验没有验证真实 NUS 教师模型正确性，也没有改变原 1000 条配对测试的评测协议。获得原数据可以恢复经验训练，但真实认证还需要合适的观测机制、完整的条件概率对象和可用的多类置信集合。
