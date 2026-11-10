# FUME 下一阶段代码需求：冻结表征下的逐对概率认证

- 日期：2026-11-10（Asia/Shanghai）
- 状态：执行规格；本文件不表示认证模块已经实现，也不表示已在 NUS 获得概率覆盖保证。
- 目标仓库：XinchengSun/fume-bound-feedback
- 交付目录：fume_bounds/
- 本轮交付：需求文档、README 入口和 CHANGELOG。执行端按本规格另行实现并记录实测结果。
- 研究目标：保留全部样本，明确逐对匹配概率下界的依据；将统计有效性、区间紧致度和检索收益分别验收。
- P0 边界：一个冻结 checkpoint、一套固定 pair features、一个静态 split-likelihood 集合。完成后再决定下一阶段。

## 1. 为什么转向这一任务

2026-11-10 用户提供的 NUS 报告已经完成 8 个批次的配对单步机制比较。每批从同一 FUME 第 6 轮 checkpoint 出发，使用 300 个实体，正负反馈权重各半，只更新两塔最后一层一次，测试使用全部 1,000 个查询。

| 反馈 | 单用反馈的双向 mAP | FUME 加该反馈的双向 mAP |
| --- | ---: | ---: |
| 普通逻辑回归点预测，无先验 | 58.88172% | 58.89679% |
| 后验平均 logit 的点预测 | 58.88615% | 58.89767% |
| 后验预测均值监督 | 58.88795% | 58.89874% |
| 下界反馈，停止标准差梯度 | 58.91228% | 58.90794% |
| 完整下界反馈 | 58.88580% | 58.89727% |
| 仅 FUME 原损失 | — | 58.89967% |

冻结起点为 58.90969%。完整下界相对后验均值监督的组合损失结果平均低 0.00147 个百分点，8 批中 3 批领先。相对停止标准差梯度，单用完整反馈平均低 0.02648 个百分点，8 批全部落后；组合使用时平均低 0.01067 个百分点，8 批中 6 批落后。组合完整下界的 Brier、NLL、十箱 ECE 相对组合均值监督分别平均降低约 0.000026、0.000129、0.000071。

这些数字的来源状态为 user_reported_not_reproduced_in_this_requirements_change。本次编写需求没有访问用户 D: 路径中的 CSV 或重新执行上述实验。

执行端应登记已有代码、配置、批次索引、paired.csv、averages.csv 的路径和 SHA-256，并导入其原始汇总。除非发现文件不一致或需要定位实现错误，不重复这 8 批消融。8 批共享 checkpoint、支持库和测试集，不能登记为 8 份独立训练库或 8 个训练种子。

本阶段要回答：

> 在明确的概率对象、独立认证数据与模型假设下，能否构造数值可靠、且在实际预算下足够有用的逐对概率界？

## 2. P0 范围及最终结论格式

P0 分为两个顺序阶段，上一阶段通过后自动继续，不新增人为批准环节：

- P0a：先审计事件语义、历史实体使用、可用独立认证池、现有特征维数/秩与所需求解依赖；实现并运行必要的标量金标准。标量运行预算 5 分钟 CPU。若来源或目标语义不成立，交付 blocked 报告与已完成检查，到此结束本轮执行，不建设完整高维证书后端。
- P0b：P0a 已找到满足协议的数据和可执行技术路径后，再实现完整对偶验证、1,000 次覆盖模拟与 NUS 小规模审计。若发现现有特征维数使后端不可行，报告具体瓶颈，不自行降维或改变概率对象。

P0 必须分别输出四个状态：

1. implementation_correct：集合、投影及数值处理是否按定义实现。
2. model_conditional_validity：给定明确假设时，统计保证是否成立；真实数据的假设是否具备可审计来源。
3. practical_informativeness：区间是否足够紧，是否能支持预定义的检索比较，计算成本是否可接受。
4. retrieval_gain：是否验证了检索收益。本阶段没有反馈训练；不得由前三项自动写成 improved。

P0 保留已有四项读出和 Laplace 代码作为比较来源。以下内容留到 P1：

- 蒸馏新的认证端点、重新训练学生或扩大完整反馈训练。
- 更新编码器、将认证端点反传。
- 对认证数据反复选模型、调权重、调置信水平。
- cross-fitting、序贯置信序列、多 checkpoint 自适应选择。
- 大规模超参数搜索和全部百万配对逐一投影。
- 将新集合拟合回 Beta，或根据集合宽度重新定义 AU/EU。

## 3. 概率对象必须先写清

令 x 表示原始图文对，Y 表示按固定规则产生的二元匹配事件。原研究目标保留为：

$$
p^*(x)=P^*(Y=1\mid X=x).
$$

冻结配对特征 v(x)，必要时包含截距。候选族为：

$$
p_\beta(1\mid x)=\sigma(v(x)^\top\beta),\qquad \beta\in\mathbb R^d.
$$

本阶段理论使用以下强正确设定假设：

$$
p^*(x)=\sigma(v(x)^\top\beta^*)
$$

对目标总体中的 x 成立。它同时包含链接函数正确和特征足以表达目标条件概率的假设。此假设标为 assumed_not_proven，不能因实体隔离、低 ECE 或通过合成测试改为 proven_on_NUS。

必须生成 estimand.md 和机器可读 event_definition.json，写明：

- Y 的具体计算规则、标签版本及多标签匹配约定。
- 随机性来自哪个总体和哪种标签/语义观测机制。
- X 包含哪些信息，不能把待预测标签或其等价信息放进条件变量。
- v(x) 的精确定义、维数、截距、标准化、归一化、拟合来源。
- 原始图文对条件概率与仅条件于特征的 P(Y=1|v(X)) 的区别。
- 若实际只能支持后一个对象，明确输出 weaker_scope；不得当作原目标已经完成。
- 若 Y 被定义为给定完整数据记录后完全确定的数据库事件，必须指出对应概率为 0/1 的语义问题，不能自动宣称有限参数 logistic 已正确设定。

logistic 优化对参数是凸问题，不意味着其概率密度族对混合封闭；不得据此自动套用模型错设下覆盖 KL 投影的特殊结果。

代码不应猜测这些定义。缺失时只完成 P0a 来源/语义报告和必要标量金标准，真实 NUS 认证状态写 target_semantics_unresolved；完整后端与 1,000 次模拟不列为本次阻塞交付的强制任务。

## 4. 数据来源和独立性检查

### 4.1 记录所有数据角色

manifest 至少分别记录：

- encoder_fit_entities：编码器训练用过的实体。
- checkpoint_selection_entities：选 epoch、调参用过的实体。
- head_fit_entities：本轮概率头拟合用的实体。
- cert_entities：认证数据实体。
- test_query_entities：既定测试查询实体。
- gallery_entities：既定检索图库实体及其与其他角色的关系。

同一图片的其他文本、其他裁剪或其他 ID 别名，应归入同一个底层实体块。必须追踪 image/text 两模态，不能只检查 pair_id 是否重复。

认证实体不得参与编码器训练、选轮、特征标准化拟合、概率头拟合、方法选择或超参数选择。历史指标若影响了后续方法设计，也应登记其数据来源。已多次查看的原测试集可以继续作为既定描述性基准，但不能改称全新确认性测试集。测试查询不得用于认证或调参。图库沿用既定检索协议，原本与训练集的关系要记录；不能因为图库与查询组合产生很多行，就把这些行当独立数据。

冻结 epoch 6 不能消除历史使用。若无法提供干净来源：

- 禁止输出“独立认证已完成”。
- 标为 provenance_blocked 或 exploratory_only。
- 完成 P0a 中不依赖这些条件的来源报告和标量金标准；记录 full_backend=not_run、oracle_mc=not_run，不继续投入完整后端或大规模模拟。
- 报告需要新增独立实体，还是需要按事前隔离协议重新训练编码器。
- 不从现有 1,000 个测试查询中事后抽走一部分，再声称沿用原测试协议。

### 4.2 认证乘积的独立单位

P0 采用最容易审计的设计：先按实体块划分，再在认证池内预先随机配对；一条认证记录所用的全部底层实体，在其他认证记录中不得复用。

配对抽样应在查看匹配标签及置信度之前确定。不得为了正负各半进行标签条件筛选后，仍使用原总体的普通 Bernoulli 似然。训练拟合器可以采用预先定义的采样和正则，但认证乘积必须与实际认证抽样机制一致。

独立性是总体抽样假设；ID 不重叠只是必要的实现检查，不是证明整个实际数据集独立同分布。

必须报告：

- n_fit_pairs、n_fit_entities、n_cert_pairs、n_cert_entities。
- 认证实体复用次数、跨角色重叠、别名/重复内容检测结果。
- 配对种子、完整索引或其受控存储路径与哈希。
- 自然正例比例；不以该比例为理由重采样认证集。

所有样本保留在原数据及检索流程中。认证角色划分和预定义审计抽样不是按置信度筛样。未进入认证似然的样本不得被悄悄删除或改标签。

## 5. 固定采用的统计集合

### 5.1 独立训练分子

只使用开发/拟合数据获得有限参数 beta_fit。默认使用同一个冻结 logistic 族的 ridge/MAP 拟合；其正则系数在认证前锁定，或者只在拟合数据内部选择。

分子是 p_beta_fit。这样 beta_fit 本身属于候选族，并提供严格可行见证。不能把认证集 MLE 替换成 beta_fit，继续使用同一保证。

### 5.2 认证使用未加权总 NLL

对认证设计矩阵 V 和 y_i∈{0,1}：

$$
f(\beta)=\sum_{i=1}^{n_{\mathrm{cert}}}
\left[\operatorname{softplus}(v_i^\top\beta)-y_i v_i^\top\beta\right].
$$

定义：

$$
c_\delta=f(\widehat\beta_{\mathrm{fit}})+\log(1/\delta),
\qquad
C_\delta=\{\beta:f(\beta)\le c_\delta\}.
$$

默认 delta=0.05。delta=0.10 仅作为固定的语义/集合包含回归检查；实际实验不能根据结果挑 delta。

硬性要求：

- f 是真实 Bernoulli 对数似然的总和。
- 不复用训练里的正负平衡权重、label smoothing 或额外温度。
- 若内部使用平均 NLL，阈值增量必须同时除以 n_cert，且测试证明与总和实现等价。
- 认证分母不加 ridge、先验惩罚、任意参数盒或参数球。
- 使用稳定 logit/softplus 计算，不用概率截断制造有限 log 值。
- beta_fit 的可行裕量应为 log(1/delta)；同族有限参数设置下报空集，视为实现、数值或数据一致性错误。

等价似然比为 exp(f(beta)-f(beta_fit))。在正确条件模型和所需数据独立性下，其在 beta_true 处的期望至多为 1，阈值 1/delta 由 Markov 不等式得到。这是 [1] 的 split likelihood 构造在本任务上的应用，不是新提出的统计定理。

### 5.3 投影及置信语义

对查询特征 a=v(x)：

$$
z_-(a)=\inf_{\beta\in C_\delta}a^\top\beta,\quad
z_+(a)=\sup_{\beta\in C_\delta}a^\top\beta,
$$

$$
L_\delta(x)=\sigma(z_-(a)),\quad U_\delta(x)=\sigma(z_+(a)).
$$

同一个 beta_true∈C_delta 事件蕴含所有查询的两侧概率界成立。查询数量不是独立训练重复次数，也不需要为了这一参数集合投影再逐查询分摊 delta。

必须区分：

- delta=0.05：95% 参数置信集合及其联合投影保证。
- 后验 q05/q95：所选后验的分位点，通常对应其中心 90% 后验质量。
- 这两者不是同一个统计对象，不能命名混淆。
- 对负类正确事件，认证下界为 1-U_delta，而非 L_delta。
- 精确集合满足 C_0.05 包含 C_0.10；精确投影满足 L_0.05≤L_0.10、U_0.05≥U_0.10。

集合不携带内部概率测度，不定义 posterior_mean、AU 或 EU。新字段与旧后验字段分开保存。

## 6. 数值求解必须提供向外的界

### 6.1 禁止将 primal 可行点当成安全端点

最小化问题的可行解只给最小值的上界；直接对它做 sigmoid 可能把概率下界抬高。最大化问题同理，可能把概率上界压低。

每个端点必须保存 primal、dual、可行性、gap、终止原因，以及最终端点用的证据。仅有 solver 的 optimal 标志或很小的 KKT 残差，不算通过。

### 6.2 可使用的显式弱对偶

令 h(r)=-r log r-(1-r)log(1-r)。对 min a^T beta subject to f(beta)≤c_delta，任何满足

$$
\lambda\ge0,\quad r_i\in[0,1],\quad
a+\lambda V^\top(r-y)=0
$$

的点都给出全局下界

$$
d(a)=\lambda\left[\sum_i h(r_i)-c_\delta\right].
$$

分别对 a 和 -a 得到向下有效的 d_-、d_+ 后，最终外包为：

$$
L_{\mathrm{safe}}=\sigma(d_-),\qquad
U_{\mathrm{safe}}=\sigma(-d_+).
$$

这是从 logistic 凸共轭得到的实现证书；不要求求解器达到精确最优才可返回保守端点。

### 6.3 验证后端

P0 至少实现一种可核验的对偶可行性处理。推荐低维头使用：

1. 凸求解器给候选 dual。
2. 置 u=lambda(r-y)，检验 V^T u=-a 和 -lambda*y_i≤u_i≤lambda*(1-y_i)。
3. 将存盘浮点特征和 beta_fit 视作其精确二进制有理数值；对一组独立主元行做有理数等式修复，并精确检查盒约束。
4. 对 c_delta 给出向上包围 c_upper；对熵和对偶目标使用高精度区间运算，得到向下的 objective bound。
5. sigmoid 下端向下、上端向上处理舍入误差。

允许其他等价、可核验的方法。若用等式残差修正，必须从原 C_delta 证明该残差方向的有效支撑界；无界 beta 下，不能把 1e-8 等式残差直接解释为 1e-8 端点误差。

求解和高精度验证使用 float64/更高精度，冻结特征的原始精度另行记录。记录证书对应的 c_upper；它定义的是原集合的保守外包，不能声称它与精确集合完全一致。

零方向 a=0 直接返回 logit [0,0] 和概率 [0.5,0.5]；不得在 lambda=0 时使用 u/lambda。熵端点按连续延拓定义 h(0)=h(1)=0，避免 0*log(0) 产生 NaN。

### 6.4 无界与失败

- beta_fit 严格可行却报 empty：实现失败，不自动改阈值。
- 参数集合不紧并不意味着无效。
- 已证明 logit 下投影为负无穷，输出 L=0；上投影为正无穷，输出 U=1。
- 可用精确验证的分离方向 d 作证：所有 (2y_i-1)v_i^T d≥0，且查询方向内积具有相应符号。
- 求解超时、dual 无法验证、精度不足时，无法认证的一侧回退 0 或 1；保留所有行和失败原因。
- 区分 unbounded_proven、numeric_uncertified、timeout、budget_not_evaluated。
- 一般浮点近似结果可以另存 approximate 字段，但不得进入 certified 字段。

输出全部 [0,1] 只能说明保守退路运行正常，不算实用性通过。

## 7. 模块和接口交付

以下为 P0b 的拟新增布局；P0a 可只交付数据契约、审计入口与标量金标准。执行端可在不改变职责的前提下对接现有代码。当前仓库没有这些模块；不能把计划中的接口写成已存在。

- certification/README.md：一键命令、定义、假设、失败状态。
- certification/data_contract.py：实体角色、配对、来源与形状检查。
- certification/fit_reference.py：只在 D_fit 拟合 beta_fit。
- certification/confidence_set.py：总 NLL、阈值、可行见证和 likelihood ratio。
- certification/projection.py：统一 linear direction 投影及保守概率端点。
- certification/dual_certificate.py：独立于优化器的证书核验。
- certification/audit.py：区间紧致度、固定排序比较、资源和状态汇总。
- certification/run.py：完整 CLI。
- certification/configs/：锁定的 oracle 与 NUS 配置。
- tests/test_certification_*.py：第 8 节必要回归。
- docs/updates/：实现说明、实际运行记录、未完成条件。

输入采用可独立加载的冻结特征缓存与 manifest，至少包含：

- features: [n,d]，有限实数。
- labels: [n]，严格 0/1。
- pair_id、image_entity_id、text_entity_id、source_block_id。
- checkpoint_hash、feature_definition_hash、label_rule_hash、split_hash。
- 所有标准化参数及其拟合集来源。
- 原始数据/权重路径由配置提供，不写死用户 Windows 路径。

核心接口职责：

~~~text
validate_provenance(manifest) -> AuditStatus
fit_reference(fit_features, fit_labels, locked_config) -> ReferenceState
build_confidence_set(cert_features, cert_labels, reference, delta) -> SetState
certify_direction(set_state, direction, solver_config) -> DirectionCertificate
project_probability(set_state, pair_features) -> CertifiedInterval
audit_fixed_pairs(intervals, direction_certificates, manifest) -> Report
~~~

DirectionCertificate 至少包含 direction_hash、set_hash、primal_bound、dual_bound、verified_dual_feasible、objective_gap、certificate_error_budget、runtime_seconds、status、fallback_reason。

CertifiedInterval 至少包含 pair_id、set_hash、delta、certified_lower、certified_upper、width、lower_status、upper_status。若保留旧读出，使用 posterior_q05、posterior_q95、posterior_mean、posterior_au、posterior_eu 字段；默认可为 null。

必须提供命令行的 validate、oracle、run、report 子命令。README 里的命令在实现后实际执行一次；本需求文件中的命令形态不是已可运行程序的声明。

## 8. 必要测试和最小覆盖验证

### 8.1 硬币模型金标准

训练给 q∈(0,1)，认证 n 条中有 k 个正例。集合为：

$$
k\log p+(n-k)\log(1-p)
\ge k\log q+(n-k)\log(1-q)+\log\delta.
$$

金标准：

- n=0：返回 [0,1]。
- k=0：[0, 1-(1-q)delta^(1/n)]。
- k=n：[q*delta^(1/n), 1]。
- 0<k<n：分别在 (0,k/n) 与 (k/n,1) 用独立高精度括区求根。

该金标准不是 Clopper-Pearson 区间。Clopper-Pearson 可用于后续重复实验的频率置信区间，不能代替这里的端点。

用固定 q、真实 p 和 n 的小网格枚举全部 k，按二项概率精确加权，核验覆盖不低于 1-delta。该测试检查阈值方向和边界，不训练学生。

### 8.2 投影方向与数值回归

至少包括：

- 一自由参数 logistic，查询方向正、负、零。
- 全正/全负、完全分离、秩亏、查询含认证矩阵未识别方向。
- 总和 NLL 与正确缩放的 mean NLL 等价。
- 错误正负加权、标签平滑、未缩放 mean 阈值必须被配置检查拒绝。
- beta_fit 严格可行见证。
- 人为制造 primal 可行但未最优的解，确认它不能直接进入 certified_lower。
- 人为制造 dual 等式小残差但不精确可行的解，确认进入修复或保守回退。
- 导出的证书由独立 verifier 重载后仍可核验。
- 失败/超时保留行，输出不越过 [0,1]。
- 95% 与 90% 的精确集合包含关系在金标准上成立。

对一般的保守外包，不能因为独立求解松紧不同就强制端点满足精确嵌套。若希望报告嵌套外包，只能做有证明的向外放宽，不能为“平滑”而向内收紧。

金标准端点验收使用高精度 bracket，要求认证区间向外包含它。端点 gap 默认 1e-6；这是数值精度目标，不是统计覆盖误差预算。无法达到时报告较宽外包及 gap，不伪造精确值。

### 8.3 正确设定的覆盖模拟

本节仅在 P0b 执行。默认一个明确的 8 维 logistic 设定，包含截距，n_fit=n_cert=512，独立重复 1,000 次。参数和特征生成机制在配置中锁定；每次独立生成拟合、认证数据。

主要检查：

$$
\mathbf1\left[
f(\beta_{\mathrm{true}})-f(\beta_{\mathrm{fit}})
\le\log(1/\delta)
\right].
$$

这不需要对每次重复的大量查询求投影。仅选少量固定重复和查询进行端点一致性检查。

如运行样本量检查，预先指定 n_fit=n_cert∈{128,512,2048}，只增加集合 membership 统计；总支持量明确为 2*n，不能与过去“128 条支持”的实验混同。固定 beta_fit 只增加认证数据，区间可能因拟合误差留下宽度下限，不能要求必然收缩至零。

报告库级覆盖率、精确二项区间及理论目标。不可要求“95% 置信区间下端必须超过 95%”作为名义 95% 方法的通过条件。

预先定义一侧欠覆盖检验：若覆盖率的单侧置信上限低于 1-delta，则定位实现/模拟错误；多配置时对检验显著性预先分摊。未拒绝不足覆盖，不等于实验已经证明覆盖定理。

若参数确实在集合内而某个已认证投影漏掉相应真实概率，这是实现错误，直接失败。

## 9. NUS 实际审计及预算

### 9.1 启动顺序

1. 导入旧实验 provenance 和新的数据 manifest。
2. 确定并锁定 estimand、特征族、delta、分子拟合配置。
3. 完成 P0a：检查认证实体来源、独立配对与必要标量金标准；不满足时输出 blocked 和具体缺项，到此交付，不启动完整后端。
4. 拟合一次 beta_fit，构造一次静态集合。
5. 做预定小批投影试跑；达到资源和数值条件后扩展。
6. 完整保存成功、无界、回退、未计算行，生成四层结论。

### 9.2 固定、标签盲的审计样本

从既定 1,000 测试查询中，按固定哈希/种子选择最多 32 个审计查询，不能依据标签、置信度、求解容易程度选择。

对每个查询，按原版余弦排序选 ranks 1、2、20、21 的候选，默认最多 128 个图文对。若图库不足相应 rank，记录原因，不能换成效果更好的 rank。

每个审计查询另外检查 (1,2) 和 (20,21) 两个相对次序。直接复用投影器，使用方向 a=v_i-v_j：

$$
\inf_{\beta\in C_\delta}(v_i-v_j)^\top\beta.
$$

向下有效界大于 0 时，才称“该模型集合中的所有候选参数都给出 i 高于 j”。这仍以正确设定假设为前提，也不保证一次随机标签结果必然遵从该次序。

若要认证反向次序 j 高于 i，必须另对 -a 取得严格大于 0 的已验证下界。a 的下界小于 0 不能证明反向次序；只说明该方向尚未被认证。两侧均不能认证时输出 order_unresolved。每个比较按两次方向投影计入总预算；directions.csv 分别保存 forward_certified、reverse_certified 和 order_unresolved。

这个小规模方向审计用于检验集合能否帮助解释排序；不新增 pairwise 学生或反馈损失。没有认证成功的比较照样保留，不能只报可认证子集的检索成绩。

### 9.3 预先锁定的工程预算

以下为本阶段人为工程门槛，不是统计定理：

- 试跑：8 个审计查询，即 32 个概率区间及 16 个排序比较；共 64+32=96 个方向投影。
- 单次方向投影与验证预算：10 秒。
- 正式真实投影总 wall time 上限：30 分钟。
- 最多扩展到上节 32 查询/128 区间/64 比较；共 256+128=384 个方向投影，仍受 30 分钟总预算约束。未计算项目记 budget_not_evaluated。
- 已尝试方向中 timeout 或 numeric_uncertified 比例超过 5%，判数值/效率实用性未通过；允许完成日志汇总，不继续扩大规模。
- 若已评估区间中超过 90% 的宽度大于 0.9，或两类排序比较的任一方向认证率均低于 5%，标记 weak_practical_value，结束当前配置的大规模扩展。
- 这些门槛和审计种子必须在看结果之前保存；修改配置需另 run_id 并说明，不覆盖原结果。

为避免全 [0,1] 通过，必须分别报告 valid-but-vacuous 与 useful。排序认证率仅用于决定是否值得推进，不能写成 mAP 增益。

不训练、不改编码器、不改原余弦排序；因此本阶段不主张反馈训练收益。若执行原检索评测，应使用完整既定查询/图库并记录冻结权重与特征哈希。不能以审计子集替代原 mAP 分母。

## 10. 必须交付的产物

每次运行记录 stage=P0a 或 P0b。P0a 阻塞交付保存配置、来源/事件审计、标量金标准和 REPORT.md，其余输出在 manifest 中明确标 not_run，不伪造空的成功产物。P0b 完整运行至少保存：

- config.json：实际解析配置、版本、预算、种子、delta。
- provenance.json、entity_overlap_report.json、pair_sampling_manifest.json。
- estimand.md、event_definition.json、assumptions.json。
- reference_state.npz：beta_fit、特征预处理及可重载信息。
- confidence_set.json：集合定义、阈值及向上包围、可行见证、输入哈希。
- intervals.csv：逐对认证端点、宽度、状态、回退原因、耗时。
- directions.csv：固定排序比较的方向界、状态、是否能判定次序。
- certificates/：独立 verifier 所需的证书和对应输入哈希。
- oracle_coverage.csv：重复单位为独立拟合/认证库的 coverage。
- summary.json、REPORT.md：四层结论、资源、限制与下一步决定。

真实数据没有已知 p*(x) 时：

- true_probability_coverage 必须为 null。
- coverage_status 必须为 not_observable_from_single_labels。
- 可以报告预先指定点预测的 Brier/NLL/ECE，字段注明预测源与评估分布。
- 不得把 label∈[L,U] 的比例当作真实概率覆盖率。
- 不得把一个库下的数千查询当独立库来计算二项置信区间。
- 不得用 interval midpoint 冒充后验均值。
- 不得给置信集合强行补出 AU/EU。

REPORT.md 必须有一个直接决定：

- 数据条件未建立：真实认证 blocked，列出具体缺失。
- 有效但过宽/过慢：本配置不进入反馈训练。
- 假设下有效且有实际区分能力：可以进入下一阶段冻结读出评估。
- 只有静态认证结果：retrieval_gain=not_tested，training_feedback_gain=not_tested。

## 11. 第一轮交付验收清单

### 必须通过

- 原始事件、条件对象和模型假设写清，未悄悄降低目标。
- beta_fit、冻结表征和认证数据的来源依赖可追溯。
- 认证实体与认证乘积单位符合协议，无法核实时不冒充满足。
- 总和 NLL、阈值方向、严格可行见证正确。
- 金标准和错误注入测试通过。
- 每个 certified 端点都有可独立核验的向外数值依据。
- 超时、无界、空泛及来源阻塞完整保留并进入统计。
- 新集合与旧后验摘要字段分开，95% 集合与 5/95 分位点语义分开。
- 所有实际执行命令、运行时间、版本、种子和原始汇总可复现。
- 结论分别说明四层状态，失败结果允许作为正式交付。

### 不作为本阶段验收条件

- 下界必须对每个样本上升。
- 区间必须逐次数据规模单调收缩。
- EU 必须下降。
- mAP 必须提升。
- 每个数值问题都必须返回非平凡端点。
- 给每个样本 95% 区间后，无需重新处理模型选择就可以继续用同一认证集训练。

## 12. GitHub 交付规则

实现、必要测试、配置、README、CHANGELOG 和实际运行说明统一放在本仓库 fume_bounds/ 下。保留现有 v3/v4/v5 及已完成实验的历史，不通过覆盖历史定义来制造可比结果。

提交说明明确区分：

- 已实现、已实际执行、只写了规格。
- 合成正确性、真实数据运行、统计假设。
- 用户提供的旧实验结果和本次执行端重现结果。

原始图文数据、大权重、凭据和私人服务器路径不提交到仓库；记录可重现的受控路径约定、哈希与摘要。成功写入远端后核对 commit 与文件内容，再告知用户可以 git pull。不得将本规格提交说成认证功能已完成。

## 13. 理论来源

[1] Larry Wasserman, Aaditya Ramdas, Sivaraman Balakrishnan. Universal Inference. PNAS 117(29):16880–16890, 2020. DOI: 10.1073/pnas.1922664117。本文使用其 split likelihood confidence set 和函数投影思路；“universal”不取消真实机制属于所声明模型族的条件。

- 作者全文：https://arxiv.org/pdf/1912.11436
- 期刊全文入口：https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/

本文的 logistic 弱对偶形式、数值验证方案、NUS 数据协议和工程门槛是针对本项目写出的实现要求。它们不表示已经证明真实 NUS 的模型正确设定，也不表示已经运行了后续认证实验。
