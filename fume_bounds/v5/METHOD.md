# v5 设计与公式：多标签监督、单位先验证据、原对比学习

## 0. 范围及来源

本版替换 fuzzy 分支，不是往完整FUME上额外叠一项损失。
上游固定为 `siyuancncd/FUME@5dd7a306aa505d8f60265b0353ff98de19ad9815`。
- [model.py](https://github.com/siyuancncd/FUME/blob/5dd7a306aa505d8f60265b0353ff98de19ad9815/2025-CVPR-FUME/model.py)
- [processor.py](https://github.com/siyuancncd/FUME/blob/5dd7a306aa505d8f60265b0353ff98de19ad9815/2025-CVPR-FUME/processor.py)
- [options.py](https://github.com/siyuancncd/FUME/blob/5dd7a306aa505d8f60265b0353ff98de19ad9815/2025-CVPR-FUME/options.py)
- [main.py](https://github.com/siyuancncd/FUME/blob/5dd7a306aa505d8f60265b0353ff98de19ad9815/2025-CVPR-FUME/main.py)
- [VLS 正文与附录](https://papers.nips.cc/paper_files/paper/2025/file/97b73904e88cc1dc0a3485595eda3753-Paper-Conference.pdf)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

不是作者官方VLS代码的逐文件移植，也不将共享预算参数化称为原论文设计。
成熟公式可以复用；复用/任务改写/新增约束在下面分别标明。

## 1. 数据：完整保留所有标签

每条记录 `(x_i^I,x_i^T,y_i)`，`y_i∈{0,1}^K`，不要求标签和为1。
例：图像和文本都描述“蓝天下一只猫与一只狗，旁边没有汽车”，
标签按[猫,狗,天空,汽车]为 `[1,1,1,0]`，两路都直接监督这个向量。
多标签矩阵不取argmax、不除以正标签数。类别编号转换需要显式编号基准。
不改变文件自带train/val/test划分，不借用test作val。

## 2. 编码器与类别读出

复用四个 `Linear+ReLU` 隐藏层，宽度4096。最后 `Linear(4096,K,bias=False)` 不加激活。
两路参数独立。隐藏输出记r，最终线性输出记h：

`h^v = A_out^v r^v`，`z^v = h^v/max(||h^v||_2,1e-8)`。

原FUME的固定方形正交W保留并注册buffer。类别分数改为 `a^v = h^v W`。
读取h而不是z的理由：固定正交W和单位z会约束类别分数的总幅度，不适合多标签同时接近1。
检索仍使用z；没有额外的类别MLP，没有400维图文拼接，没有PairEvidenceHead。

## 3. 保留“证据+1”，统一最终概率

方向 `r_k = sigmoid(a_k)`；它只是证据分配方向，**不是最终预测均值**。
证据预算 `s = softplus(w_s^T r_hidden+b_s)`，每样本每模态一个标量。

`e_k^+ = s r_k`, `e_k^- = s(1-r_k)`

`alpha_k = 1+e_k^+`, `beta_k = 1+e_k^-`

`P_k ~ Beta(alpha_k,beta_k)`

`p_k = alpha_k/(alpha_k+beta_k) = (1+s r_k)/(s+2)`

等价于 `p_k = [s/(s+2)] r_k + [2/(s+2)] (1/2)`。
零证据回到Beta(1,1)，而不是未定义分布；没有删除先验，没有用epsilon替代1。
各类别互不争夺总概率，可同时接近1。alpha与beta仅在各自标签内归一化。

共享s是明确的新参数约束：减小输出自由度，各类别不能独立拥有任意证据总量。
同一样本vacuity各类相同；TU/AU/EU仍随各类均值变化。不能宣称无表达能力损失。
每路新增Linear(4096,1)，合计8194个可训练参数。初始化s=2，只是初始化，不是固定值。

## 4. 原始类别监督（直接使用multi-hot）

`L_cls = mean_{v,i,k}[-y_ik log p_ik^v -(1-y_ik)log(1-p_ik^v)]`。

数值实现使用 `predictive_logits=log(alpha)-log(beta)`，然后BCEWithLogits。
不能直接监督方向logits a后又宣称监督了同一个Beta均值；也不能把概率再送进重复Sigmoid。
由于+1先验使均值依赖s，原始BCE也会训练s。本版明确保留这条梯度，不声称浓度只受VLS监督。

## 5. 逐标签证据期望损失（复用VLS损失的二类形式）

每个标签：

`ell(t;alpha,beta)=t[psi(alpha+beta)-psi(alpha)] +(1-t)[psi(alpha+beta)-psi(beta)]`。

按B、K与两种模态取平均。S=alpha_k+beta_k，**不是sum_k alpha_k**。
训练中Digamma小张量使用float64计算，梯度正常返回float32编码器。

### 5.1 Mixup路径

`lambda~Beta(10,10)`，同一随机排列j和lambda用于两模态：

`x_mix^v=lambda*x_i^v+(1-lambda)*x_j^v`

`y_mix=lambda*y_i+(1-lambda)*y_j`

例如 `[1,1,1,0]` 与 `[0,1,0,1]` 以0.6混合得到 `[0.6,1,0.6,0.4]`，不归一化。
输入空间各自混合，再进入同一套网络；不是在单位球面上插值。
两模态的逐标签证据损失平均为L_vic。

### 5.2 噪声路径

`rho~Beta(1,1)`，`epsilon^v~N(0,sigma_v^2 I)`。

`x_noise^v=rho*x_i^v+(1-rho)*epsilon^v`

`y_noise=rho*y_i+(1-rho)*0.5*ones(K)`。

每个标签是二元事件，故无偏目标为0.5，不是互斥K分类的1/K。
两路rho相同，噪声独立。sigma_v仅由训练特征RMS计算；不更改原始特征、不接触测试统计。
这是输入特征尺度适配，不声称高斯噪声等于真实歧义。
噪声和Mixup并行生成，不串联叠加。

## 6. 原FUME对比目标

拼接Z=[ZI;ZT]。配对索引pi(a)=(a+B) mod 2B，自身对角线屏蔽：

`L_CL=-(1/B) sum_{a=1}^{2B} log exp(z_a dot z_pi(a)/tau) / sum_{b!=a} exp(z_a dot z_b/tau)`。

默认tau=1。原代码loss1+loss2对称相等，因此实现为2倍普通2B锚点CE。
测试逐值与逐梯度对照原processor算法。该Softmax是候选间归一化，和多标签类别输出无关。
保留同模态负例及非对应同语义假负例的原协议，不悄悄换监督对比。
仅原始输入使用这项损失。

## 7. 总训练目标与计算

`L=L_cls + lambda_v L_vic + lambda_n L_noise + lambda_CL L_CL`。

初始lambda_v=lambda_n=1，lambda_CL=.05，Adam lr5e-5、betas(.5,.999)、wd0。
类别损失缩放已变，原权重只是起点，不能声称作者替v5调过。
三批输入依次forward/backward累积梯度，中途不step，最后一次Adam.step。
共享W固定，编码器与预算层可训练；不使用detach阻断可信损失到编码器的梯度。
实际处理3B输入/模态，没有按K重复编码，也没有正负pair扩展成7B。
默认从头训练。继承完整FUME checkpoint的实验必须标注fuzzy预训练来源。

## 8. 不确定性与测试

`TU_k=-p_k log p_k-(1-p_k)log(1-p_k)`

`AU_k=psi(S_k+1)-p_k psi(alpha_k+1)-(1-p_k)psi(beta_k+1)`

`EU_k=TU_k-AU_k`, `vacuity_k=2/S_k`。

微分熵使用Beta解析式，可以为负。类别平均值不是完整标签组合的联合熵。
原始检索用单位z余弦，不按置信度删除查询或候选。相关性以标签交集定义，保留原始对应项。
Brier/NLL/ECE/风险覆盖针对各模态标签概率，不再拿旧的图文配对指标冒充。

## 9. 理论与实现边界

- 合法的Beta边缘模型和熵分解，不代表真实AU/EU已经识别正确。
- 期望CE可能继续推动浓度增大；+1不是防止所有虚假自信的定理。
- 强混合的标签仍是人为监督，不保证等于虚拟输入的真实概率。
- 未直接继承VLS式16--18的跨模态测试风险结论；Beta(10,10)独立扰动方差系数为11/21，不是mAP提升比例。
- 真实数据、多个seed、同预算消融及有真实歧义依据的评测仍需执行。

## 10. 与旧版的硬区别

旧版：fuzzy损失 + 原对比 + 二元图文pair VLS。
本版：multi-hot均值BCE + 原对比 + 类别Mixup/噪声Beta监督。
不调用原calc_loss，不生成必要性/c，不使用固定W后的ReLU作为概率，不跨类归一化。
