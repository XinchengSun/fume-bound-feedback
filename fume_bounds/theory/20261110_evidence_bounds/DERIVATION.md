# FUME：证据、有效概率下界、信息半径与检索改进

日期：2026-11-10（北京时间）。核对的实现基线：cb4cf88dfdba4008a11c0c88b46f0c414d4999d6。

本轮交付是文献核验、条件性定理与可复跑的数学核验，没有新增 NUS 训练结果。六组 train_prior 权重对照暂停作为研究主线；它不能修复本文指出的统计与目标对应问题。历史代码和负结果保留。

## 0. 本文真正证明了什么

可以闭合的链条是：

$$
\text{明确且正确的观测模型}
\longrightarrow
\text{全时间有效的参数置信集合}
\longrightarrow
\text{有效且嵌套的概率区间}
\longrightarrow
\text{最坏对数损失遗憾单调不增}.
$$

再增加一个明确的检索概率模型，可以在保持实际 cosine 排序的条件下，对“同一真实模型中的新旧期望 AP 差值”给出下界；只有差值下界通过验收，才有相应的条件期望改进保证。

这不证明旧损失 softplus(ks-ta) 会降低真实 EU，也不证明任何数据、任何初始化下都存在性能更好的候选。这里的统计工具与信息半径等价关系已有文献基础，不作为新的理论创新宣称。

### 必须分清的三个量

1. 后验 EU：某一个指定后验下的互信息，沿用此前定义。
2. 信息半径：有效候选集合允许的最大预测分歧，等于最坏对数损失遗憾；它是不同的对象。
3. AP 改进下界：同一概率机制下，两个实际排序的条件期望 AP 差值的保守界。

不能把三者重命名为同一个“置信度”，也不能用其中一个数值下降代替另外两个结论。

## 1. 现有直接下界损失的边界

当前近似后验使查询 logit 满足 Z~N(a,s²)，正例下界为 L=σ(a-ks)，k=Φ⁻¹(.95)。已知标签 y 的符号 t=2y-1，对应正确事件下界和损失为：

$$
L_y=\sigma(ta-ks),\qquad
\ell_L=-\log L_y=\operatorname{softplus}(ks-ta).
$$

这是近似后验下的分位点风险，不是整个高斯支撑上的硬下界。完整梯度为：

$$
\nabla_\theta\ell_L
=(1-L_y)(-t\nabla_\theta a+k\nabla_\theta s).
$$

### 1.1 任意固定 k 都不能保证“损失下降则 EU 下降”

取正例，比较 (a₀,s₀)=(0,ε) 与 (a₁,s₁)=(k+1,1)。有 L₀<1/2<L₁=σ(1)，故 ℓ₁<ℓ₀；但 ε→0 时 EU₀→0，而第二个非退化概率分布的 EU₁>0。取足够小的 ε 即构成反例。

这个反例针对 logistic-normal 模型族中的无条件蕴含，不声称某个有限网络能达到任意参数，也不否认某些设置可能出现经验改进。

### 1.2 两条确实成立的放缩

令 P=σ(Z)、m=EP，采用自然对数：

$$
\mathrm{EU}=h(m)-Eh(P)
=E D_{\mathrm{KL}}(\mathrm{Ber}(P)\Vert\mathrm{Ber}(m)).
$$

对任意 r，E D(P||r)=EU+D(m||r)。取 r=σ(a)，用 A(z)=log(1+exp z)、A''≤1/4：

$$
D(\sigma(Z)\Vert\sigma(a))
=A(a)-A(Z)-A'(Z)(a-Z)
\le (Z-a)^2/8.
$$

因此：

$$
\boxed{\mathrm{EU}\le \min\{\log2,s^2/8\}.}
$$

若正确事件概率 R 的 δ 分位点为 L_y，则 ER≥(1-δ)L_y，所以：

$$
\boxed{-\log ER\le\ell_L-\log(1-\delta).}
$$

这给出后验预测 NLL 的上界。上界下降不意味着实际量必然下降；更不把模型内结论转成真实概率覆盖。

### 1.3 为什么采用下界必须先指定决策目标

若目标只是该后验下的期望对数损失，令 m=EP，则预测 r 的风险为：

$$
R_\pi(r)=-m\log r-(1-m)\log(1-r),\qquad
R_\pi(r)-R_\pi(m)=D_{\rm KL}(\mathrm{Ber}(m)\Vert\mathrm{Ber}(r))\ge0.
$$

因此后验预测均值才是这个目标的最优点预测；把 r 换成下界没有一般优势。保守下界的理由应来自另外明确提出的决策要求，例如对仍可能为真的机制控制最坏风险，或认证某次检索更新的最小改进量。后文分别给出这两种对象。不存在仅凭“有不确定性”就必须使用下界训练的结论。

## 2. 保留原 EU 定义时：新增数据后的条件期望收缩

设 W、所有数据和未来查询标签来自同一个一致的 Bayesian 联合模型。固定查询 x、事件与映射 p_W(x)，令 F_n 为已经获得的信息，O 为新观测。要求：

$$
\Pr(Y_x=1\mid W,F_n)=p_W(x),\qquad
F_{n+1}=\sigma(F_n,O).
$$

Y_x 是没有被历史揭示的未来标签，并假设 Y_x 与 O 在给定 W,F_n 后条件独立。下文的条件互信息是在给定当前信息实现后的条件分布中计算。

定义：

$$
e_n(x)=h(E[p_W(x)\mid F_n])
-E[h(p_W(x))\mid F_n].
$$

AU 项由迭代期望满足：

$$
E[\mathrm{AU}_{n+1}(x)\mid F_n]=\mathrm{AU}_n(x).
$$

预测熵的期望减少为 I(Y_x;O|F_n)，因而：

$$
\boxed{
e_n(x)-E[e_{n+1}(x)\mid F_n]
=I(Y_x;O\mid F_n)\ge0.
}
$$

这是一条真正保留原 AU/EU 定义的结果，但期望来自当前 posterior predictive；它不是每个固定未知真参数下的频率保证，也不是每条已实现轨迹都下降。罕见观测可能纠正错误先验，并使一次更新后的 EU 上升。

同一支持库重复训练、改变编码器映射、反复重新做 Laplace 近似，都不自动满足上述条件。

### 2.1 近似后验的代价

若精确后验 π_n 与近似后验 q_n 的总变差距离不超过 ε_n≤1/2，定义：

$$
\eta_n=h(\varepsilon_n)+\varepsilon_n\log2,
$$

则均值与熵的连续性给出：

$$
|\mathrm{EU}(q_n)-\mathrm{EU}(\pi_n)|\le\eta_n.
$$

故近似读出的条件期望收缩最多可以写成：

$$
E[\mathrm{EU}(q_{n+1})\mid F_n]
\le \mathrm{EU}(q_n)-I(Y_x;O\mid F_n)
+\eta_n+E[\eta_{n+1}\mid F_n].
$$

当前实验没有 ε_n 的有效上界，不能省去这些误差项。

## 3. 不靠后验分位点，直接构造有效置信集合

### 3.1 统计假设

固定特征 ψ(x)，并假设存在同一个固定 w*，使依次获得的观测满足：

$$
Y_i\mid F_{i-1},X_i
\sim\mathrm{Ber}(\sigma(w^{*\top}\psi(X_i))).
$$

F_0 包含用于固定特征、先验／λ、协议选择的全部初始信息；上述条件模型须在条件于这些信息后仍成立。正确模型假设还须适用于全部要声明覆盖的查询域，不能只在已观察设计点成立后，就把域外投影解释成真实条件概率。

特征可以在另外的训练数据上学习，但在认证期间固定。更一般的结论只要求条件似然 p_w(Y_i|X_i,F_{i-1}) 正确；不要求机械的 iid，但必须正确处理共享实体、条件抽样和已经获知的信息。预训练数据独立是一种使条件模型成立的设计方式，仅冻结一次或更换数据文件名本身不消除历史依赖。

真实概率对象是该固定特征条件下的匹配事件概率，不自动等于“同一原始图文对经任意人类语义判断”的频率。

### 3.2 预序似然比构造及证明

在看到 Y_i 前，根据允许使用的过去信息与 X_i 产生归一化预测 q_i(·)。它可以由近似 Laplace 或其他模型给出，预测器本身不要求正确。定义：

$$
E_n(w)=\prod_{i=1}^n
\frac{q_i(Y_i)}{p_w(Y_i\mid X_i)}.
$$

在真参数处，条件期望为：

$$
E[E_i(w^*)\mid F_{i-1},X_i]
=E_{i-1}(w^*)\sum_y
p_{w^*}(y\mid X_i)\frac{q_i(y)}{p_{w^*}(y\mid X_i)}
=E_{i-1}(w^*).
$$

因此 E_n(w*) 是初值为 1 的非负鞅。由 Ville 不等式：

$$
\Pr_{w^*}\{\exists n:E_n(w^*)>1/\delta\}\le\delta.
$$

构造：

$$
\boxed{
C_n=\bigcap_{j\le n}\{w:E_j(w)\le1/\delta\}.
}
$$

于是：

$$
\boxed{
\Pr_{w^*}\{\forall n:w^*\in C_n\}\ge1-\delta.
}
$$

这是 Universal Inference 的置信序列与运行交集构造 [1]，不是我们的新定理。q_i 必须在对应 Y_i 揭示前确定；不能用最终模型重写历史 q_i。

### 3.3 一个共同参数集合给全部查询覆盖

定义：

$$
L_n(x)=\inf_{w\in C_n}p_w(x),\qquad
U_n(x)=\sup_{w\in C_n}p_w(x).
$$

只要真实参数始终在集合里，所有 x 的投影都包含真值。因此：

$$
\boxed{
\Pr_{w^*}\{\forall n,\forall x:
L_n(x)\le p^*(x)\le U_n(x)\}\ge1-\delta.
}
$$

由 C_{n+1}⊆C_n，在集合非空时：

$$
\boxed{L_{n+1}(x)\ge L_n(x),\quad U_{n+1}(x)\le U_n(x).}
$$

这次单调性来自保留有效证据约束，不来自直接把网络输出抬高。它是弱单调，可能不动；运行交集也可能为空。空集不能静默重置，须报告该认证链失败。

## 4. 可计算的 logistic 版本：用全局曲率替代 Laplace 覆盖假设

本节对应 Clerico 等（2025 预印本）Theorem 4.1 的 logistic 实例 [2]。

定义未正则负对数似然和设计矩阵：

$$
F_n(w)=\sum_{i=1}^n[\operatorname{softplus}(w^\top\psi_i)-Y_iw^\top\psi_i],
\quad G_n=\sum_i\psi_i\psi_i^\top.
$$

事前固定 λ>0，令：

$$
J_n(w)=F_n(w)+\frac\lambda2\|w\|^2,
\quad \widehat w_n=\arg\min J_n(w).
$$

可取：

$$
\boxed{
D_n=\left\{w:F_n(w)\le
J_n(\widehat w_n)+
\frac12\log\det\left(I+\frac{G_n}{4\lambda}\right)+
\log\frac1\delta\right\},
\qquad C_n=\bigcap_{j\le n}D_j.
}
$$

### 4.1 为什么此集合有效

令固定混合分布 π=N(0,λ⁻¹I)，精确混合似然为：

$$
Q_n=\int e^{-F_n(w)}\pi(dw).
$$

Q_n/e^{-F_n(w*)} 是有效的混合似然比鞅。这里 π 是构造工具，真实 w* 无须从 π 抽样。

因为：

$$
\nabla^2J_n(w)\preceq A_n:=\lambda I+G_n/4,
$$

在 MAP 点有二次上界：

$$
J_n(w)\le J_n(\widehat w_n)
+\tfrac12(w-\widehat w_n)^\top A_n(w-\widehat w_n).
$$

取负指数并积分：

$$
Q_n\ge
e^{-J_n(\widehat w_n)}
\det(I+G_n/(4\lambda))^{-1/2}.
$$

用该下界形成的似然比被有效鞅逐路径支配，故 Ville 不等式给出第 4 节的集合。

对任意数值点 v，保守版本仍有效。二次上界包含梯度 r=∇J_n(v)，积分后多出 exp(rᵀA_n⁻¹r/2)≥1；舍去此正因子仍是下界。因此近似 MAP 的误差会扩大该保守集合，而不需要把小 KKT 残差当成统计覆盖证书。

正则项只进入阈值，参数 w 的分母仍是未正则数据似然。λ 若看完同一认证数据后任意挑选，需要额外处理选择效应。

### 4.2 投影计算

每个 F_j 的子水平集凸，C_n 为凸集合。二元 logistic 端点由以下线性目标凸约束问题得到：

$$
\underline z_n(x)=\inf_{w\in C_n}w^\top\psi(x),
\quad
\overline z_n(x)=\sup_{w\in C_n}w^\top\psi(x),
$$

$$
L_n(x)=\sigma(\underline z_n(x)),\quad
U_n(x)=\sigma(\overline z_n(x)).
$$

投影可以无界，此时合法结果可能为 0 或 1。不能任意裁出较窄区间。最小化问题的可行点仅给最优值的上界；构造可信概率下界需要全局下界、对偶证书或向外误差控制。

### 4.3 收缩到零还需要什么

再假设已知 ||w*||≤S、所有相关特征范数≤R，并将 C_n 与已知参数球 B(S) 相交。令：

$$
c=\sigma(SR)(1-\sigma(SR)),
\quad
\beta_n=\frac{\lambda S^2}{2}
+\frac d2\log\left(1+\frac{nR^2}{4\lambda d}\right)
+\log\frac1\delta.
$$

若 G_n 正定，强凸性与 sigmoid 的 1/4-Lipschitz 性给：

$$
U_n(x)-L_n(x)\le
\min\left\{1,\sqrt{\frac{\beta_n}{2c}}
\sqrt{\psi(x)^\top G_n^{-1}\psi(x)}\right\}.
$$

证明：以球内约束 MLE 为中心，C_n 中参数的 F_n 超额损失至多 β_n；球内 Hessian≥cG_n，故半径至多 sqrt(2β_n/c)。两点直径再乘 sigmoid 的 1/4-Lipschitz 常数即得。

若还满足 λ_min(G_n)≥κn，则宽度至多 R sqrt(β_n/(2cκn))，为 O(sqrt((d log n+log(1/δ))/n))。

这条简明收缩率使用全局 MAP；若只知道目标值优化误差至多 η_n，应把 η_n 加入 β_n。第 4.1 节任意数值点的覆盖保证本身不要求这个优化误差界。已知 S、信息增长及正确模型是实质假设。c 很小时该界可能完全无用；有效性不等于小样本有用性。更精细 GLM 置信序列见 [3]。

## 5. 从有效区间得到真正逐路径收紧的信息量

令 h(p)=-p log p-(1-p)log(1-p)。对固定 x，在 C_n 上允许任意混合分布 Q，定义：

$$
\mathcal E(C_n,x)
=\sup_{Q:\operatorname{supp}Q\subseteq C_n}
\left[h(E_Qp_w(x))-E_Qh(p_w(x))\right].
$$

它是集合允许的最大模型分歧。C_{n+1}⊆C_n 时，可选 Q 的范围缩小，故：

$$
\boxed{\mathcal E(C_{n+1},x)\le\mathcal E(C_n,x).}
$$

这里没有声称某个实际后验的 EU 相等于此值。

### 5.1 二元精确闭式

设 L<U、d=U-L，并定义：

$$
c_{LU}=\frac{h(U)-h(L)}{U-L}.
$$

由 h 的凹性，区间内：

$$
h(p)\ge h(L)+(p-L)c_{LU}.
$$

令 m=EP，得到：

$$
h(m)-Eh(P)
\le h(m)-h(L)-(m-L)c_{LU}.
$$

由端点混合可取等，所以：

$$
\mathcal E(L,U)
=\max_{\lambda\in[0,1]}
\left[h((1-\lambda)L+\lambda U)
-(1-\lambda)h(L)-\lambda h(U)\right].
$$

最大点满足 h'(m*)=c_{LU}，故：

$$
\boxed{
m^*=\sigma(-c_{LU}),\quad
\lambda^*=\frac{m^*-L}{U-L},
\quad
\mathcal E(L,U)=h(m^*)-h(L)-(m^*-L)c_{LU}.
}
$$

L=U 时定义 E=0；[0,1] 时 E=log2。端点仅可逼近时，以上结论按上确界理解。该结果是信息半径/通道容量的一维特例，不宣称原创 [4]。

### 5.2 操作意义：精确最坏对数损失遗憾

还可以直接证明：

$$
\boxed{
\mathcal E(L,U)
=\min_{r\in[0,1]}\max_{p\in[L,U]}
D_{\mathrm{KL}}(\mathrm{Ber}(p)\Vert\mathrm{Ber}(r)).
}
$$

KL 关于 p 凸，最大值取于 L 或 U。在 r∈[L,U] 内，D(L||r) 递增、D(U||r) 递减，最优点让两者相等。解得 r*=m*，代回即为第 5.1 节闭式。

对真值 p*，预测 r 的条件期望对数损失相对最优值的遗憾为：

$$
R(r;p^*)-h(p^*)=D_{\mathrm{KL}}(\mathrm{Ber}(p^*)\Vert\mathrm{Ber}(r)).
$$

因此，在有效覆盖事件上：

$$
\boxed{R(r_n^*;p^*(x))-h(p^*(x))\le\mathcal E(L_n(x),U_n(x)).}
$$

现在单调下降的是有精确操作意义的“最坏可允许遗憾”。并不声称实现的真实遗憾每一步下降。

### 5.3 与宽度、AU 的关系

令 d=U-L，可得 E(L,U)≤d log2。若区间在 [ε,1-ε] 内，由熵二阶导数和 Var(P)≤d²/4：

$$
\mathcal E(L,U)\le\frac{d^2}{8\varepsilon(1-\varepsilon)}.
$$

所以有效区间宽度趋零时，信息半径趋零。若 p*=0.5，最终 E→0，但真实事件熵 h(p*)=log2；知识不足可以消失，事件随机性仍然保留。

同样宽度、不同概率位置的信息半径可以不同。因此用“最坏预测遗憾”命名比宣称它是唯一正确的纯认知不确定性更准确。相关概念争议见 [5]。

## 6. 90% 后验区间不能冒充置信集合的全支撑投影

若原后验只有 1-α 的质量在 [L,U] 内，则一般不能写 EU≤E(L,U)。以下 α 表示实际尾质量；若仅知道尾质量上界，本节直接代入该上界的说法限定在 α≤1/2。将“落在区间内/外”作为额外二元变量，互信息链式分解给：

$$
\mathrm{EU}\le
\min\{\log2,\ h(\alpha)+(1-\alpha)\mathcal E(L,U)+\alpha\log2\}.
$$

固定尾质量 α 时，区间宽度变零也不消除尾部项。若只知道任意 α>1/2 的尾质量上界，应对实际尾质量 β∈[0,α] 最大化右侧。旧的 5%/95% 端点与本方案的置信集合极值有不同定义，不能换标签沿用旧覆盖数字。

## 7. 为什么检索需要比较“同一机制下的新旧差值”

即使某个新策略的最坏 AP 值更高，也未必在真实机制下更好。例：

旧模型在两种机制下的值为 (0.2,0.9)，新模型为 (0.3,0.8)。最坏值提高，但第二种机制下降。

所以真正的验收对象应是：

$$
\boxed{
\underline\Delta_n(\theta,\theta_0)
=\inf_{w\in C_n}
[V_w(\theta)-V_w(\theta_0)].
}
$$

若真实 w* 在 C_n 中，便有真实差值≥该下界。θ=θ₀ 时下界恰为 0，原模型在构造上可行。

分别计算 inf V_new-sup V_old 虽然安全，却会丢掉新旧模型共享同一个真实机制的信息。例如旧值 (0.4,0.8)、新值 (0.5,0.9)，同机制总是提高 0.1，分开极值却得到 -0.3。

针对基线差值做鲁棒改进有既有研究先例 [6]。这不是新的安全改进原理，也不保证非平凡可行更新一定存在。

## 8. 不改变 cosine 排序的精确 AP 接口

这一节是新增理论接口，不是原二元 Laplace 头已经实现的性质。本节先写单类别相等的相关性事件；若任务采用多标签交集，潜在状态必须改成完整标签向量，并按原交集规则计算 AP，不能用 argmax 改成单标签任务。此时联合状态数可能达到 2^K，只有逐标签边缘概率通常不足以确定期望 AP。

### 8.1 依赖结构与信息条件

设固定 gallery G={(g_j,c_j)} 的类别已知，新查询类别 C_q∈{1,...,K} 尚未观察。令 H_n 包含整个固定查询池的特征、图库及允许使用的图库标签、截至 n 的训练/认证数据、候选生成与选择历史。要求对每个查询和每个认证时刻：

$$
\Pr(C_q=c\mid H_n)
=p_{w^*}(c\mid\psi_0(q)).
$$

旧学生与候选学生都须是 H_n 可测的。查询池的真实类别不得被历史揭示，也不能从已知图库标签与配对 identity 中泄露。一个可满足该设定的例子是：图库、未来查询池与训练/认证实体独立构建，类别在给定固定完整特征后服从同一个正确条件模型。若学生或候选选择使用教师条件概率未覆盖的额外信息，应扩大条件模型，不能忽略。这个条件比“单个查询的边缘分类概率拟合得好”强。

学生仍使用实际 FUME cosine 分数。对每个可能查询类别 c，按实际排序计算：

$$
A_c(\theta,q)=
\mathrm{AP}(\operatorname{rank}(s_\theta(q,G)),
\{\mathbf1[c_j=c]\}_j).
$$

某类在 gallery 没有正例时预先规定 AP=0，不能临时改变口径。

由全期望公式：

$$
\boxed{
V_{w,n}(\theta,q)
=E_w[\mathrm{AP}\mid H_n]
=\sum_c p_w(c\mid q)A_c(\theta,q).
}
$$

给定查询类别，整个相关性向量确定，因此不需要假设同一查询下各候选相关性独立。

### 8.2 多类教师的有效集合与概率外界

使用固定特征 softmax 教师可直接采用第 3 节的一般条件似然比；多类负对数似然是凸的。

固定参考类 w_K=0。对每对类别 c,k，定义 d_ck=(w_c-w_k)ᵀψ₀(q)，求有效集合上的线性下/上界 l_ck,u_ck。由：

$$
p_c=\left(1+\sum_{k\ne c}e^{-d_{ck}}\right)^{-1},
$$

得到保守外界：

$$
\boxed{
L_c=\left(1+\sum_{k\ne c}e^{-l_{ck}}\right)^{-1},\quad
U_c=\left(1+\sum_{k\ne c}e^{-u_{ck}}\right)^{-1}.
}
$$

这些极值不必由同一个参数同时达到；因此外界可以较松，但仍包含真实投影。集合嵌套使各 L_c 非降、U_c 非升。若将这些外界代入第 5 节的信息半径公式，得到的是原参数集合信息半径的保守上界，不应声称等于原集合的精确半径。

若使用第 4 节的混合似然解析下界，不能照搬二元 1/4 曲率：多类 softmax 在全部 logit 上的 Hessian 谱上界一般为 1/2。参考类参数化下可安全使用 (1/2)I_(K-1)⊗G_n，因而行列式项变为：

$$
\tfrac12\log\det\left(
I_{(K-1)d}+
\frac{I_{K-1}\otimes G_n}{2\lambda}
\right).
$$

这个多类形式是同一混合积分证明的直接扩展，不是文献 [2] 标量响应 GLM Theorem 4.1 的原样引用。该保守构造可能很宽，应先报告可用性，不能以存在公式代替实际验收。

### 8.3 精确可计算的新旧 AP 差值证书

令 d_c=A_c(θ,q)-A_c(θ₀,q)，构造概率盒与单纯形交集：

$$
\mathcal P_n(q)=
\{p:\sum_cp_c=1,\ L_c\le p_c\le U_c\}.
$$

定义：

$$
\boxed{
\underline\Delta_n^{\rm box}(q)
=\min_{p\in\mathcal P_n(q)}p^\top d.
}
$$

这是 K 维线性规划。精确贪心解：从 p=L 开始，把剩余质量 1-ΣL 按 d_c 从小到大分配，直到对应 U_c 或质量用完，复杂度 O(K log K)。

应检查 0≤L≤U≤1、ΣL≤1≤ΣU；不满足时报告错误，不归一化修补。

真实类别概率向量在盒内，因此：

$$
\boxed{
E_{w^*}[\mathrm{AP}_{\rm new}-\mathrm{AP}_{\rm old}
\mid H_n]
\ge\underline\Delta_n^{\rm box}(q).
}
$$

这次是实际 cosine 排序的条件期望 AP 差值保证，不是另一个 surrogate 的上界变小。

### 8.4 一个可手算的非零改进证书

gallery 有类别 0、1 两个项目。旧排序为类 1 在前，新排序为类 0 在前，故 d=(+0.5,-0.5)。

若 p₀∈[0.4,0.9]，最坏差值为 -0.10，不能保证改进。若有效证据使区间收紧到 p₀∈[0.7,0.9]，最坏差值成为：

$$
0.7(0.5)+0.3(-0.5)=0.20.
$$

同一候选交换因此获得至少 0.20 的条件期望 AP 改进证书。两种排序都可由 cosine 分数产生，所有项目均保留。这是数学构造例，不是 NUS 增益。

### 8.5 汇总而非逐查询强制不退步

对事先确定权重的全部查询，令 Σ_qω_q=1、ω_q≥0，计算：

$$
\underline\Delta_n^Q
=\sum_q\omega_q\underline\Delta_n^{\rm box}(q).
$$

它是平均条件期望 AP 差值的保守下界。每个查询单独最坏，可能比共享同一 w 的总体最坏更保守，但仍有效。

只要求这个事前定义的平均证书通过，可以允许个别查询负贡献；不删除低置信查询，不改权重挑选成功子集。双向检索分别建有效教师时，需要联合失败预算，例如各 δ/2；此外两个方向均须满足第 8.1 节的条件概率假设。若另一模态的同一实体特征也进入历史，它可能提供额外类别信息，不能仅靠分配 δ 就忽略这种条件依赖。

在同一个全时间覆盖事件上，对每次接受更新要求该差值证书≥0，可推出相同固定查询/图库条件下的期望平均 AP 不下降。正收益候选不一定存在，训练也不保证找到它。

硬 AP 非光滑。训练可使用平滑损失产生候选，但证书必须重新计算候选的实际 cosine 排序和真实 AP 功能。不能用平滑损失变好替代这个步骤。

## 9. 同一查询下的标签相关性不可忽略

构造联合相关性分布：概率 0.6 时 (Y_A,Y_B,Y_C)=(1,0,1)，概率 0.4 时为 (0,1,0)。

排序 ABC 的期望 AP 为 0.70；BAC 为 0.75。真正按边缘概率 (.6,.4,.6) 降序的 ACB/CAB，其期望 AP 为 11/15≈0.73333，也低于 BAC。

因此，单靠各候选边缘匹配概率排序，在任意标签依赖下不保证 expected AP 最优。第 8 节使用潜在查询类别生成整条相关性向量，显式保留了这种依赖。分类或 AUC surrogate 也不能自动当成 MAP 保证 [7]。

### 9.1 前次讨论中的 AP 错序界

固定二元相关性向量，有 M>0 个正例。第 i 个正例前有 b_i 个负例，K=Σb_i，则：

$$
1-\mathrm{AP}=\frac1M\sum_i\frac{b_i}{i+b_i}
\le\frac1M\sum_i\frac{b_i}{1+b_i}
\le\frac{K}{M+K}.
$$

最后一步用 b/(1+b) 的凹性。它是有效的 surrogate 连接，但降低这个上界不自动保证实际 AP 改善。因此本文的验收主张使用第 8 节的新旧实际 AP 差值，而不把错序上界下降当作替代。

## 10. 为什么不能直接宣布现有 NUS 协议已被认证

1. 原 head 是二元配对 Laplace 近似；第 8 节需要类别概率模型及有效集合，接口不同。
2. 原反馈反传会改变用于后验拟合的表征。固定教师的证明要求 ψ₀ 不随学生更新；若把全编码器纳入参数集合，投影通常成为非凸全局问题。
3. 原测试查询与图库来自配对实体。若条件信息包含图库类别与配对 identity，查询类别可能已被确定；不能继续未经证明使用“独立新查询类别概率”的条件模型。
4. 多个配对共享实体标签，不能把 N² 对当成 N² 条独立认证观测。
5. 按已知标签强行抽成正负 50:50 会改变条件观测机制；认证似然须处理该选择机制。
6. 同一已知标签反复训练不是新增证据。现有历史数据若已用于选择特征/模型，须处理选择依赖，不能倒排一个“看似预序”的历史。
7. “Universal” 不表示任意模型误设下都覆盖真实概率。逻辑回归损失对参数凸，不等于密度族对混合凸。
8. 对固定认证查询池的条件期望保证，不是某次已实现的测试 mAP 保证，也不是任意未来查询总体的自动泛化保证。

无假设的逐点条件概率精确收紧本身受到已知不可行性结果约束 [8]。因此不能通过把一项假设隐去，宣称同时得到无分布假设、逐对高覆盖和必然收窄。

## 11. 对原动机的完成度

### 已闭合的条件性命题

- 有效集合的来源：给出鞅构造和可计算 logistic 集合，而非仅假设集合有效。
- 有依据的两端收紧：运行交集给全时间覆盖与嵌套。
- 不确定性意义：原 EU 给 Bayesian 条件期望收缩；信息半径给逐路径的最坏预测遗憾收缩，两者明确分开。
- 与实际检索连接：保留 cosine 排序，比较同一概率机制下的新旧 AP 差值，给精确 LP 证书。
- 非空性：原模型对应零差值；存在可手算的严格正改进例。

### 没有完成的现实结论

- 未证明 NUS 的真实条件分布属于所选教师族。
- 未对原 1000 个配对测试查询给出新的覆盖或 mAP 证书。
- 未证明当前数据足以使高维置信集合窄到有用。
- 未证明一定存在可认证的正收益更新，或非凸训练一定找到它。
- 未证明现有直接下界辅助损失应当继续调参。
- 未主张这些统计与信息论工具本身是新贡献。

下一次接入应先检查观测机制、固定教师、区间可用性及真实排序差值证书。若有效集合很宽，或所有非平凡候选的差值下界≤0，应如实报告当前证据不足；不能靠缩小集合、删除样本或改口径制造成功。

## 12. 可复跑的数学核验

核验脚本使用 numpy/scipy，包含信息半径的不同计算路径、Beta 后验期望收缩恒等式、类别概率盒 LP 与通用线性规划、相关性反例、有限二元排序枚举，以及标量混合置信序列的有限时域检查。

在仓库根目录运行：

    python fume_bounds/theory/20261110_evidence_bounds/verify_claims.py

输出见 verification.json。所有数值均为数学构造或金标准核验，不是 NUS 训练指标。有限网格检查不替代覆盖定理的证明。

## 13. 文献及使用范围

[1] Wasserman, Ramdas, Balakrishnan. Universal Inference. PNAS, 2020. §7, Theorem 11：预序似然比与置信序列运行交集。  
https://pmc.ncbi.nlm.nih.gov/articles/PMC7382245/

[2] Clerico, Flynn, Kotłowski, Neu. Confidence Sequences for Generalized Linear Models via Regret Analysis. 2025 预印本，Theorem 4.1；本文按已核对 v1 引用，不声称最终发表状态。  
https://arxiv.org/html/2504.16555v1

[3] Lee, Yun, Jun. A Unified Confidence Sequence for Generalized Linear Models, with Applications to Bandits. NeurIPS 2024：GLM 置信序列及参数范围、曲率与效率条件。  
https://arxiv.org/html/2407.13977v2

[4] Nakiboğlu. The Rényi Capacity and Center. IEEE Transactions on Information Theory, 2019：容量与信息半径的 minimax 等价；本文给二元特例的直接证明。  
https://doi.org/10.1109/TIT.2018.2861002  
https://users.metu.edu.tr/bnakib/aux/j/The_Rnyi_Capacity_and_Center.pdf

[5] Hüllermeier, Destercke, Shaker. Quantification of Credal Uncertainty in Machine Learning: A Critical Analysis and Empirical Comparison. UAI 2022：概率集合的不确定性及分解困难。  
https://proceedings.mlr.press/v180/hullermeier22a.html

[6] Petrik, Ghavamzadeh, Chow. Safe Policy Improvement by Minimizing Robust Baseline Regret. NeurIPS 2016：直接比较同一模型下基线差值的思想先例；其 MDP 结果不能原样迁移到 FUME。  
https://proceedings.neurips.cc/paper/2016/file/9a3d458322d70046f63dfd8b0153ece4-Paper.pdf

[7] Yue, Finley, Radlinski, Joachims. A Support Vector Method for Optimizing Average Precision. SIGIR 2007：MAP 与分类/AUC surrogate 的目标差异。  
https://www.cs.cornell.edu/people/tj/publications/yue_etal_07a.pdf

[8] Barber. Is Distribution-free Inference Possible for Binary Regression? Electronic Journal of Statistics, 2020：条件概率分布无关区间的非消失长度限制；不等于所有二元概率估计都不可能。  
https://arxiv.org/abs/2004.09477

[9] Houlsby, Huszár, Ghahramani, Lengyel. Bayesian Active Learning for Classification and Preference Learning. 2011：信息增益与预测熵背景；第 2 节等式按固定查询条件独立假设直接推导。  
https://arxiv.org/abs/1112.5745
