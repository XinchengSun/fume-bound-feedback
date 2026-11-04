# 给执行端：v4改变参考规则与反馈梯度，不改事务控制器

## 0. 保留现场，先区分可行性错误

保留现有控制器、control_adapter.py、32次尝试日志和83轮checkpoint。
不要重复原8档学习率，不调大κ，不删类别约束，不降低求解器精度以制造接受。
本地未取得最新adapter，下方函数名属于接入接口，不能声称与服务器源码已经逐行匹配。

若要查旧规则为何失败，用当时保存的固定风险二次式和本次完整阈值：

```python
from live_reference import phase_one
result = phase_one(old_risk_system, candidate_current_risks + kappa,
                   start=old_witness)
print(result.as_dict())
```

- feasible：返回新的可行witness。不要因旧witness失败就宣称集合为空。
- infeasible_certificate：正的数值对偶侧下界支持集合为空（浮点证书，不是形式认证）。
- unresolved：无法判定，保留证据；不能当成证明不可行。

## 1. 在独立输出目录建立v4基线

候选族从“epoch80固定输出”改成“冻结的4组头作用于当前编码器”。
这是新参考目标，协议hash/version必须更新。先在83轮前的原状态重算v4基线。
**不得将v4新目标与v3旧目标的数值差记为增益。**

不改变固定risk样本ID及顺序、标签、8个事件及权重、κ向量、风险预算、当前组合alpha0、全量测试口径。
当前预测规则必须就是alpha0对4组头的凸组合；若旧实现另有融合/温度/取最大规则，先修正一致性，不能偷偷改定义。

## 2. 初始化（只一次）

```python
from live_reference import LiveReference, Event

problem = LiveReference(
    full_risk_labels,                  # [2700,100], 原one-/multi-hot标签
    anchor_alpha=existing_alpha0,      # 原实际预测使用的混合，不擅自改成均匀
    events=existing_fixed_events,      # Event(modality,index,class_index,target)
    event_weights=existing_fixed_weights,
    kappa=existing_kappa_vector,
    sample_ids=existing_ordered_risk_ids,
    heads_fingerprint=actual_frozen_heads_checkpoint_sha256,
    task=existing_task,
    gap_tol=existing_solver_gap_tolerance,
)
```

heads_fingerprint应包含头参数和固定输入标准化器；重试不更新这些状态。
冻结头参数用 requires_grad_(False)，梯度仍必须通过它的输入传到encoder。
不能对整个预测头调用 no_grad 或将 encoder 输出 detach 后再算训练loss。

## 3. 替换求界函数

```python
def solve_current_reference():
    # eval+no_grad只用于参考求解；每个调用重提当前encoder的完整risk预测。
    # 返回 [N,M,K]，两个bank必须来自同一当前状态、固定相同样本顺序。
    pi, pt = collect_current_banks_no_grad()
    snap = problem.solve(pi, pt,
        reported_current_risks=existing_full_current_risk_vector())
    return snap.reference_solve()
```

risk矩阵、事件目标系数、当前误差三者必须来自这次pi/pt。
不能重用第80轮bank；第80轮快照只能作为独立基线归档。
`reported_current_risks`是接口一致性检查，若不一致直接报错，而不是放宽κ。

## 4. 替换反馈loss：完整三项，不能再只用 wbar*R_current

最小正确实现（适合小批量验证而非2700条深网一次存图）：

```python
pi, pt = current_probability_banks_with_grad()   # 全risk行，graph连到两路encoder
snapshot = problem.solve(pi.detach().cpu().numpy(),
                         pt.detach().cpu().numpy())
loss_bound = problem.loss(pi, pt, snapshot)
loss = original_fume_loss + declared_beta * loss_bound
```

`loss_bound`中包括事件概率、候选风险、当前风险。
不要再归一化lambda而只保留一个w*R项。需要缩放时对**整个**loss_bound乘同一个预先声明的正数，记录协议。
旧版100倍经过归一化的Brier与本版未归一化完整梯度并不等尺度，不能写成“只修bug、所有loss超参含义不变”。
不声称beta默认值能改善真实数据；先核查两路梯度有限、正确，再使用既定训练预算作同条件对照。

### 大模型内存：概率空间梯度+分块重放（已测试的等价一阶计算）

```python
# 在一次optimizer.step之前，同一模型状态：
pi_np, pt_np = collect_current_banks_no_grad()  # [2700,4,100]
snapshot = problem.solve(pi_np,pt_np)
pi = torch.from_numpy(pi_np).to(device)
pt = torch.from_numpy(pt_np).to(device)
gpi, gpt = problem.coefficients(pi,pt,snapshot)

optimizer.zero_grad(set_to_none=True)
# 先对原FUME损失backward，或随后再累积，按既定批量协议缩放。
original_loss.backward()
for ids in complete_risk_microbatch_plan:
    pi_chunk, pt_chunk = current_banks_with_grad(ids)
    surrogate = declared_beta * (
        (pi_chunk*gpi[ids]).sum() + (pt_chunk*gpt[ids]).sum())
    surrogate.backward()
optimizer.step()  # 全部risk行累积完，才走一次真实step
```

`gpi/gpt`已包含对完整N的平均、事件权重及各项乘子，不要再除一次N。
重放过程中不能先step或改变模型参数/预测头、标准化器。含Dropout/BN时必须保证两次预测一致；本轮核对的官方FUME MLP没有BN/Dropout，但新增头的情况仍需检查。
`.eval()`不等于禁用梯度。训练图与求解图应采用一致的确定性预测模式。
分块重放的梯度数值必须与完整图版本做小规模对照，然后再用大MLP。

## 5. v3控制器不变，重新验证同一v4目标

仍使用run_encoder_segment/TransactionalController，仍在最多8次真实减半步长后停止。
继续核查：
- new.lower - old.upper > existing_min_gain；
- 完整risk平均值及累计分项损害满足原预算；
- 参数变化与Adam轨迹一致；
- 未做事后插值；
- 新可行见证的每项slack = κ（数值误差内）。

首个正确性版本每个optimizer step更新snapshot与梯度；4步短段结束后再验收。
若后续为速度而固定4步的内层解，属于stale surrogate：loss接口允许verify_snapshot_values=False，
但不再声称后3步是精确envelope梯度，且仍须完整重新求解终点。首先不要做这个近似。

## 6. 记录与验收

另外记录：reference_version、bank_hash、head_fingerprint、anchor_alpha、witness_slack_min、
per-event求解gap、三项梯度范数、实际参数差、风险变化、v4新旧目标的数值包络。
κ不变；头参数不变；全部样本不筛选；不拿test mAP决定接受。

20项测试和toy demo只支持实现检查。不能写成真实83轮回放、真实INRIA提升或总体不确定性分解已解决。
若v4仍被拒绝，必须明确记录拒绝在风险/目标改善/数值精度哪一项，不再把所有问题统称“候选不可行”。
