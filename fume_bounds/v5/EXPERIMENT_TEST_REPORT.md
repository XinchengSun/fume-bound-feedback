# v5 实验扩展验证 — 2026-11-06

父提交：8d9cb43c3e20857d0b0b12ad2463c9bcb2f1f4db。
环境：Python 3.13.5、PyTorch 2.10.0+cpu；CUDA不可用。

## 本轮执行：26项新测试通过（分组执行）

- `pytest -q tests/test_experiment.py -k 'not cli_train and not validation_cadence'`：23 passed，3 deselected，9.43s。
- `pytest -q 'tests/test_experiment.py::test_cli_train_resume_select_and_explicit_test[shared]'`：1 passed，13.16s。
- per_label相同测试 + validation_cadence测试：2 passed，20.22s。

前两次尝试整套运行触及工具的20/60秒执行上限，未算为通过；以上是拆分后实际完成的结果。
没有借用旧33项报告作为本次测试计数，也没有声称重跑全部历史套件。

## 检查内容

1. 共享/逐类预算的+1、形状、多标签均值、零证据Beta(1,1)。
2. 共享结构直接载入原v5模型，前向所有输出逐tensor完全一致。
3. 禁用新机制时，损失和所有参数梯度与原v5完全一致；epoch1的KL零权重也一致。
4. KL解析简式与torch.distributions.kl_divergence及梯度一致；屏蔽正确证据，拒绝软标签。
5. KL退火边界、极小/很大错误证据的有限值、非法设置拒绝。
6. 意见一致性区分同均值不同证据，双方梯度可达。
7. 全模块开启时，分段/联合反传梯度一致；两塔与两证据层均有真实Adam更新，固定W不动。
8. 逐类别模式不改变初始共享编码器、W或后续随机序列。
9. 验证恢复训练/评测模式和随机状态；改变验证频率不改变训练轨迹。
10. 13条样本batch5不丢尾批；训练/保存/独立测试执行成功。
11. 两种证据模式下，1轮+恢复到2轮，与连续2轮模型及Adam状态逐tensor一致。
12. best按val选；训练不产生test结果；test需显式命令；val与test完全相同则拒绝。
13. 调参计划有配置数上限且可复现，只生成命令、不启动训练。
14. 汇总按同一验证轮报告各指标，不读取测试指标选最优。

## 完整尺寸一步检查

完整4096/1000输入、4096隐藏宽、4隐藏层、100输出；122396674参数。
在CPU上以合成batch2完成BCE、CL、Mixup、noise、KL与可选一致性的一次Adam更新。
两塔最后一层梯度范数约0.52847/0.65684；两预算bias约0.03295/0.03278；
四个监测参数均有约5e-5最大变化，W不变。原始数值见experiment_smoke.json。
该本机一步用时不是对服务器完整训练时间的估计。

## 复用文件的字节核验

本地测试使用的以下文件与远端父提交Git blob SHA完全一致，提交中不修改它们：

- model.py: 4f6c761eccc038b048c38a77b63633faed1ba179
- losses.py: 261e8ea2b99066264b1537d185ef81674c9443f0
- data.py: 533956cca15e0fd7fdf417915f92b949e381b809
- metrics.py: 330eee718b5364353d0f63b9eab33a5994c4623a

未完成真实INRIA、GPU、完整训练、OOD或超参数搜索；不声明mAP/Brier/NLL/AU/EU提升。
新增KL与跨模态意见一致性是任务适配，非TCL官方实现或其完整训练复现。
