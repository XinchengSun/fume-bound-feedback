# M1：E＋匹配证据，单独关闭 KL

本目录是已经确认的 **M1 单因素实验**，不是新的模型设计，也不会自动继续跑 M2/M3。

## 本次只允许一个改变

当前已报告的 E＋匹配证据训练（INRIA，seed2，GPU1，150轮）使用：

| 参数 | 旧 E＋匹配 | **M1** |
|---|---:|---:|
| Clean E 隶属度＋FUME对比损失 | 不变 | 不变 |
| 匹配头 / +1 Beta 先验 / 采样规则 | 不变 | 不变 |
| Mixup 证据损失权重 | 1 | 1 |
| Noise 证据损失权重 | 1 | 1 |
| **错误证据 KL 权重上限** | **0.01** | **0** |
| 原始配对证据 CE 权重 | 0 | 0 |
| 种子 / batch / epoch / 训练步 | 2 / 300 / 150 / 4500 | 不变 |
| 学习率 / Adam betas | 5e-5 / (0.5,0.999) | 不变 |
| mAP | 余弦 z，全部4366条查询 | 不变 |

最终目标必须是：

    L_M1 = L_E + 1 * L_EDL_Mixup + 1 * L_EDL_Noise

无 KL；原始 Pair CE 也没有启用。这里所谓 Pair 是构造相关/不相关图文对，供两路增强训练；不是偷偷启用原始 Pair CE。辅助证据仍为 Beta expected CE，不替换原始 E 损失。

**旧实验的已报告末轮测试结果（单位百分数）：**
- E：图→文55.2771，文→图57.0841，均值56.1806。
- E＋匹配证据（KL0.01）：图→文53.6494，文→图55.8080，均值54.7287。

M1 尚无真实测试结果，不要预先填写 mAP。最重要的对照是 M1 相对旧匹配组的差值，同时报告相对 E 的差值。

## GPU1 操作

首先在目标服务器对应仓库中拉取最新提交，确认当前 GPU1 对应的 4090 UUID，并运行：

    git pull --ff-only origin main
    cd fume_bounds/v5
    python -m pytest -q evidence_on_e/tests/test_on_e.py

之前四组验收回执中的 GPU1 UUID 是 GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35；以实际机器上本次核验为准。指定一个唯一可见 GPU，脚本将强制检查 CUDA 可用、只看得到一张卡、卡名包含4090：

    CUDA_VISIBLE_DEVICES=GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35 \
      bash evidence_on_e/experiments/M1_no_kl/run_M1.sh

脚本默认读原 INRIA 数据文件并写到一个新的运行目录。也可以给两个位置参数（不改任何训练参数）：

    CUDA_VISIBLE_DEVICES=GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35 \
      bash evidence_on_e/experiments/M1_no_kl/run_M1.sh \
      /path/to/INRIA-Websearch.mat /path/to/NEW_M1_RUN

运行脚本调用原有 evidence_on_e/run.py，**仅把 --error-kl 设置为0**，显式固定其他关键配置；使用 --require-ab150 校验原始数据哈希、共同双塔与 W 初值、每轮原始样本顺序。核验失败时停止，不能强行跳过再声称同条件对照。

为了避免覆盖已完成的结果，运行目录及旁边的 .console.log 都必须不存在。命令会同步执行150轮和第150轮一次全量测试；训练过程直接保存每轮 JSONL、梯度记录与 checkpoint。当前 ChatGPT 会话**没有直接登录这台服务器 GPU1 的执行通道，所以 GitHub 提交脚本不等于 GPU 已经开跑**。

## 训练后自动核验

原命令跑完后将自动执行 verify_M1.py，它不会再次训练、不会修改 checkpoint，也不会用测试结果选模型。严格核查：

1. 解析后的参数必须为 matching、seed2、batch300、Mixup1、Noise1、KL0、Pair CE0；
2. 历史记录必须为完整150轮，每轮9000样本/30个更新，合计4500步；
3. KL 系数在每轮均为0，且没有实际加权错误证据损失；
4. 150轮原样本排序哈希必须与已归档的 A 组逐轮一致；
5. 验证集每方向1332条，最终测试每方向4366条，均无筛选；
6. 梯度记录每轮首批都有图像、文本输出层记录；
7. 生成与旧 E、旧匹配组的百分点差。

自动生成：
- history.jsonl：150轮分项原始训练损失；
- validation.jsonl：150轮原始验证 mAP；
- gradient_probe.jsonl：150轮首批各损失加权梯度；
- order.jsonl、provenance.json、completion.json、last.pt；
- test_metrics.json：第150轮全量测试；
- **M1_raw_epochs.csv**：原始150轮训练损失、h范数与验证mAP，无插值无平滑；
- **M1_verification.json**：两方向 mAP、相对 E/旧匹配的百分点差、训练时长及核验结果；
- 目录旁边的 .console.log：完整终端输出。

后续如果 M1 改善，再分别测试只用 Mixup 或只用 Noise；**这一轮不自动触发其他实验**。不要在 M1 中再修改 pair sampling、证据头、增强 Beta 参数、对比权重或读出。

**验证边界**：run_M1.sh 已经做过 Bash 语法检查；verify_M1.py 通过了 Python 语法与150轮合成日志验证。本轮没有 GPU 实测；历史 mAP 来自此前用户提供的 GPU1 记录，而不是本脚本生成。
