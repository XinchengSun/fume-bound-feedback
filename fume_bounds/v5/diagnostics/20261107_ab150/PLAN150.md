# INRIA 上的 FUME / v5 对比方案，已批准执行

2026-11-07。GitHub main 固定为 2a8980a3ab85d35da68660db79813387016047c1。本地与服务器已快进到同一提交，服务器核对 72 个受管文件。服务器 CPU 的 tests/test_evidence_ab.py 得到 25 passed；GPU1 已执行不含训练的 CUDA 算术、加载训练入口、SciPy 与 h5py。用户已确认每组 150 轮；已启动独立的 GPU1 串行任务。

## 四组与比较目的

| 组 | 训练目标 | 主要比较 |
|---|---|---|
| A 原版 FUME | frozen_fume/model.py 的 FUME 和 processor.calc_loss，模糊监督加论文实现的对比损失 | 检验完整新方法相对原方法的检索变化 |
| B 普通监督 | direction multi-hot BCE 和对比损失，类别支路的 vicinal、noise、kl、consistency 权重为零 | 给 C、D 提供普通监督对照 |
| C v5 类别证据 | category + direction，加类别 Mixup、噪声和错误证据 KL | C 减 B 衡量类别证据整套监督的作用 |
| D v5 匹配证据 | matching + direction，将证据目标换为图文对相关性 | D 减 C 对比两种证据方案，D 减 A 衡量相对原版的净变化 |

这里的 v5 不调用 v4 下界求解器。本次不据此归因于 v4 下界反馈。C 与 D 都以 sigmoid(h@W) 做普通分类。D 的关系头、监督对象、增强样本量一起变化，因此 D 减 C 衡量整套替换，不能单独归因为证据作用位置。

## 训练与验收

INRIA-Websearch.mat 的 SHA256 为 aae9ec202e06080b3b3c7c5ecab01ae1aa69483eb7ec93e2b47d65590cf50dbf。沿用 train / val / test 各 9000 / 1332 / 4366 条；0 起始编号、100 类，每条一个正类。首轮 seed 2，从头训练，统一初始双塔权重及固定 W，独立生成器重放相同的逐轮原始样本顺序。batch 300，每组 150 轮、4500 次 Adam 更新，学习率固定 5e-5，对比项权重 0.05。串行使用已分配的物理 GPU1。模型不同，另外记录参数量、逐轮耗时。150 轮为固定预算的首轮比较；不作收敛性能或跨 seed 声明。

B/C/D 使用仓库 run_evidence_ab.py。C/D 的 vicinal/noise=1、KL=0.01、10 轮退火、consistency=0，D 的 overlap 正例、每锚点 1 个无交集负例、pair_chunk=128。原版 A 复用独立冻结源码的模型和原损失，训练外层需与本次数据迭代和公共余弦评测接好。正式更新前先校验 A 与 B/C/D 初始编码器和 W 哈希、首批编号，检查 A 调用原始损失、共同 mAP 实现对同一特征返回相同结果，满足后再做正式训练。逐轮保存可核对的数据顺序哈希。

## 判读和归档

预先固定以第 150 轮权重分别对 test 评测一次。用全部 4366 个查询和各自方向的完整候选池、余弦排序、标签交集判相关。主指标为双向平均 mAP，列出两方向、D−A、D−C、C−B、D−B 的百分点差。每轮保存实际训练损失均值和验证集两个方向的 mAP，图按原始轮次点绘制，同时交付 JSONL/CSV。冻结配置，训练期间只看 val。匹配概率的 Brier/NLL 另列，使用固定均匀抽取的 20000 对，和检索 mAP 独立；类别与匹配的原生不确定性不跨事件比较。历史 56.7381% 单列为旧协议记录。多种子、长训另行决定。

## 运行位置

共享目录 /inspire/ssd/project/video-generation/public/sunxincheng/FUME/local_ops/v5_20261106。
GPU 作业 job-ecb844ba-8556-40b0-b386-257ad322d9ed，worker-0，HIGH/10。使用 GPU1_HANDOFF.md 的 with-one-gpu.sh 1 调用 run-python.sh。CUDA 唯一可见的卡为 GPU-fa8cd361-887f-8284-92e2-737dc0ac6d35。验证回执见 validation/ab-2a8980a-tests.log 和 validation/ab-2a8980a-gpu1.log。本次运行目录为 /inspire/ssd/project/video-generation/public/sunxincheng/FUME/runs/inria_ab150_seed2_20261107。每组 history.jsonl、validation.jsonl 按轮写入；完成时生成 raw_curves.csv，四组依次更新 test_map.json。
