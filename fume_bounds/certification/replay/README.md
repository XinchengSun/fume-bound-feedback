# 从受控检查点重放八批 NUS 单步对照

[检查点重放实测](verification/checkpoint_replay.json) · [88 个分支的起点](verification/branch_starts.json) · [原件与发布件哈希对应表](source_map.json)

从仓库根目录运行以下命令。每个输出目录须为新目录。第一条命令仅需 Python、NumPy、Pandas，不导入 Torch，不需要服务器或数据。

~~~sh
python fume_bounds/certification/replay/run.py summary --out summary-replay
python fume_bounds/certification/replay/run.py replay --checkpoint "$CHECKPOINT" --dataset "$NUS_H5" --out checkpoint-replay
~~~

第二条使用受控的原始 best.pt 与 NUS HDF5。[输入 SHA-256 和实际依赖版本](verification/checkpoint_replay.json)；CPython 3.12 环境、CPU、3 个 Torch 线程。脚本首先核验输入与发布源码，建立独立工作目录，载入检查点和 W，按历史划分提取倒数第二层激活，再运行原机制脚本。原始数据与权重由操作者提供。checkpoint 已按验证选择第 6 轮；选轮曲线、原始 result.json、历史 GPU 重放入口、相对上游的日志补丁与 audit_io.py 均在 sources/。当前 CPU 入口从这个已有检查点运行，无需重训前 100 轮。GPU 历史 replay.py 作为原件保留，需要其历史曲线及 saved 目录；使用本页 CPU 命令即可完成本次八批重放。

## 源码与状态

mechanism.py、experiment.py、summarize.py 及历史 replay.py 等均从原工作目录取回。extract.py 仅把运行根路径改为环境变量，原件与发布件的哈希分别登记。sources/upstream 下 6 个模块取自原版 [FUME 固定提交](https://github.com/siyuancncd/FUME/tree/5dd7a306aa505d8f60265b0353ff98de19ad9815/2025-CVPR-FUME)，逐文件内容与服务器八批实际使用版本核对。源哈希证明文件身份；复跑数字另由 run.py 从实际结果读回比较。

固定设置见 [源脚本](sources/mechanism.py) 与 [历史配置](../records/20261110/provenance.json)。每批 300 个训练加载器行；八批间行不重复。支持对索引来自 NumPy 种子 41531，新增批次的专用 Torch 生成器种子 110104。每一批在同一权重上求各损失的梯度，各更新组分别克隆原双塔末层并创建空状态 Adam，lr=5e-5，betas=(0.5,0.999)。基础组 1 个、五个反馈各有单用与合用两组，每批 11 个更新。每组一步，验证和测试各 1000 个查询。终点评估之后既不把模型也不把 Adam 状态传给下一组。

observe.py 在实际 Adam.step 之前采集 88 份记录，核对两个权重、固定 W、Adam 配置及空状态、CPU Torch 与两种 NumPy 和 Python RNG 状态，并核对最终全部批次 ID。专用 Torch 批次生成器在分支开始前一次性生成索引。实际提取链使用的架构没有注册 buffer，W 为独立张量。隐层来自同一次冻结提取。这些是本次重放新增的证据，历史八批没有留下这些逐分支快照。RNG 状态比较限于同次重放各分支，不要求不同进程的未使用随机源有同一启动状态。

## 已运行的验收

完整重放重新生成了 178 行指标、16 行梯度摘要、24 行均值与 140 行配对统计。按主键逐列比较，四表最大绝对数值差均为 0，全部批次索引一致。CSV 快速入口单独重算后也得到零差异。比较使用绝对容差 1e-12，保留冻结行的缺失 loss。原四组学生结果及解析端点八批结果是两个实验。没有新增训练臂或使用测试集选择配置。
