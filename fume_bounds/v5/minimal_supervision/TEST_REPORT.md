# 本次交付验证 — 2026-11-07

环境：Python 3.13.5，PyTorch 2.10.0+cpu，NumPy 2.3.5，pytest 9.0.2；CUDA不可用。

实际执行：

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_minimal.py
14 passed in 1.69s
```

覆盖：直接隶属度监督的数值、固定W不更新、不调用FUME.forward/c、batch1/2/7/12下原对比算术与稳定等价式的值/梯度、拒绝multi-hot/软标签/无标签/NaN、梯度探针不更新参数或污染.grad、真实小型Adam更新两塔、零残差有限子梯度、全查询AP独立循环对照、评测恢复mode与CPU随机状态、合成训练末批不丢弃、新目录保护、续训2轮与连续2轮模型逐tensor一致、显式控制最终测试。

端到端测试注入的是小型双塔和合成TensorDataset，验证训练流程，**不是归档原模型或真实MAT/GPU验收**。正式入口会导入仓库已有的归档原版model.py以及既有v5/data.py，不替换其文件；`--require-ab150` 在执行端校验初值、源码、真实数据与逐轮顺序。

另实际运行 `check_mechanism.py --output <新文件>`，结果见 mechanism_check_recheck.json。常数参照与上一轮一致；100倍范数缩放的梯度检查最大绝对误差约4.07e-20；合成完全同向状态最大梯度约7.31e-20；单正标签构造最大残差约8.11e-16。极小浮点值可能随环境改变，不视为真实学习增益。

没有下载或重算Release NPZ，没有读取用户真实特征或checkpoint，没有重跑A/B/C/D，没有运行E的真实INRIA实验。没有新mAP或性能提升声明。当前只完成代码与文档交付，服务器执行结果须另行归档。
