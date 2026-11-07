# E＋证据接入测试（2026-11-07）

本地环境：Python 3.13.5；PyTorch 2.10.0+cpu；NumPy 2.3.5；SciPy 1.17.0。无 CUDA。

执行过的验证（本地已测试的同方案源文件）：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_on_e.py
    19 passed in 29.45s

核验：none/category/matching 的原始 FUME 双塔、W、初始 RNG、前向完全一致；关闭辅助损失后，与 minimal_control 的 E 监督、所有参与参数的梯度、一次 Adam 更新逐张量相同。category 与 matching 在分段/联合 backward 下保持一致，手动禁止 BCE 函数后仍能训练。涵盖标签交集正负例、无负例边界、尾块平均、Mixup/Noise 软目标、+1先验、Beta期望CE与masked KL、RNG与模型状态不受验证改变、三个模式各自的真实小型 CLI 训练及断点续训、末轮测试明确开关、拒绝覆盖及旧格式误续训。

完整尺寸的 matching 结构：图像4096维、文本1000维、隐藏宽4096、四隐藏层、输出100维、122,440,194参数；合成 batch=2 一次 Adam 更新完成，双塔最后层和 pair head 都发生变化，梯度为有限值。详细原数值可用本目录的 smoke.py 重新生成。

**尚未执行新的真实 INRIA/GPU 实验；上述测试不构成 AU/EU 识别正确、泛化或 mAP 提升的证据。**本次只新增新目录，没有更改原实验代码及归档。
