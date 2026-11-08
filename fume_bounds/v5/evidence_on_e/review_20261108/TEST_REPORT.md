# 实际检查记录

环境：Python 3.13.5；PyTorch 2.10.0+cpu；CPU；无 GPU 训练。

实际执行：

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_query_mixup.py
    18 passed in 1.50s
    python check_mechanisms.py --output CHECK_RESULTS.json

包括：两方向查询保持不变、lambda=0/1端点、软关系目标正确、上述双边Mixup冲突反例、固定查询候选顺序对称性、多标签标签不归一化、输入不变、随机种子复现、两模态特征梯度存在且有限、非法数据拒绝；浓度增大可降低期望CE并降低模型EU的数值例子。

未执行：完整核心训练器集成、真实 NUS/INRIA 训练、checkpoint评测、实际图库校准、AU/EU可控数据实验。

旧主训练路径未更改；本目录只包含 opt-in 模块与可运行检查，不声称自动修复真实 mAP，也没有证明论文创新性。
