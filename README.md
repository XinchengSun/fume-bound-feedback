# FUME bound-feedback

本仓库用于集中保存 FUME 下界反馈研究代码、接入说明、回归测试和版本变更，替代反复传递 ZIP。

主要目录：[`fume_bounds/`](fume_bounds/)。最新核心由 **v3 事务式更新控制**和 **v4 当前表示参考族**组成。原始 FUME 数据、权重及服务器上的训练结果不随源码发布。

```bash
git clone https://github.com/XinchengSun/fume-bound-feedback.git
cd fume-bound-feedback
# 后续更新
git pull --ff-only origin main
```

每次变更见 [`fume_bounds/CHANGELOG.md`](fume_bounds/CHANGELOG.md)。请按接入说明检查参考目标版本，不能把不同版本的下界数值直接比较成训练收益。

本仓库不声称已经完成真实 INRIA 上的性能提升；合成测试与真实实验必须区分记录。
