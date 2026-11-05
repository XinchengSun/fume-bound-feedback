# FUME 下界反馈：当前源码

当前版本为 **v4 当前表示参考族 + v3 事务式更新控制**。此目录替代此前 ZIP 交付；后续直接更新本仓库。

## 获取与自检

```bash
git clone https://github.com/XinchengSun/fume-bound-feedback.git
cd fume-bound-feedback/fume_bounds
python -m pip install -r requirements.txt
python -m pytest -q
python demo.py
```

已有仓库更新：

```bash
git pull --ff-only origin main
```

## 文件与接入

- `live_reference/`：当前编码器参考族、完整反馈梯度、旧族可行性诊断。
- `fume_control/`：新旧数值包络验收、真实缩步重试、模型/Adam/RNG 一致回滚。
- `tests/`：合并 v3 的 36 项与 v4 的 20 项回归测试。
- `demo.py`：合成双编码器示例，不是 INRIA。
- `replay_reported_rule.py`：审计摘要的规则回归，不是原 checkpoint 重放。
- `INTEGRATE.md`：执行端替换求界函数与新增损失的完整接口。
- `CHANGELOG.md`、`docs/updates/`：每次修改说明与交付记录。
- `TEST_REPORT.md`：原 v4 的实际测试范围和限制。

先阅读 **[INTEGRATE.md](INTEGRATE.md)**。复制模块到现有训练器目录可用：

```bash
python install_reference.py --target /path/to/current_encoder_trainer
```

安装器不覆盖不同源码，也不修改未知的 `control_adapter.py`、数据或 checkpoint。本仓库尚未包含服务器上的该适配脚本；需要按接入说明连接现有训练器的两个接口，不能直接声称完整 INRIA 程序已部署。

## 当前修订改变了什么

预测头参数冻结，但四组候选概率每次由当前编码器重新计算。当前实际混合是显式可行见证；反馈梯度同时包含事件概率、候选风险、当前风险。v3 接受/回滚规则不变。

**这是参考目标的修订，不保证候选集合逐轮嵌套。** 从旧 checkpoint 开始时先重新计算 v4 基线，不把不同版本下界数值直接比较成学习收益。

## 实验边界

2026-11-04 仓库整理时在 CPU 实际运行合并测试：**56 passed**。不代表 GPU 或真实 INRIA 验证；尚未证明 mAP 提升、统计覆盖或认知/偶然不确定性正确分离。数据、大权重、私人路径和服务器实验输出不随源码上传。


## VLS 跨模态迁移（2026-11-05）

完整 VLS 初版见 [`vls_transfer/`](vls_transfer/)，包含强混合、噪声混合、二元 EDL、FUME 原损失及训练入口。请先阅读其 `README.md` 和 `TEST_REPORT.md`；已有 v3/v4 下界反馈代码保持不变。
