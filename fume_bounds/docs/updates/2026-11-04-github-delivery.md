# 2026-11-04：源码迁移到 GitHub

本次是代码交付方式调整，不改变 v4 参考目标或 v3 更新控制算法。

## 发布内容

`live_reference/`、`fume_control/`、四个测试文件、两个示例脚本、安装器、依赖和接入说明。
原 FUME 数据、服务器 checkpoint、私有运行日志和实验输出未上传。

## 本次实际验证

在整理后的 `fume_bounds/` 内运行 `python -m pytest -q`：56 passed，0 failed。
这是本地 CPU 回归，不是 GitHub Actions 结果，也不是 GPU/INRIA 训练。

## 执行端

更新仓库后先运行测试，再按 INTEGRATE.md 接入两个函数。安装器不会覆盖不同版本文件，保留现场后再处理冲突。
v4 改变参考族，旧 checkpoint 可作为初始化，但必须重新计算 v4 基线，不能拿旧版下界比较增益。
数据、事件、风险预算、κ 和当前混合按原实验协议明确记录，不因上传仓库而改变。

## 后续每次提交

同时更新 CHANGELOG.md，并在 docs/updates/ 中记录修改原因、涉及文件、接口/数学目标变化、实际测试、未验证内容及 checkpoint 兼容性。最终运行版本用 commit SHA 固定。
