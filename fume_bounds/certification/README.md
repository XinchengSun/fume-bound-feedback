# 冻结 NUS 表征下的认证可行性审计

[八批实验复跑](replay/README.md) · [新实体资格与概率观测机制](source_intake.md)

[本次决定与分层验收](records/20261110/REPORT.md) · [概率对象](records/20261110/estimand.md) · [规范](../docs/updates/2026-11-10-probability-certification-requirements.md)

本次运行范围为 P0a。复跑审计需有权读取与归档 SHA-256 对应的 NUS HDF5、第 6 轮检查点和历史 features.pt。以下命令从本目录执行，环境变量为各自的受控文件路径，OUT 为新的空目录。历史特征缓存由上一轮的已审计提取器生成。

~~~sh
mkdir -p "$OUT"
python audit_nus.py --dataset "$NUS_H5" --cache "$FROZEN_CACHE" --checkpoint "$CHECKPOINT" --out "$OUT"
python scalar_gold.py --out "$OUT/scalar_gold.json"
python ../tests/test_certification_coin.py -v
~~~

标量程序用精确有理数比较似然并向外二分，另以 80 位十进制对数似然及边界解析式复核。结果 JSON 中的浮点样例供阅读，精确有理数括区由 coin_bounds 返回。本阶段没有高维认证 API。

