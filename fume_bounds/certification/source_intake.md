# 新来源入库和逐对概率对象

## 实体资格的计算

~~~sh
python fume_bounds/certification/eligibility.py fume_bounds/certification/historical_manifest.json --details historical-rows.jsonl
python fume_bounds/tests/test_certification_eligibility.py -v
~~~

[清单实例](historical_manifest.json)使用按 split 展开的加载器行和原哈希锁定的历史角色记录；head_fit 的 2,048 个行号通过已存 pair 索引加入。[结果](replay/verification/eligibility.json)是按已登记的历史用途计算的 10,000 个 ineligible 行。全部行的规范实体 ID 及别名审计仍缺失。已确认用于拟合或方法选择足以排除资格，即使底层身份信息缺失。该计算复用原历史登记，不构成新的独立训练史证明。

新清单使用 records 数组。每行含唯一的 record_id、candidate 布尔值、modality（image/text/image_text）、canonical_id、source_block、aliases，以及 uses 数组（每项含 role、ref）。历史用途行也需导入，对非候选行设置 candidate=false。规范 ID 和别名使用一致的、带来源命名空间的 ID，裁剪、同图多文本等派生关系通过共同 ID、别名或 source_block 链接，排除用途沿这些链接传递。使用登记中的 encoder_fit、epoch_selection、method_selection、hyperparameter_selection、feature_fit、head_fit、feedback、test_query、gallery 均排除本次认证资格。

来源审核还需要 identity_review、history_review、population_review、prospective_isolation、observation_unit_review 五个证据引用。evidence 字典为每个引用提供 file（相对清单路径）及 sha256。新来源应提供可人工复查的底层记录，例如身份映射及派生关系、完整历史访问记录、目标总体/抽样框、事前隔离日期与冻结模型、配对和独立单位协议。仅给记录声明不等于统计证明；哈希只能验证附件身份。行的已用、未知、可入选及原因分别输出。共享来源块按块计一次，配对前仍须人工确认记录的完整性、时间顺序和独立性，并事前固定配对。未核实的证据不要登记为已审核。固定 10K 的历史结果归档保留原字节；新的 audit_nus.py 对该历史清单动态计算资格和角色重叠。

## 概率随机性需要单独验收

现有检索事件是两条记录的存档单类别相等。观察固定完整记录时该事件确定，标签机制没有提供对同一原始图文的重复语义判断。冻结特征上的条件群体平均率是另一个较弱对象。新增图文实体只能解决来源条件。

可供后续另行锁定的语义事件为：固定原始图文、呈现方式与问题，从预先界定的评审人群和抽样程序获取一次独立二元判断 Y，目标为该机制下 P(Y=1|原始图文)。需在采集前写明语言和人群纳入条件、独立分派、是否允许弃权及分母、重复评审单位与审计规则，并保留各对判断总数与正例数。现有 NUS 不具备此观测过程；该候选目标与 FUME 存档同类检索的关系还需验证。正确 logistic 设定、特征充分性以及适用于依赖结构的覆盖条件继续标为假设。
