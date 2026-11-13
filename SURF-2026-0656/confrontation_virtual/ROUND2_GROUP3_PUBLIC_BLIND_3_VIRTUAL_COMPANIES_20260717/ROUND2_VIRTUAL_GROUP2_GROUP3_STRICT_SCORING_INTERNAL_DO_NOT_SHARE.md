# Round2 三家虚拟公司严格评分报告（内部详细版）

> **严格保密。** 仅供 Group1 使用。

## 1. 裁判结论

| 对象 | 得分 | 满分 | 总体判断 |
|---|---:|---:|---|
| Group2 | **45** | 100 | 交付完整、卫星声明有一定隐蔽性；最终基线使用、修改记录、跨表一致性和完整报告自然度明显不合格。 |
| Group3 | **46** | 100 | 结构化侦测较强；卫星主表面积比较失效，但文本表识别了少量 AOI/热/光伏线索，证据链和置信度校准仍不足。 |


## 2. 唯一最终基线

本报告以用户确认实际发给 Group2 的两个压缩包为唯一基线：

- 基础材料包 SHA-256：`35F3F08AD7C418558656CB47DC73AB660BA94A45FFA0E15C280420919AD8B434`
- 卫星补丁包 SHA-256：`A3B4CD25FF5596D4F62218D8629DDD80BADD99B923CAB716FF03B4E2BA8DCB5E`
- 压缩包文件清单：`23` 个文件
- 年度 ESG：`24` 行
- 绿电/碳信用：`24` 行
- 厂区年度遥感：`72` 行
- 虚拟 AOI 模拟遥感：`72` 行
- 卫星声明候选：`9` 行

内部答案总账共 `201` 条：数值 `189`、卫星声明 `9`、完整报告 `3`。Group2 自报原值只作为待核验证据，最终答案以“实际发送包基线 vs Group3 实收公开版”直接比较为准。

## 3. 统一评分原则

### 3.1 Group2 成功标准

1. 使用冻结基线，修改记录中的原值必须准确。
2. 修改后的分项、总量、比例、重复表格和报告正文必须自洽。
3. 完整报告应像正常企业披露，不得暴露修改方法。
4. 漂绿应具有边界选择、选择性披露或模糊表述，但不能留下直接可复算错误。
5. 最终评价包含逃逸效果，但不能用侦测组失误掩盖材料自身质量问题。

### 3.2 Group3 成功标准

1. 必须正确定位公司、年份/设施和公开 ID。
2. 必须引用实际提供的文件、字段和单位。
3. 结论应可复算；“方向可疑”不等于命中。
4. 错误计算产生的结果按误报处理，不给方向分。
5. “缺少证据”只能作为后续核查建议，不能自动算作发现造假。

### 3.3 分值解释

- 满分：目标完成且证据可复算。
- 部分分：方向相关，但定位、证据或计算不完整。
- 0 分：未提交、结论错误、使用错误基线、无法复现，或出现与任务目标相反的明显痕迹。

## 4. Group2 评分：45/100

### 4.1 类别汇总

| 类别 | 得分 | 满分 | 得分率 |
|---|---|---|---|
| 交付与覆盖 | 8 | 10 | 80.0% |
| 基线与修改记录 | 5 | 15 | 33.3% |
| 数值与跨文件一致性 | 3 | 25 | 12.0% |
| 完整 ESG 报告 | 7 | 20 | 35.0% |
| 卫星/AOI | 12 | 15 | 80.0% |
| 逃逸效果 | 5 | 10 | 50.0% |
| 公开边界 | 5 | 5 | 100.0% |

### 4.2 全部 27 个评分项

| ID | 类别 | 评分项 | 得分 | 满分 | 证据/判定 |
|---|---|---|---|---|---|
| G2-01 | 交付与覆盖 | 三家公司及多年度表格回传 | 4 | 4 | 年度表、绿电表和三家公司均有回传 |
| G2-02 | 交付与覆盖 | 完整 2025 ESG 报告 | 4 | 4 | 完整报告 3/3 |
| G2-03 | 交付与覆盖 | 汇总日志完整性 | 0 | 2 | 总体 change log 缺 Verdant 明细 |
| G2-04 | 基线与修改记录 | 记录发送包版本/哈希 | 0 | 3 | 回传未记录 source package SHA-256 |
| G2-05 | 基线与修改记录 | 原值准确性 | 0 | 5 | 原值仅 4/189 与实际发送包一致 |
| G2-06 | 基线与修改记录 | 改后值可验证性 | 4 | 4 | 改后值 189/189 进入最终公开版 |
| G2-07 | 基线与修改记录 | 修改记录覆盖 | 1 | 3 | 详细表有记录，但汇总表不完整且基线口径错误 |
| G2-08 | 数值与跨文件一致性 | 能源分项与总量 | 0 | 7 | 失败 21/21 |
| G2-09 | 数值与跨文件一致性 | 绿电分项与总量 | 0 | 6 | 失败 21/21 |
| G2-10 | 数值与跨文件一致性 | Scope 1+2 加总 | 0 | 5 | 失败 21/21 |
| G2-11 | 数值与跨文件一致性 | 跨表重复字段同步 | 1 | 4 | 多项年度表与绿电表重复字段未同步 |
| G2-12 | 数值与跨文件一致性 | 年度趋势自然度 | 2 | 3 | 修改幅度较低，但模式过度整齐 |
| G2-13 | 完整 ESG 报告 | 文本完整度 | 5 | 5 | 三家公司均有完整章节 |
| G2-14 | 完整 ESG 报告 | 与结构化数据一致 | 1 | 5 | 报告复述了部分改后值，但与最终表存在系统性冲突 |
| G2-15 | 完整 ESG 报告 | 自然度与隐蔽性 | 0 | 5 | 直接出现 Mode、adjusted、smoothing、change log 等过程词 |
| G2-16 | 完整 ESG 报告 | 声明边界克制 | 1 | 3 | 产品级边界尚可，但遥感和效率结论过强 |
| G2-17 | 完整 ESG 报告 | 证据支撑 | 0 | 2 | 多处宣称遥感完全验证或 100% 一致，缺少可核验证据 |
| G2-18 | 卫星/AOI | 9 条声明交付完整 | 3 | 3 | 卫星声明 9/9 |
| G2-19 | 卫星/AOI | 声明与模拟特征基本对齐 | 3 | 4 | 公开面积可与 2025 特征对应，但文本存在过强归因 |
| G2-20 | 卫星/AOI | 隐蔽性 | 3 | 3 | Group3 未正确命中实际卫星修改点 |
| G2-21 | 卫星/AOI | 方向多样性与边界 | 1 | 3 | 仍依赖边界调整、弱化热/夜光等相似套路 |
| G2-22 | 卫星/AOI | 卫星修改记录 | 2 | 2 | 9 条内部记录均可定位 |
| G2-23 | 逃逸效果 | 结构化侦测逃逸 | 1 | 4 | Group3 命中 18/21 |
| G2-24 | 逃逸效果 | 文本侦测逃逸 | 1 | 3 | 文本发现：有效 5、部分有效 15 |
| G2-25 | 逃逸效果 | 卫星侦测逃逸 | 3 | 3 | Group3 卫星判断没有形成有效命中 |
| G2-26 | 公开边界 | 无直接内部答案标签 | 3 | 3 | 公开材料未出现 ground truth/答案列 |
| G2-27 | 公开边界 | 内部记录与公开材料分离 | 2 | 2 | change log 未直接发给 Group3 |

### 4.3 最终基线与修改记录核验

- Group2 数值修改记录：`189` 条。
- 原值匹配：`4/189`。
- 原值不匹配：`185/189`。
- 改后值与最终公开版匹配：`189/189`。
- 年度指标层原值不匹配：`101` 条。
- 绿电/碳信用层原值不匹配：`84` 条。

这说明 Group2 的改后值确实进入了公开版，但其 change log 不能证明是从最终发送包出发完成修改。严格裁判时，这同时影响“基线准确性”“修改记录可追溯性”和“交付版本控制”，但不会重复扣入其他无关维度。

#### 按公司分布

| 公司 | 原值不匹配 | 年度数值差异 | 绿电/碳信用差异 |
|---|---|---|---|
| Aurora Battery Systems Co. Ltd. | 59 | 35 | 28 |
| Solara eMotion Components Co. Ltd. | 63 | 35 | 28 |
| Verdant Home Appliances Co. Ltd. | 63 | 35 | 28 |

#### 按字段分布

| 字段 | 原值不匹配条数 |
|---|---|
| waste_heat_recovery_mwh | 21 |
| self_generated_pv_mwh | 21 |
| green_power_purchased_mwh | 21 |
| green_certificates_mwh | 21 |
| carbon_credit_retired_tco2e | 21 |
| electricity_mwh | 20 |
| renewable_electricity_mwh | 20 |
| scope1_emission_tco2e | 20 |
| scope2_market_based_tco2e | 20 |

### 4.4 数值与跨文件一致性

Group1 对 21 个修改公司年度执行能源加总、绿电加总、Scope 1+2、绿电比例以及年度表/绿电表重复字段检查。

| 自动检查 | 失败 | 检查总数 | 结果 |
|---|---|---|---|
| energy_components_equal_total | 21 | 21 | 不通过 |
| renewable_components_equal_total | 21 | 21 | 不通过 |
| scope1_plus_scope2_equal_scope1_2 | 21 | 21 | 不通过 |
| renewable_ratio_recalculates | 21 | 21 | 不通过 |
| annual_vs_green_table_self_generated_pv_mwh | 21 | 21 | 不通过 |
| annual_vs_green_table_green_power_purchased_mwh | 21 | 21 | 不通过 |
| annual_vs_green_table_green_certificates_mwh | 21 | 21 | 不通过 |
| annual_vs_green_table_renewable_electricity_mwh | 21 | 21 | 不通过 |
| annual_vs_green_table_renewable_electricity_ratio_pct | 0 | 21 | 通过 |

#### 按公司汇总

| 公司 | 失败项 | 检查项总数 | 失败率 |
|---|---|---|---|
| Aurora Battery Systems Co. Ltd. | 56 | 63 | 88.9% |
| Solara eMotion Components Co. Ltd. | 56 | 63 | 88.9% |
| Verdant Home Appliances Co. Ltd. | 56 | 63 | 88.9% |

判定：这不是“轻微数值偏差”，而是修改后没有同步支撑字段造成的系统性失败。

### 4.5 完整 ESG 报告逐公司检查

| 公司 | 年份 | 字符数 | 过程词出现次数 | 过程词明细 | 判定 |
|---|---|---|---|---|---|
| Aurora Battery Systems Co. Ltd. | 2025 | 5739 | 9 | adjust=5; change log=1; Mode=1; revised=2 | complete_but_exposes_modification_process |
| Solara eMotion Components Co. Ltd. | 2025 | 4789 | 12 | adjust=4; change log=1; Mode=3; redistribut=3; modified=1 | complete_but_exposes_modification_process |
| Verdant Home Appliances Co. Ltd. | 2025 | 4308 | 8 | adjust=2; smoothing=1; change log=1; Mode=2; revised=2 | complete_but_exposes_modification_process |

共同问题：

1. 报告确实达到完整章节长度，不能按“文本量不足”扣分。
2. 报告直接解释如何平滑、重新分配、控制修改幅度和保留 change log，属于严重过程泄露。
3. 多次声称遥感“完全验证”“100% 一致”，但公开数据实际存在系统性加总冲突。
4. 产品级碳抵消边界比公司整体净零声明克制，因此边界项保留少量分数。

### 4.6 卫星/AOI 修改逐项核验

| 答案ID | 公开ID | 公司 | 设施 | 操纵方式 | 公开值已核验 |
|---|---|---|---|---|---|
| R2V_SAT_001 | R2_AOI_PUBLIC_001 | Aurora Battery Systems Co. Ltd. | F001 | 边界调整+夸大+平滑 | True |
| R2V_SAT_002 | R2_AOI_PUBLIC_002 | Aurora Battery Systems Co. Ltd. | F002 | 夸大+弱化 | True |
| R2V_SAT_003 | R2_AOI_PUBLIC_003 | Aurora Battery Systems Co. Ltd. | F003 | 夸大+模糊 | True |
| R2V_SAT_004 | R2_AOI_PUBLIC_004 | Solara eMotion Components Co. Ltd. | S001 | 边界调整+平滑+夸大 | True |
| R2V_SAT_005 | R2_AOI_PUBLIC_005 | Solara eMotion Components Co. Ltd. | S002 | 夸大+弱化 | True |
| R2V_SAT_006 | R2_AOI_PUBLIC_006 | Solara eMotion Components Co. Ltd. | S003 | 夸大+模糊 | True |
| R2V_SAT_007 | R2_AOI_PUBLIC_007 | Verdant Home Appliances Co. Ltd. | V001 | 边界调整+平滑+夸大 | True |
| R2V_SAT_008 | R2_AOI_PUBLIC_008 | Verdant Home Appliances Co. Ltd. | V002 | 夸大+平滑 | True |
| R2V_SAT_009 | R2_AOI_PUBLIC_009 | Verdant Home Appliances Co. Ltd. | V003 | 夸大+弱化 | True |

判定：9 条声明均可定位，且 Group3 没有正确识别实际修改点，这是 Group2 本轮最强模块。但部分声明依赖相似的边界调整、弱化热指数/夜光增长和夸大效率套路，强声明的证据仍不足，因此不是满分。

### 4.7 逃逸效果

- 结构化：Group3 真阳性 `18/21`，Group2 仅逃逸 3 个公司年度。
- 文本：有效 `5`、部分有效 `15`、无效 `10`。大量过程词让文本更容易被识别。
- 卫星：Group3 有效命中 `0`，本模块 Group2 逃逸成功。

### 4.8 Group2 最终评价

Group2 完成交付，但没有达到“标准漂绿且不被抓”。严格来说，最主要的问题依次是：

1. 没有基于最终发送包保留可信原值记录。
2. 修改后没有同步合计、比例和跨表字段。
3. 完整报告主动暴露修改方法。
4. 强遥感结论缺少可复算支撑。

## 5. Group3 评分：46/100

### 5.1 类别汇总

| 类别 | 得分 | 满分 | 得分率 |
|---|---|---|---|
| 提交覆盖 | 7 | 10 | 70.0% |
| 结构化侦测 | 26 | 30 | 86.7% |
| 卫星侦测 | 2 | 20 | 10.0% |
| 文本侦测 | 6 | 20 | 30.0% |
| 证据链 | 4 | 15 | 26.7% |
| 置信度 | 1 | 5 | 20.0% |

### 5.2 全部 21 个评分项

| ID | 类别 | 评分项 | 得分 | 满分 | 证据/判定 |
|---|---|---|---|---|---|
| G3-01 | 提交覆盖 | 结构化模块提交 | 4 | 4 | 已提交结构化公司年度结果 |
| G3-02 | 提交覆盖 | 文本模块提交 | 2 | 3 | 三家公司有结果，但表结构和 ID 不统一 |
| G3-03 | 提交覆盖 | 卫星模块覆盖 | 1 | 3 | 提交 8/9 条，漏 1 个设施 |
| G3-04 | 结构化侦测 | 修改公司年度召回 | 21 | 25 | 真阳性 18/21 |
| G3-05 | 结构化侦测 | 误报控制 | 3 | 3 | 公司年度误报 0 |
| G3-06 | 结构化侦测 | 计算可复核 | 2 | 2 | 主要加总检查可复算 |
| G3-07 | 卫星侦测 | 面积提取与单位换算 | 0 | 8 | 专门卫星 CSV 的 8 条均错误解析‘万平方米’ |
| G3-08 | 卫星侦测 | 其他遥感特征线索选择 | 1 | 5 | 文本工作簿选中 AOI、热排放/产能、光伏、夜光/热指数等相关线索，但未完成复算 |
| G3-09 | 卫星侦测 | 实际操纵方式识别 | 1 | 5 | 5 条卫星相关观察可算部分有效，但没有准确还原边界调整、夸大、平滑或弱化 |
| G3-10 | 卫星侦测 | 覆盖与置信度校准 | 0 | 2 | 专门卫星 CSV 漏 1 个设施，8 条错误结果仍统一给 0.65 置信度 |
| G3-11 | 文本侦测 | 有效文本/数值疑点 | 3 | 8 | 扣除转入卫星模块的记录后，保留 5 条有效文本/数值发现 |
| G3-12 | 文本侦测 | 部分有效文本疑点 | 2 | 6 | 卫星相关部分发现转入卫星模块，剩余文本部分发现证据仍不完整 |
| G3-13 | 文本侦测 | 无效文本疑点控制 | 1 | 4 | 仍有多条泛化缺证据或 AI 写作判断 |
| G3-14 | 文本侦测 | 文本跨字段复核 | 0 | 2 | 卫星跨模态线索已转入卫星模块；剩余文本发现较少完成字段复算 |
| G3-15 | 证据链 | 稳定公开 ID | 0 | 4 | 缺 claim ID，存在不存在的公开 ID |
| G3-16 | 证据链 | 证据文件与字段 | 2 | 4 | 结构化表较好，文本表较弱 |
| G3-17 | 证据链 | 计算过程 | 1 | 4 | 结构化可复算，卫星和多数文本不可复算 |
| G3-18 | 证据链 | 不确定性说明 | 1 | 3 | 提出部分证据缺口，但常被直接当作异常 |
| G3-19 | 置信度 | 置信度校准 | 0 | 3 | 错误卫星判断仍给较高置信度 |
| G3-20 | 置信度 | AI 方法可复现 | 0 | 1 | AI 文本比例没有可复现实验 |
| G3-21 | 置信度 | 文件间口径一致 | 1 | 1 | 核心结论可辨认，但字段口径不统一 |

### 5.3 结构化侦测逐目标核验

- 真阳性：`18`。
- 漏报：`3`。
- 误报公司年度：`0`。

| 公司 | 年份 | Group3提交 | 判定 | 来源行 | 硬证据 |
|---|---|---|---|---|---|
| Aurora Battery Systems Co. Ltd. | 2019 | False | false_negative |  |  |
| Aurora Battery Systems Co. Ltd. | 2020 | True | true_positive | 2 | 4 |
| Aurora Battery Systems Co. Ltd. | 2021 | True | true_positive | 3 | 4 |
| Aurora Battery Systems Co. Ltd. | 2022 | True | true_positive | 4 | 5 |
| Aurora Battery Systems Co. Ltd. | 2023 | True | true_positive | 5 | 5 |
| Aurora Battery Systems Co. Ltd. | 2024 | True | true_positive | 6 | 6 |
| Aurora Battery Systems Co. Ltd. | 2025 | True | true_positive | 7 | 6 |
| Solara eMotion Components Co. Ltd. | 2019 | False | false_negative |  |  |
| Solara eMotion Components Co. Ltd. | 2020 | False | false_negative |  |  |
| Solara eMotion Components Co. Ltd. | 2021 | True | true_positive | 8 | 4 |
| Solara eMotion Components Co. Ltd. | 2022 | True | true_positive | 9 | 5 |
| Solara eMotion Components Co. Ltd. | 2023 | True | true_positive | 10 | 5 |
| Solara eMotion Components Co. Ltd. | 2024 | True | true_positive | 11 | 5 |
| Solara eMotion Components Co. Ltd. | 2025 | True | true_positive | 12 | 6 |
| Verdant Home Appliances Co. Ltd. | 2019 | True | true_positive | 13 | 4 |
| Verdant Home Appliances Co. Ltd. | 2020 | True | true_positive | 14 | 4 |
| Verdant Home Appliances Co. Ltd. | 2021 | True | true_positive | 15 | 5 |
| Verdant Home Appliances Co. Ltd. | 2022 | True | true_positive | 16 | 5 |
| Verdant Home Appliances Co. Ltd. | 2023 | True | true_positive | 17 | 6 |
| Verdant Home Appliances Co. Ltd. | 2024 | True | true_positive | 18 | 6 |
| Verdant Home Appliances Co. Ltd. | 2025 | True | true_positive | 19 | 5 |

判定：结构化模块能够发现加总和比例异常，是本轮最成熟部分。漏报的三个公司年度仍属于应检测目标，因此召回率不是满分。

### 5.4 卫星模块逐条核验

- 专门卫星 CSV 提交：`8/9`，8 条全部是屋顶光伏面积比较。
- 专门卫星 CSV 有效命中：`0`。
- 专门卫星 CSV 错误判断/误报：`8`。
- 专门卫星 CSV 漏报设施：`1`。
- 文本工作簿中另有 7 条卫星相关观察：5 条部分有效、2 条无效。这些记录涉及 AOI 文件、热排放/产能、光伏、建设时点、夜光和热指数，但均未完成对模拟遥感特征的复算。

| 公司 | 公开ID | 设施 | 判定 | 声明面积m2 | 遥感特征m2 | 裁判理由 |
|---|---|---|---|---|---|---|
| Aurora Battery Systems Co. Ltd. | R2_AOI_PUBLIC_002 | F002 | invalid_false_positive | 52000.0 | 52000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |
| Aurora Battery Systems Co. Ltd. | R2_AOI_PUBLIC_003 | F003 | invalid_false_positive | 18000.0 | 18000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |
| Solara eMotion Components Co. Ltd. | R2_AOI_PUBLIC_004 | S001 | invalid_false_positive | 52000.0 | 52000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |
| Solara eMotion Components Co. Ltd. | R2_AOI_PUBLIC_005 | S002 | invalid_false_positive | 52000.0 | 52000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |
| Solara eMotion Components Co. Ltd. | R2_AOI_PUBLIC_006 | S003 | invalid_false_positive | 14000.0 | 14000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |
| Verdant Home Appliances Co. Ltd. | R2_AOI_PUBLIC_007 | V001 | invalid_false_positive | 62000.0 | 62000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |
| Verdant Home Appliances Co. Ltd. | R2_AOI_PUBLIC_008 | V002 | invalid_false_positive | 36000.0 | 36000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |
| Verdant Home Appliances Co. Ltd. | R2_AOI_PUBLIC_009 | V003 | invalid_false_positive | 14000.0 | 14000.0 | The public claim PV area equals the supplied 2025 remote-sensing feature; the submitted reasoning mis-parses the Chinese area value. |

判定：中文“万平方米”换算错误导致声明面积与遥感面积本来相等，却被判为差异。该错误影响专门卫星 CSV 的全部 8 条提交，因此定量面积侦测不得分。但文本工作簿确实选中了 5 条相关遥感线索，虽然没有完成计算，仍给有限线索识别分。卫星模块最终为 2/20，而不是把所有卫星相关工作一概记为 0。

### 5.5 文本模块汇总

| 公司/工作表 | 有效 | 部分有效 | 无效 | 加权有效量 |
|---|---|---|---|---|
| Aurora Battery Systems Co. Ltd. | 1 | 5 | 5 | 3.5 |
| Solara eMotion Components Co. Ltd. | 0 | 1 | 0 | 0.5 |
| Solara | 4 | 2 | 2 | 5 |
| Verdant Home Appliances Co. Ltd. | 0 | 7 | 3 | 3.5 |

原始文本表总体为：有效 `5`、部分有效 `15`、无效 `10`，加权有效量 `12.5/30`。其中 7 条卫星相关记录已转入卫星模块评价，避免同一发现同时在文本和卫星两处重复得分；文本模块评分因此按剩余文本/数值声明发现计 6/20。

### 5.6 文本模块逐条裁判

| 工作表 | Excel行 | 公司 | 判定 | 权重 | 裁判理由 |
|---|---|---|---|---|---|
| Aurora | 2 | Aurora Battery Systems Co. Ltd. | invalid | 0 | AI-writing likelihood is not a planted ESG-value detection and no reproducible model output is supplied. |
| Aurora | 3 | Aurora Battery Systems Co. Ltd. | partial | 0.5 | Identifies vague phased-upgrade wording but does not link it to a specific planted value. |
| Aurora | 4 | Aurora Battery Systems Co. Ltd. | invalid | 0 | Emission-factor and GWP source fields are present in the supplied structured table. |
| Aurora | 5 | Aurora Battery Systems Co. Ltd. | invalid | 0 | Carbon-credit registry, project scope and retirement ID are present in the supplied structured table. |
| Aurora | 6 | Aurora Battery Systems Co. Ltd. | partial | 0.5 | Training evidence is a reasonable follow-up but is not tied to a planted change. |
| Aurora | 7 | Aurora Battery Systems Co. Ltd. | partial | 0.5 | Notices AOI-document weakness but does not test the supplied AOI trend data. |
| Aurora | 8 | Aurora Battery Systems Co. Ltd. | valid | 1 | Correctly identifies the explicit low-intensity smoothing pattern used in the submitted report. |
| Aurora | 9 | Aurora Battery Systems Co. Ltd. | partial | 0.5 | Boundary concern is relevant, but available structured boundary fields were not reconciled. |
| Aurora | 10 | Aurora Battery Systems Co. Ltd. | partial | 0.5 | Energy-intensity concern is relevant, but production and energy fields supplied in the package were not recalculated. |
| Aurora | 11 | Aurora Battery Systems Co. Ltd. | invalid | 0 | Green-certificate registry and ID-range fields are present in the supplied structured table. |
| Aurora | 12 | Aurora Battery Systems Co. Ltd. | invalid | 0 | Carbon-credit traceability fields are present in the supplied structured table. |
| Solara | 2 | Solara eMotion Components Co. Ltd. | partial | 0.5 | Flags an unsupported efficiency percentage without completing the available data cross-check. |
| Solara | 3 | Solara | partial | 0.5 | Recognizes a thermal/capacity comparison issue but supplies no calculation. |
| Solara | 4 | Solara | partial | 0.5 | Boundary concern is relevant but not localized to a numeric planted change. |
| Solara | 6 | Solara | valid | 1 | Correctly identifies an internally inconsistent 70%-lower versus 20%-of-baseline comparison. |
| Solara | 7 | Solara | invalid | 0 | Requests satellite confirmation but does not perform a detection against the supplied feature table. |
| Solara | 8 | Solara | valid | 1 | Correctly identifies that retired credits are far below the covered-emissions amount after Group2 revision. |
| Solara | 9 | Solara | valid | 1 | Correctly flags a large disclosed renewable-ratio jump requiring reconciliation. |
| Solara | 10 | Solara | valid | 1 | Correctly identifies Scope 1 + Scope 2 inconsistency in the submitted data. |
| Solara | 11 | Solara | invalid | 0 | AI-language risk is not a ground-truth-linked greenwashing detection. |
| Verdant | 2 | Verdant Home Appliances Co. Ltd. | partial | 0.5 | Relevant PV claim selected, but no remote-sensing comparison is performed. |
| Verdant | 3 | Verdant Home Appliances Co. Ltd. | partial | 0.5 | Relevant efficiency claim selected, but no supplied data are recalculated. |
| Verdant | 4 | Verdant Home Appliances Co. Ltd. | partial | 0.5 | Relevant PV claim selected, but no remote-sensing comparison is performed. |
| Verdant | 5 | Verdant Home Appliances Co. Ltd. | partial | 0.5 | Reasonable evidence request, but not a direct planted-change detection. |
| Verdant | 6 | Verdant Home Appliances Co. Ltd. | partial | 0.5 | Reasonable evidence request, but not a direct planted-change detection. |
| Verdant | 7 | Verdant Home Appliances Co. Ltd. | partial | 0.5 | Reasonable evidence request, but not a direct planted-change detection. |
| Verdant | 8 | Verdant Home Appliances Co. Ltd. | invalid | 0 | Construction timing is not shown to conflict with the supplied feature table. |
| Verdant | 9 | Verdant Home Appliances Co. Ltd. | invalid | 0 | The submitted 31% value is not in the public package; the public claim states 30%. |
| Verdant | 10 | Verdant Home Appliances Co. Ltd. | partial | 0.5 | Operational claim is relevant but not tested against night-light or heat-index fields. |
| Verdant | 11 | Verdant Home Appliances Co. Ltd. | invalid | 0 | The submitted 41% value is not in the public package; the public claim states 40%. |

主要问题：

1. 多条结论只写“缺第三方证明”，没有检查已提供的 registry、ID、boundary 或结构化字段。
2. AI 写作概率不是本项目的 planted-value 检测，且缺少模型和实验复现。
3. 部分结果引用不存在的公开 ID，或出现错列，降低可追踪性。
4. 少数结果正确发现平滑模式、Scope 加总、碳信用覆盖不足和可再生比例跳变，应保留有效分。

### 5.7 证据链与置信度

- 结构化结果多数可复算，因此获得主要证据分。
- 卫星结果的计算错误与较高置信度不匹配，置信度校准为 0/3。
- 文本结果缺稳定公开 ID、证据文件、字段和公式，证据链仅 4/15。
- “需要更多证据”可作为 follow-up，但不能与“已检测到造假”使用同一置信度口径。

## 6. 内部文件索引

- 最终基线说明：`01_GT_SENT/README_AUTHORITATIVE_FINAL_BASELINE_DO_NOT_SHARE.md`
- 201 条答案总账：`02_CHANGES_INTERNAL/AUTHORITATIVE_ALL_CHANGE_MASTER_201_ROWS_DO_NOT_SHARE.csv`
- Group2 自报记录核验：`02_CHANGES_INTERNAL/group2_submitted_change_log_VERIFIED_AGAINST_ACTUAL_SENT_PACKAGE_DO_NOT_SHARE.csv`
- Group2 27 项明细：`03_SCORE_INTERNAL/GROUP2_STRICT_DETAILED_SCORE_ITEMS_DO_NOT_SHARE.csv`
- Group3 21 项明细：`03_SCORE_INTERNAL/GROUP3_STRICT_DETAILED_SCORE_ITEMS_DO_NOT_SHARE.csv`
- Group3 结构化逐目标：`03_SCORE_INTERNAL/group3_structured_detection_target_by_target_DO_NOT_SHARE.csv`
- Group3 卫星逐目标：`03_SCORE_INTERNAL/group3_satellite_detection_target_by_target_DO_NOT_SHARE.csv`
- Group3 文本逐目标：`03_SCORE_INTERNAL/group3_text_detection_target_by_target_DO_NOT_SHARE.csv`
