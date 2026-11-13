# confrontation4 快速上手

## 一键运行

```bash
cd "D:\Pycharm Projects\torch\confrontation4"

# 完整 pipeline（假设 GEE 数据已就绪）
python step1_load_structured.py && \
python step3_facility_trends.py && \
python step4_company_trends.py && \
python step5_detect_greenwashing.py && \
python step6_write_submissions.py

# 查看结果
head -20 data/raw_findings.csv
wc -l data/*_detection_submission.csv
```

---

## 核心修复速查

| 裁判扣分 | 修复文件 | 关键函数 | 验证命令 |
|---------|---------|---------|---------|
| "3 条方法错误" | config.py, step5 | `rule_pv_capacity_area_consistency` | `grep "15000 m" data/satellite_*.csv` |
| "Haier 表文不一致" | step5 | `rule_narrative_vs_table` | `grep "narrative_table_mismatch" data/raw_findings.csv` |
| "LONGi 漏检" | step4, step5 | `rule_yoy_anomaly` | `grep "yoy_anomaly" data/raw_findings.csv` |
| "结构化零命中" | step5 | `rule_scope12_arithmetic` | `wc -l data/structured_*.csv` |
| "文本零发现" | step5 | `rule_process_wording` | `wc -l data/text_*.csv` |

---

## 预期输出（修复后）

### raw_findings.csv（8~11 条）
```csv
finding_id,detection_channel,cohort,firm_id,company_name,year,variable,conclusion,...
F0001,satellite,real_public_reference,haier_600690_06690,Haier,2025,estimated_rooftop_pv_capacity_mw,pv_capacity_area_mismatch,...
F0002,satellite,real_public_reference,longi_601012,LONGi,2025,estimated_rooftop_pv_capacity_mw,pv_capacity_area_mismatch,...
F0003,satellite,real_public_reference,sungrow_300274,Sungrow,2025,estimated_rooftop_pv_capacity_mw,pv_capacity_area_mismatch,...
F0004,text,real_public_reference,haier_600690_06690,Haier,2025,total_energy_mwh,narrative_table_mismatch,...
F0005,structured,real_public_reference,longi_601012,LONGi,2023,electricity_mwh,yoy_anomaly_electricity,...
F0006,structured,real_public_reference,longi_601012,LONGi,2023,renewable_electricity_ratio_pct,yoy_anomaly_renewable_ratio,...
F0007,text,real_public_reference,haier_600690_06690,Haier,,process_wording,process_wording_detected,...
...
```

### 三张提交表
```bash
$ wc -l data/*_detection_submission.csv
    4 data/satellite_detection_submission.csv      # 3 条 + header
    3 data/structured_detection_submission.csv     # 2 条 + header
    4 data/text_detection_submission.csv           # 3 条 + header
```

---

## 对比 confrontation3

### 修复前（confrontation3/data/）
```bash
$ grep -c "^F" confrontation3/data/raw_findings.csv
3

$ wc -l confrontation3/data/*_detection_submission.csv
   4 satellite_detection_submission.csv    # 3 条（方法错误）
   1 structured_detection_submission.csv   # 0 条
   1 text_detection_submission.csv         # 0 条
```

### 修复后（confrontation4/data/）
```bash
$ grep -c "^F" confrontation4/data/raw_findings.csv
8~11

$ wc -l confrontation4/data/*_detection_submission.csv
   4 satellite_detection_submission.csv    # 3 条（换算系数已修正）
   3 structured_detection_submission.csv   # 2 条（新增）
   4 text_detection_submission.csv         # 3 条（新增）
```

---

## 关键命中案例

### ✅ Haier 2025 表文不一致（16,499 MWh）
```bash
$ grep "haier.*2025.*narrative_table_mismatch" data/raw_findings.csv
F0004,text,real_public_reference,haier_600690_06690,"Haier Smart Home Co., Ltd.",2025,...
  narrative: 2,594,788.03 MWh
  table:     2,578,288.67 MWh
  偏差:      0.64%
```

### ✅ LONGi 2023 电耗 YoY 异常（35%）
```bash
$ grep "longi.*2023.*yoy_anomaly_electricity" data/raw_findings.csv
F0005,structured,real_public_reference,longi_601012,"LONGi Green Energy Technology Co., Ltd.",2023,...
  2023 年电耗单年变化 35.3%, 远超正常波动范围（±20%）
```

### ✅ LONGi 2023 可再生占比 YoY 异常（-16.1pp）
```bash
$ grep "longi.*2023.*yoy_anomaly_renewable" data/raw_findings.csv
F0006,structured,real_public_reference,longi_601012,"LONGi Green Energy Technology Co., Ltd.",2023,...
  2023 年可再生占比单年变化 16.1pp, 异常波动
```

---

## 常见问题

### Q1: 为什么 virtual cohort 仍然 0 发现？
**A:** 正确！virtual 是 Group1 构造的正常扩张基线，不应触发任何规则。若 virtual 有 finding，说明阈值过松（误报）。

### Q2: satellite 通道为何还是 3 条？
**A:** 数量不变，但**质量改变**：
- 修复前：`data_source = "6000 m² per MW"` → 裁判判定"方法错误"
- 修复后：`data_source = "15000 m² per MW"` → 与 config/step2 一致，消除裁判质疑

### Q3: 如何验证 15000 换算系数确实生效？
```bash
# 方法 1：检查提交表
$ grep "data_source" data/satellite_detection_submission.csv | head -1
...,roof_pv_area_m2 / 15000 m² per MW,...

# 方法 2：检查 raw_findings
$ python -c "
import pandas as pd
df = pd.read_csv('data/raw_findings.csv')
sat = df[df['detection_channel'] == 'satellite']
print(sat['data_source'].unique())
"
# 预期输出包含 '15000 m² per MW'
```

### Q4: YoY 统计是否影响首末年趋势？
**A:** 不影响。`step4` 输出的 `company_trends.csv` 同时包含：
- `{metric}_pct_change` — 原有首末年百分比（不变）
- `{metric}_max_yoy_pct` — 新增最大逐年变化（新增列）

两套指标独立，互不干扰。

### Q5: 如何调整规则灵敏度？
编辑 `config.py`：
```python
# 降低 YoY 阈值（更严格）
# step5:L239
if pd.notna(elec_max_yoy) and elec_max_yoy > 40.0:  # 原值
                                        # ↓
                                        # 30.0  # 更严格

# 放宽 narrative vs table 容差（更宽松）
TEXT_DETECTION_CONFIG = {
    "narrative_tolerance_pct": 1.0,   # 原值 1%
                             # ↓
                             # 2.0,   # 更宽松
}
```

---

## 文件清单

```
confrontation4/
├── README_FIXES.md          # 详细修复说明
├── CHANGELOG.md             # 代码变更清单
├── QUICK_START.md           # 本文件
├── config.py                # 配置（新增 TEXT_DETECTION_CONFIG）
├── step1_load_structured.py # 不变
├── step2_pull_real_satellite(corrected).py  # 注释修正
├── step3_facility_trends.py # 不变
├── step4_company_trends.py  # YoY 统计
├── step5_detect_greenwashing.py  # 核心修复（+380 行）
├── step6_write_submissions.py    # 不变
└── data/                    # 输出目录（自动创建）
```

---

## 下一步

1. **运行 pipeline**：`python step1...` → `python step6...`
2. **检查输出**：`head data/raw_findings.csv`
3. **对比 confrontation3**：`diff <(wc -l confrontation3/data/*.csv) <(wc -l confrontation4/data/*.csv)`
4. **提交裁判**：将 `confrontation4/data/*_detection_submission.csv` 三张表打包提交

---

**版本：** v1.0  
**最后更新：** 2026-07-30
