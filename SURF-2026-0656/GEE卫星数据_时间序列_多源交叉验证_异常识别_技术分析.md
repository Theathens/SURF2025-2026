
## 一、GEE 卫星数据获取

### 1.1 用了哪些卫星数据源

项目从 Google Earth Engine 调用了 **6 类遥感数据集**，分层互补：

| 数据源 | 变量 | 分辨率 | 用途 |
|--------|------|--------|------|
| Sentinel-2 SR (`COPERNICUS/S2_SR_HARMONIZED`) | NDVI (B8-B4)/(B8+B4)、屋顶 PV 面积（蓝红比+NDVI+DW 联合阈值）、RGB 缩略图 | **10m** | 植被变化、光伏识别、目视核验 |
| MODIS LST (`MODIS/061/MOD11A2`) | 地表温度 (LST_Day_1km, ×0.02-273.15)、**热岛指数** = LST(AOI) − LST(3km 环形缓冲区) | **1000m** | 工厂余热/热排放 |
| VIIRS DNB (`NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG`) | 夜光强度 avg_rad | **500m** | 夜间生产活动强度 |
| Dynamic World (`GOOGLE/DYNAMICWORLD/V1`) | built 概率、land cover 分类 | **10m** | 建成区面积、建筑掩膜 |
| Landsat 8/9 (`LANDSAT/LC08/C02/T1_L2`) | 作为 Sentinel-2 缺失时的回退 | **30m** | 备选光学源 |

### 1.2 实际获取的关键指标与数值

从 `zonal_stats2_output/report.txt` 的实际运行结果（鲁尔区 892 个 patch）：

```
总 patch 数:        892
含工业区 patch 数:  774 (86.8%)
CAI 均值:           0.2967
CAI 中位数:         0.2773
CAI 95百分位:       0.5983
平均工业覆盖率:     21.57%
```

光谱指标实测值（鲁尔区 AOI）：

| 指标 | 工业区 | 背景 | 说明 |
|------|--------|------|------|
| NDVI | 0.0814 | 0.6775 | 工业区植被抑制显著 |
| 夜光 | 4.52 | 4.67 | 比值 0.968，接近但略低 |
| 温度 | 34.16°C | 33.92°C | 热岛 Δ=0.24°C |
| 工业像素 | 318,474 px（占 AOI 6.16%） | — | — |

### 1.3 踩过的坑与解决方案

**坑 1：PV 面积识别阈值不稳定**

屋顶光伏板的光谱特征在 Sentinel-2 上表现为"蓝红比 > 1.05 + NDVI < 0.15 + Dynamic World built > 0.4 + B2 在 0.05~0.35 之间"。但这个启发式阈值**在不同地区/季节偏移很大**，导致 confrontation2 中 PV 面积的遥感估算值偏差巨大（如 Sungrow 42 MW 披露 vs 遥感 0.06 MW，偏差 67,502%）。最终策略是：**将 PV 检测定位为"再计算值"用于质疑而非断言**，置信度设为 0.70 而非定论。

**坑 2：MODIS 1km 分辨率太粗**

MODIS LST 是 1km 像素，对于单个工厂 AOI（通常 500m×500m 量级）而言，一个像素就覆盖了整个厂区甚至周围区域，热岛指数的计算噪声极大。解决方案是引入**3km 环形缓冲区**做差值（`aoi.buffer(3000).difference(aoi)`），用空间差分来放大信号。但即便如此，热岛信号仍然很弱（Δ=0.24°C），所以 `heat_scope1_mismatch` 的置信度只给了 0.50。

**坑 3：GEE 请求频率限制**

每个 (facility, year) 的统计要调 5-6 个 `reduceRegion`，加上图像导出，每处理一个点需要约 2 秒间隔。代码中 `request_interval: 2.0` 是硬编码的经验值。confrontation3 的 step2 有"断点续传"机制——如果 PNG 已存在就跳过，否则每次中断都要从头来。

**坑 4：云覆盖过滤的两难**

Sentinel-2 用 `CLOUDY_PIXEL_PERCENTAGE < 30` 过滤。阈值太低导致很多 AOI（尤其中国南方）无可用影像；阈值太高则中值合成被云污染。最终在 step2 中改用**夏季（6-9月）合成**来规避冬季低太阳角问题，并增加了 Landsat 回退机制。

---

## 二、时间序列分析

### 2.1 分析框架

时间序列分两层：

1. **设施级趋势**（step3）：对每个 facility 的遥感代理变量（夜光、热岛、NDVI、建成区、PV 面积）做首末年对比 + YoY 逐年变化
2. **公司级趋势**（step4）：对年度 ESG 指标（电耗、排放、可再生比例等 18 个字段）做首末年变化 + **YoY 逐年变化检测**

### 2.2 关键时间序列指标与阈值

| 检测维度 | 首末年阈值 | YoY 阈值 | 置信度 |
|---------|-----------|---------|--------|
| 建成区扩张 vs 能耗 | 建成区 +10% 且 能耗 +5% | — | 0.70 |
| PV 面积增长 vs 可再生占比 | PV +30% 且 可再生 < +5pp | — | 0.65 |
| 夜光 vs 产量 | 夜光 Δ≤0 且 产量 +20% | — | 0.55 |
| 电耗 YoY | — | 单年 ±30% | 0.70 |
| 可再生占比 YoY | — | 单年 ±10pp | 0.75 |
| Scope1/2 排放 YoY | — | 单年 +40%/-30% | 0.65 |
| 能耗强度骤降 | 首末年 -15% | — | 0.55 |
| **排放-电量脱钩** | — | 电量与排放方向相反且均 >10% | 0.70 |

### 2.3 实际发现的时间序列异常

从 `raw_findings.csv` 的实际输出（confrontation4 修复后）：

**F0001 — LONGi 电耗 YoY 异常**：

```
2022→2023: electricity_mwh 由 9,068,091 变为 12,267,985
单年变化 +35.29%（阈值区间 [-20%, +30%]）
结论: yoy_anomaly_electricity — 疑似中间年数据注水
```

**F0002/F0003 — LONGi 可再生占比 YoY 剧烈波动**：

```
2022→2023: renewable_ratio 由 47.2% 降至 31.1%（-16.10pp）
2023→2024: renewable_ratio 由 31.1% 升至 47.5%（+16.40pp）
结论: 一年暴跌一年暴涨，不符合能源转型的渐进性特征
```

**F0004 — LONGi 排放-电量脱钩（核心突破）**：

```
2022→2023: 电量 +35.3% 而 Scope1 -14.1%
方向相反，排放轨迹与活动水平脱钩（疑似平滑）
```

### 2.4 踩过的坑与思考

**坑 1：首末年对比会"跳过"中间年造假**

这是 confrontation3 被裁判扣分的核心问题。LONGi 在 2022→2024 的首末年对比中看起来"正常"（电耗从 9,068,091 到 12,267,985 是逐年增长），但 **2023 年的 YoY 异常被完全跳过了**。

> **感悟**：时间序列检测不能只看端点。首末年变化率是"总量视角"，YoY 变化率是"增量视角"。造假者可能在某一年大量注水，但把首末年拉平后看起来趋势合理。confrontation4 新增的 `company_yoy_details.csv` 长表（每个 firm × metric × year 一行）就是为了捕捉这类"中间年脉冲"。

**坑 2：可再生占比用"百分点差"而非"百分率变化"**

```python
# renewable_ratio 用绝对差（pp），其他用百分比变化（%）
if short == "renewable_ratio":
    signed_pct = curr_val - prev_val  # 有符号百分点变化
```

> **思考**：可再生占比从 5% 升到 10% 是 +100% 的增长率（吓人），但从 5pp 到 10pp 是 +5pp（温和）。用百分率会产生误导性高 YoY，用百分点才反映物理实质。这个区分在处理 ESG 数据时极其重要。

**坑 3：排放-电量脱钩检测的物理先验**

```python
# 物理前提：直接排放（Scope 1）随活动水平同向变动
# 若电量大幅变化而排放反向大幅变化，排放轨迹大概率被人为平滑
if (emis_pct > 0) == (elec_pct > 0):
    continue  # 同向变动，物理正常
```

> **感悟**：这是一个很朴素但极其有效的物理约束——**能量守恒和质量守恒意味着排放不会在产量大增时反向下降**（除非有独立的能效改善证据）。LONGi 2022→2023 电量 +35.3% 而 Scope1 -14.1% 是 6 家公司中**唯一反向**的，这个规则直接把它锁定。

---

## 三、多源数据 Cross-Validation

### 3.1 三通道交叉验证架构

项目最终形成了 **卫星通道 / 结构化通道 / 文本通道** 的三通道架构：

```
        卫星通道                    结构化通道                 文本通道
  ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ Sentinel-2 NDVI  │      │ Scope1+2 算术    │      │ 叙述 vs 表格交叉  │
  │ MODIS LST 热岛   │      │ YoY 年度异常      │      │ 过程措辞扫描      │
  │ VIIRS 夜光       │      │ 排放-电量脱钩     │      │ PV 声明 vs 遥感  │
  │ DynamicWorld 建成区│     │ 能耗强度异常      │      │                  │
  │ PV 面积估算      │      │                  │      │                  │
  └────────┬────────┘      └────────┬─────────┘      └────────┬─────────┘
           │                        │                         │
           ▼                        ▼                         ▼
       raw_findings.csv（统一发现表，每行含 detection_channel 字段）
```

### 3.2 具体的交叉验证规则

**规则 A：叙述 vs 表格交叉**（`rule_structured_claim_vs_table`）

```python
# 声称值 vs 结构化表中的对应值，偏差 > 1% 即触发
dev_pct = abs(claim_val - table_val) / table_val * 100.0
if dev_pct > 1.0:  # 1% 容差
    → narrative_table_mismatch
```

实例：Haier 2025 报告叙述"总能耗 2,594,788 MWh"，结构化表 2,578,288 MWh，差 16,499 MWh。

**规则 B：卫星声称 vs 遥感特征**（`rule_satellite_claim_vs_proxy`）

```python
# 披露的 PV 装机/面积 vs 遥感估算值，偏差 > 20% 即触发
dev_pct = abs(claim_val - proxy_val) / proxy_val * 100.0
if dev_pct > 20.0:  # 20% 容差
    → satellite_claim_mismatch
```

实例：Sungrow 披露 42 MW 屋顶光伏，GEE 遥感特征仅 0.06 MW（偏差 67,502%）。

**规则 C：定性声称 vs 遥感趋势**（`rule_qualitative_claim_vs_trend`）

```python
# 热岛指数上升 > 0.5°C 但声称"控制/低" → 矛盾
# 夜光上升但声称"低负荷/错峰" → 矛盾
```

**规则 D：排放-电量脱钩**（`rule_emission_electricity_decoupling`）

```python
# 电量变化 > 10% 且 排放变化 > 10% 且 方向相反 → 轨迹平滑
```

**规则 E：PV 面积 vs 可再生比例**（`rule_pv_growth_vs_renewable`）

```python
# PV 面积 +30% 但可再生占比 < +5pp → 物理不匹配
```

### 3.3 三通道各自的置信度基线

```python
CONFIDENCE_BASE = {
    "scope12_arithmetic_error":       0.85,  # 算术校验最硬，置信度最高
    "narrative_table_mismatch":       0.75,  # 叙述 vs 表格，几乎不可能是巧合
    "physical_expansion_energy_mismatch": 0.70,  # 物理扩张 vs 能耗
    "claim_pv_inflation":             0.70,  # PV 面积虚报
    "emission_electricity_decoupling": 0.70,  # 排放-电量脱钩
    "pv_growth_renewable_mismatch":   0.65,  # PV vs 可再生
    "pv_capacity_area_mismatch":      0.60,  # 装机 vs 面积
    "process_wording_detected":       0.60,  # 过程措辞
    "nightlight_production_mismatch": 0.55,  # 夜光 vs 产量
    "energy_intensity_suspicious_drop": 0.55,  # 能耗强度骤降
    "heat_scope1_mismatch":           0.50,  # 热岛 vs Scope1（信号最弱）
}
```

> **感悟**：置信度的设计反映了"**数据硬度**"——算术校验（Scope1+Scope2=合计）是铁证（0.85），因为数学不可能说谎；文本与表格交叉是强证据（0.75），因为两处数据独立产生；而热岛指数这类遥感代理变量受大气/季节/分辨率影响大，只能给 0.50。**规则的置信度应该反映物理证据链的坚固程度，而不是检测算法的灵敏度。**

### 3.4 greenwash_scorer 的五层管道

`GEE/greenwash_scorer.py` 实现了一个更理论化的评分框架：

```
第一层: 声称结构化 (Claim) → 4类: OPERATION / EXPANSION / GREEN / CONTRADICTION
第二层: z-score 归一化 → BaselineStats 用同类 AOI 的 Δ 分布做归一化（不用固定阈值）
第三层: 单维度打分 s_p = clip(sign(方向) × z / k, -1, 1)
第四层: 加权融合 → 专家权重表按声称类别分配 proxy 权重
第五层: 阈值分级 → 用"已知正常"对照组的嫌疑度分布校准 mean±1σ/2σ
```

专家权重表的物理先验：

| 声称类别 | nightlight | lst | builtup_area | ndvi |
|---------|-----------|-----|-------------|------|
| OPERATION（产能运营） | **0.45** | 0.35 | 0.20 | — |
| EXPANSION（扩建新增） | 0.25 | 0.15 | **0.60** | — |
| GREEN（光伏绿化） | — | — | 0.30 | 0.30 |

> **思考**："产能运营"声称用夜光权重最高（0.45），因为夜间灯光是生产活动最直接的代理；"扩建"声称用建成区权重最高（0.60），因为扩建的物理表现就是建筑面积增加；"光伏绿化"只用 NDVI 和建成区，且**置信度上限被截断到 0.5**（`CATEGORY_CREDIBILITY_CAP`），因为 10m 分辨率下光伏板的光谱证据本身很弱——**宁可"证据不足"也不能"错误指控"**。

---

## 四、异常识别

### 4.1 命中的异常实例

| 发现 ID | 公司 | 年份 | 通道 | 异常类型 | 具体数值 | 置信度 |
|---------|------|------|------|---------|---------|--------|
| F0001 | LONGi | 2023 | structured | 电耗 YoY +35.3% | 9,068,091 → 12,267,985 MWh | 0.70 |
| F0002 | LONGi | 2023 | structured | 可再生占比 -16.1pp | 47.2% → 31.1% | 0.75 |
| F0003 | LONGi | 2024 | structured | 可再生占比 +16.4pp | 31.1% → 47.5% | 0.75 |
| F0004 | LONGi | 2023 | structured | 排放-电量脱钩 | 电量 +35.3% vs Scope1 -14.1% | 0.70 |
| F0005 | Sungrow | 2025 | satellite | 声明 vs 遥感 | 披露 42 MW vs 遥感 0.06 MW | 0.70 |

### 4.2 红队攻击的异常评分体系

`GEE/对抗/对抗.py` 实现了三种绿洗攻击的检测：

| 攻击类型 | 检测逻辑 | 异常评分 |
|---------|---------|---------|
| **GREEN_PEAK_COUPLING**（绿电-峰值耦合） | 卫星高工业活动 × 高可再生占比 → 矛盾 | `anomaly = activity × renewable_ratio × 5` |
| **VOLATILITY_SMOOTHING**（波动平滑） | 用电量与卫星活动指数不一致 | `cons_anomaly = |实际消耗 - 期望消耗| / 期望消耗` |
| **DAILY_TEMPORAL_SHIFT**（时间平移） | 卫星拍摄日期的季节 vs 热特征不符 | `thermal_anomaly = |实际温度 - 季节期望温度| / 15` |

综合评分：

```python
composite = rr_anomaly * 0.4 + cons_anomaly * 0.35 + thermal_anomaly * 0.25
verdict = "HIGH_RISK" if composite >= 0.55 else "MEDIUM_RISK" if >= 0.35 else "LOW_RISK"
```

### 4.3 greenwash_scorer 的矛盾类检测

```python
LOGICAL_CONSISTENCY_RULES = {
    ("产能", "排放"): True,   # 产能提升应伴随排放上升（除非有独立能效证据）
    ("产能", "能效"): False,  # 产能提升不必然伴随能效改善
}
```

> **感悟**：这个"矛盾类"检测的核心思想是——**两条声称之间的逻辑自洽性**。如果一家公司同时声称"产能大幅提升"和"排放大幅下降"，且没有给出独立的能效改善证据，那这两条声称在物理上是矛盾的。proxy 作为仲裁票，判断哪条更可能是真的。

### 4.4 U-Net 工业分割的异常检测

`Zonal stats2.py` 在 892 个 patch 上跑 U-Net 推理，计算四类活动指标：

| 指标 | 公式 | 物理含义 |
|------|------|---------|
| **VDI（植被抑制指数）** | 1 − NDVI_industrial / NDVI_background | 植被越少 → 工业越活跃 |
| **BAI（亮度活跃指数）** | mean反射率_industrial / mean反射率_background | 工业设施反射率高于背景 |
| **TCI（纹理复杂指数）** | texture_entropy_industrial / texture_entropy_background | 工业区纹理比自然区复杂 |
| **CAI（综合活动指数）** | VDI×0.4 + BAI×0.3 + TCI×0.3 | 0=完全非活跃，1=高强度运行 |

实际分布：CAI 均值 0.2967，95 百分位 0.5983 → 大部分 patch 处于低活跃状态，少数高活跃 patch 是重点审查对象。

---

## 五、贯穿四个维度的核心感悟

### 感悟 1：可解释性 > 精度

> *"这是一个「可解释规则模型」，不是端到端机器学习分类器——因为我们没有'绿洗/未绿洗'的标签数据，规则模型的每一步都可以拆开检查、和领域专家权重对齐，也方便向评委解释每个判断是怎么来的。"*

这段代码注释精确地解释了为什么整个项目没有用深度学习做最终判别。没有 ground truth 标签，黑箱模型无法被信任。每条 finding 都需要能回答"为什么判定为异常"，规则模型的每个参数都能追溯到物理先验。

### 感悟 2：多源数据的"互相证伪"比"互相证实"更可靠

项目的交叉验证不是"多个证据指向同一个结论"，而是**"一个数据源的声称在另一个数据源中被证伪"**：

- 报告说"产能提升" → 夜光没涨（卫星证伪）
- 报告说"PV 42MW" → 遥感只有 0.06MW（卫星证伪）
- 报告说"能耗 2,594,788 MWh" → 表格写 2,578,288 MWh（文本-表格互证）
- 报告说 Scope1+2=8,061,970 → Scope1(550,932)+Scope2(7,511,038)=8,061,970（算术通过）

> **证伪比证实更有力**——因为证实只需要一个吻合，证伪只需要一个矛盾。

### 感悟 3：阈值的设定需要"数据校准"而非"拍脑袋"

`greenwash_scorer.py` 中的 `calibrate_thresholds` 函数体现了一个关键思想：

```python
# 用一批"已知为正常/已确认真实"的对照组声称的嫌疑度分布，
# 校准出 mean/std，再用 mean+1σ / mean+2σ 分级。
```

阈值不应该来自"感觉"，而应该来自**同类样本的统计分布**。用 z-score 而非固定阈值，正是因为不同工厂类型/规模的基线波动差异很大——一个大型钢铁厂的夜光变化量天然比一个小型电子厂高一个量级。

### 感悟 4：造假者会进化，检测器也必须迭代

从 confrontation → confrontation2 → confrontation3 → confrontation4 的演进过程中，每一轮裁判反馈都揭示了新的攻击面：

- **Round1**：造假者用假数据 → 检测器发现遥感不匹配
- **Round2**：造假者开始构造"合理"的时间序列 → 检测器引入 YoY 检测
- **Round3**：引入真实公司 → 发现"排放轨迹平滑"这种更隐蔽的造假
- **Round4**：裁判指出"文本零发现" → 检测器新增叙述-表格交叉、过程措辞扫描

> **安全领域的攻防博弈在 ESG 检测中完全适用**：检测器的每一次升级都会驱动造假技术的进化，反之亦然。

---

## 附录：代码架构概览

| 分组 | 核心职责 | 关键文件 |
|------|---------|---------|
| `greenwash_detection_orchestra/` | 模块化检测引擎原型 | `base_models.py`, `orchestrator.py`, `main.py` |
| `confrontation/` | T2 Round1 对抗 pipeline | step1-step5, data/ |
| `confrontation2/` | T2 Round2 对抗（真实公司） | step1-step5, 01_send_to_group3/ |
| `confrontation_virtual/` | T2 Round2 虚拟公司对抗 | step1-step5, 真实数据/ |
| `confrontation3/` | T5 Round3 对抗（Haier/LONGi/Sungrow） | config, run_all, step1-step6, 26.7.26/ |
| `confrontation4/` | T5 Round4 修复版（6项扣分修复） | step1-step6, CHANGELOG, README_FIXES |
| `GEE/` | 卫星遥感数据处理 + U-Net 训练 | greenwash_scorer.py, train_unet, Zonal stats2, 对抗/ |
| `graphify-out/` | 代码依赖分析可视化 | graph.html, manifest.json |
