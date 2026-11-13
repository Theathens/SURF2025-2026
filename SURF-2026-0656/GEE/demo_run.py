# -*- coding: utf-8 -*-
"""
demo_run.py
============
演示如何使用 greenwash_scorer.py 跑通完整的五层管道：
  1) 构造/加载 Claim（声称）
  2) 构造 BaselineStats 并 fit（同类AOI的历史变化量分布）
  3) 组装每条声称对应的 ProxyObservation（T0/T1两期观测）
  4) compute_credibility() 算出加权可信度
  5) calibrate_thresholds() 用对照组校准阈值，给出嫌疑度分级
  6) 导出结果表格 + 打印可解释输出

底部额外给了一段"如何从你们真实的 Zonal_stats2.py 产出的两期 Excel
(industrial_activity_index.xlsx) 里自动构造 ProxyObservation"的示例函数。
"""

import numpy as np
import pandas as pd
from greenwash_scorer import (
    Claim, ClaimCategory, ClaimDirection, ProxyObservation,
    BaselineStats, compute_credibility, check_contradiction,
    calibrate_thresholds, explain, batch_score_to_dataframe,
)

np.random.seed(42)

# ------------------------------------------------------------
# 步骤2：拟合基线分布
# 实践中：把同一批工业AOI(比如你们892个patch里含工业区的774个)
# 在历史两期之间的 Δnightlight / ΔLST / Δbuiltup_area 都算出来，
# 喂给 BaselineStats.fit()。这里用模拟数据代替。
# ------------------------------------------------------------
baseline = BaselineStats()
baseline.fit("nightlight", list(np.random.normal(0, 0.6, 200)))      # 模拟：774个AOI的历史夜光变化量
baseline.fit("lst", list(np.random.normal(0, 0.8, 200)))              # 模拟：地表温度变化量(°C)
baseline.fit("builtup_area", list(np.random.normal(0, 500, 200)))     # 模拟：建成区面积变化量(像素数)
baseline.fit("ndvi", list(np.random.normal(0, 0.05, 200)))            # 模拟：NDVI变化量

# ------------------------------------------------------------
# 步骤1+3：构造几条示例声称 + 对应的proxy观测
# ------------------------------------------------------------
cases = []

# 案例A：产能声称与proxy矛盾（夜光下降、温度没什么变化，但声称"持续提升"）
claim_a = Claim(claim_id="C001", category=ClaimCategory.OPERATION,
                 direction=ClaimDirection.INCREASE, aoi_id="AOI_017",
                 text="该基地2025年生产活动持续提升，夜间生产稳定")
obs_a = {
    "nightlight": ProxyObservation("nightlight", value_t0=5.2, value_t1=4.3, quality=1.0),
    "lst": ProxyObservation("lst", value_t0=33.9, value_t1=34.0, quality=1.0),
    "builtup_area": ProxyObservation("builtup_area", value_t0=12000, value_t1=12050, quality=0.8),
}
cases.append((claim_a, obs_a))

# 案例B：扩建声称与proxy高度一致
claim_b = Claim(claim_id="C002", category=ClaimCategory.EXPANSION,
                 direction=ClaimDirection.INCREASE, aoi_id="AOI_033",
                 text="新增厂房及配套物流仓储区域")
obs_b = {
    "nightlight": ProxyObservation("nightlight", value_t0=3.1, value_t1=3.6, quality=1.0),
    "lst": ProxyObservation("lst", value_t0=32.0, value_t1=32.9, quality=1.0),
    "builtup_area": ProxyObservation("builtup_area", value_t0=8000, value_t1=9400, quality=1.0),
}
cases.append((claim_b, obs_b))

# 案例C：光伏/绿化声称，proxy证据不足（弱验证，且这次云覆盖导致NDVI质量低）
claim_c = Claim(claim_id="C003", category=ClaimCategory.GREEN,
                 direction=ClaimDirection.INCREASE, aoi_id="AOI_055",
                 text="厂区屋顶光伏比例提高，绿化覆盖提升")
obs_c = {
    "ndvi": ProxyObservation("ndvi", value_t0=0.18, value_t1=0.20, quality=0.3),   # 云覆盖导致质量打折
    "builtup_area": ProxyObservation("builtup_area", value_t0=8000, value_t1=8000, quality=1.0),
}
cases.append((claim_c, obs_c))

# ------------------------------------------------------------
# 步骤4：批量打分
# ------------------------------------------------------------
results = []
for claim, obs in cases:
    r = compute_credibility(claim, obs, baseline)
    results.append(r)

# ------------------------------------------------------------
# 步骤5：校准阈值
# 实践中：baseline_suspicion_scores 应该来自一批"已确认真实"的对照组声称，
# 这里暂时就用本次三条案例的嫌疑度自己模拟对照（正式使用前请替换成真实对照组）
# ------------------------------------------------------------
bands = calibrate_thresholds([r.suspicion for r in results])

print("=" * 70)
print("单条声称打分详情")
print("=" * 70)
for r in results:
    print(explain(r, bands))
    print("-" * 70)

# ------------------------------------------------------------
# 矛盾类示例：产能提升 vs 排放下降
# ------------------------------------------------------------
print("=" * 70)
print("矛盾类检查示例")
print("=" * 70)
contradiction = check_contradiction(
    claim_a_topic="产能", claim_a_dir=ClaimDirection.INCREASE,
    claim_b_topic="排放", claim_b_dir=ClaimDirection.DECREASE,
    arbiter_score=results[0],   # 用案例A(产能声称)的proxy可信度做仲裁参考
)
for k, v in contradiction.items():
    print(f"  {k}: {v}")

# ------------------------------------------------------------
# 步骤6：导出表格，方便交给 T5/T6 做可视化或汇总
# ------------------------------------------------------------
df = batch_score_to_dataframe(results, bands)
out_path = "greenwash_scores_demo.xlsx"
df.to_excel(out_path, index=False)
print("\n结果表已保存:", out_path)
print(df[["claim_id", "category", "credibility", "suspicion", "level"]])


# ============================================================
# 【实战对接】如何从你们真实的两期 Zonal_stats2.py 输出自动构造 ProxyObservation
# ============================================================
def build_observations_from_zonal_stats(t0_xlsx: str, t1_xlsx: str,
                                         aoi_patch_name: str) -> dict:
    """
    读取两期 industrial_activity_index.xlsx（Zonal_stats2.py的产出），
    按 patch文件名 (aoi_patch_name) 关联，自动算出 nightlight/lst/ndvi 的Δ，
    并根据是否为 NaN 自动设置 quality。

    用法示例：
        obs = build_observations_from_zonal_stats(
            "industrial_activity_index_2025Q1.xlsx",
            "industrial_activity_index_2025Q3.xlsx",
            "gee_patch_0027_r0384_c0000.png")
        result = compute_credibility(my_claim, obs, baseline)
    """
    df0 = pd.read_excel(t0_xlsx, sheet_name="逐Patch活动指数")
    df1 = pd.read_excel(t1_xlsx, sheet_name="逐Patch活动指数")

    row0 = df0[df0["Patch 文件名"] == aoi_patch_name]
    row1 = df1[df1["Patch 文件名"] == aoi_patch_name]
    if row0.empty or row1.empty:
        raise ValueError(f"在两期表格中找不到 patch: {aoi_patch_name}")
    row0, row1 = row0.iloc[0], row1.iloc[0]

    def make_obs(name, col):
        v0, v1 = row0.get(col), row1.get(col)
        quality = 1.0
        if pd.isna(v0) or pd.isna(v1):
            quality = 0.0
            v0, v1 = v0 if not pd.isna(v0) else 0.0, v1 if not pd.isna(v1) else 0.0
        return ProxyObservation(name, float(v0), float(v1), quality=quality)

    return {
        # 注意：列名要跟你们实际Excel的表头对上，这里用的是之前看到的表头
        "builtup_area": make_obs("builtup_area", "工业像素数"),
        # 如果两期都跑过 activity_features.tif 的光谱指标提取，也可以在这里加 nightlight/lst/ndvi
    }
