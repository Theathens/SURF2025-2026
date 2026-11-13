# step5_detect_anomalies.py
"""
Step 5: 基于合并数据，应用规则检测绿洗嫌疑。
输出：提交模板格式的CSV。
"""

import pandas as pd
import numpy as np
import re
from config import MERGED_DATA_OUTPUT, SUBMISSION_OUTPUT

# ================= 硬编码配置区 =================
# 各检测规则的置信度基础值（后续会根据偏差动态调整）
CONFIDENCE_CONFIG = {
    "physical_expansion_energy_mismatch": 0.70,
    "pv_growth_renewable_mismatch": 0.60,
    "nightlight_production_mismatch": 0.55,
    "claim_pv_inflation": 0.65,  # 基础值，实际会根据偏差调整
}
DEFAULT_CONFIDENCE = 0.50

# 规则触发阈值
THRESHOLDS = {
    "built_expansion_threshold_pct": 20,  # 建成区增长 > 20%
    "energy_growth_threshold_pct": 10,  # 能耗增长 < 10% 或负增长
    "pv_growth_threshold_pct": 100,  # PV面积增长 > 100%
    "renewable_delta_threshold_pct": 10,  # 可再生比例增长 < 10个百分点
    "production_growth_threshold_pct": 30,  # 产量增长 > 30%
    "claim_pv_deviation_threshold_pct": 20,  # 文本PV与遥感PV偏差 > 20%
}


# ===============================================


def parse_chinese_area(text: str) -> tuple:
    """
    从中文文本中提取面积数值和单位，统一转换为平方米 (m²)。
    返回 (面积值_m2, 原始字符串) 或 (None, None)
    支持格式：
      - 12万平方米
      - 5.2万平方米
      - 18000平方米
      - 3.6 km² (较少见)
    """
    # 匹配模式：数字（含小数） + 可选空格 + 单位
    # 优先匹配“万”单位
    pattern_wan = re.compile(r'(\d+\.?\d*)\s*万\s*平方米')
    match = pattern_wan.search(text)
    if match:
        value = float(match.group(1))
        return value * 10000, f"{value}万平方米"

    # 匹配“平方米”直接
    pattern_m2 = re.compile(r'(\d+\.?\d*)\s*平方米')
    match = pattern_m2.search(text)
    if match:
        value = float(match.group(1))
        return value, f"{value}平方米"

    # 匹配“km²”或“平方公里” (较少)
    pattern_km2 = re.compile(r'(\d+\.?\d*)\s*km²')
    match = pattern_km2.search(text)
    if match:
        value = float(match.group(1))
        return value * 1_000_000, f"{value}km²"

    return None, None


def detect_anomalies(merged_df):
    submissions = []

    # 检查必要列是否存在
    required_cols = [
        'firm_id', 'facility_id', 'facility_name_x',
        'start_year_facility', 'end_year_facility',
        'built_pct_change', 'pv_pct_change', 'night_light_delta',
        'energy_pct_change', 'elec_pct_change',
        'scope1_pct_change', 'scope2_pct_change',
        'renewable_ratio_delta', 'prod_pct_change',
        'public_claim_id', 'public_esg_claim_text', 'end_pv_m2'
    ]
    for col in required_cols:
        if col not in merged_df.columns:
            print(f"警告: 缺少列 {col}，将创建空列")
            merged_df[col] = np.nan if col not in ['public_claim_id', 'public_esg_claim_text'] else ''

    for idx, row in merged_df.iterrows():
        # 提取数值
        built_pct = row.get('built_pct_change', np.nan)
        energy_pct = row.get('energy_pct_change', np.nan)
        elec_pct = row.get('elec_pct_change', np.nan)
        scope1_pct = row.get('scope1_pct_change', np.nan)
        scope2_pct = row.get('scope2_pct_change', np.nan)
        pv_pct = row.get('pv_pct_change', np.nan)
        renewable_delta = row.get('renewable_ratio_delta', np.nan)
        night_delta = row.get('night_light_delta', np.nan)
        prod_pct = row.get('prod_pct_change', np.nan)
        end_pv = row.get('end_pv_m2', np.nan)
        claim_text = row.get('public_esg_claim_text', '')
        start_year = row.get('start_year_facility', np.nan)
        end_year = row.get('end_year_facility', np.nan)

        if pd.isna(start_year) or pd.isna(end_year):
            continue

        review_period = f"{int(start_year)}-{int(end_year)}"
        facility_id = row.get('facility_id', 'unknown')
        firm_id = row.get('firm_id', 'unknown')
        company_name = row.get('company_name', '') or row.get('company_name_y', '') or row.get('company_name_x', '')
        public_claim_id = row.get('public_claim_id', 'N/A')

        # ---------- 规则1：物理扩张 vs 能耗 ----------
        if not pd.isna(built_pct) and built_pct > THRESHOLDS["built_expansion_threshold_pct"]:
            if not pd.isna(energy_pct) and (energy_pct < THRESHOLDS["energy_growth_threshold_pct"] or energy_pct < 0):
                submissions.append({
                    'submission_id': f"ANOM_{facility_id}_{int(start_year)}_{int(end_year)}_ENE",
                    'public_claim_id': public_claim_id,
                    'firm_id': firm_id,
                    'company_name': company_name,
                    'facility_id': facility_id,
                    'review_period': review_period,
                    'detection_module': 'remote_sensing_cross_check',
                    'suspected_statement_or_metric': 'Energy consumption growth vs physical expansion',
                    'evidence_file': 'round2_public_blind_virtual_aoi_remote_sensing_features_2018_2025.csv; round2_public_blind_annual_esg_metrics_2018_2025.csv',
                    'evidence_field_or_section': 'built_up_area_ha, total_energy_mwh (company level)',
                    'detection_reasoning': f"设施建成区增长{built_pct:.1f}%，但公司总能耗仅增长{energy_pct:.1f}%，可能存在能耗转移或虚报效率。",
                    'suspected_issue_type': 'physical_expansion_energy_mismatch',
                    'confidence_0_to_1': CONFIDENCE_CONFIG.get("physical_expansion_energy_mismatch",
                                                               DEFAULT_CONFIDENCE),
                    'recommended_follow_up': '核查该设施实际用电量与公司分配方法。'
                })

        # ---------- 规则2：PV面积 vs 可再生能源比例 ----------
        if not pd.isna(pv_pct) and pv_pct > THRESHOLDS["pv_growth_threshold_pct"]:
            if not pd.isna(renewable_delta) and renewable_delta < THRESHOLDS["renewable_delta_threshold_pct"]:
                submissions.append({
                    'submission_id': f"ANOM_{facility_id}_{int(start_year)}_{int(end_year)}_PV",
                    'public_claim_id': public_claim_id,
                    'firm_id': firm_id,
                    'company_name': company_name,
                    'facility_id': facility_id,
                    'review_period': review_period,
                    'detection_module': 'remote_sensing_cross_check',
                    'suspected_statement_or_metric': 'PV area increase vs renewable ratio',
                    'evidence_file': 'round2_public_blind_virtual_aoi_remote_sensing_features_2018_2025.csv; round2_public_blind_annual_esg_metrics_2018_2025.csv',
                    'evidence_field_or_section': 'roof_pv_area_m2, renewable_electricity_ratio_pct',
                    'detection_reasoning': f"设施光伏面积增长{pv_pct:.1f}%，但公司可再生能源比例仅增加{renewable_delta:.1f}个百分点，可能光伏利用率低或绿电证书虚增。",
                    'suspected_issue_type': 'pv_growth_renewable_mismatch',
                    'confidence_0_to_1': CONFIDENCE_CONFIG.get("pv_growth_renewable_mismatch", DEFAULT_CONFIDENCE),
                    'recommended_follow_up': '检查光伏实际发电量与并网记录。'
                })

        # ---------- 规则3：夜光与产量背离 ----------
        if not pd.isna(night_delta) and not pd.isna(prod_pct):
            if night_delta < 0 and prod_pct > THRESHOLDS["production_growth_threshold_pct"]:
                submissions.append({
                    'submission_id': f"ANOM_{facility_id}_{int(start_year)}_{int(end_year)}_NIGHT",
                    'public_claim_id': public_claim_id,
                    'firm_id': firm_id,
                    'company_name': company_name,
                    'facility_id': facility_id,
                    'review_period': review_period,
                    'detection_module': 'remote_sensing_cross_check',
                    'suspected_statement_or_metric': 'Night light vs production volume',
                    'evidence_file': 'round2_public_blind_virtual_aoi_remote_sensing_features_2018_2025.csv; round2_public_blind_annual_esg_metrics_2018_2025.csv',
                    'evidence_field_or_section': 'annual_night_light_mean, production_volume',
                    'detection_reasoning': f"产量增长{prod_pct:.1f}%，但夜间灯光强度下降，可能实际生产活动未同步增长，或夜光数据被调整。",
                    'suspected_issue_type': 'nightlight_production_mismatch',
                    'confidence_0_to_1': CONFIDENCE_CONFIG.get("nightlight_production_mismatch", DEFAULT_CONFIDENCE),
                    'recommended_follow_up': '对比该工厂用电量与夜光趋势。'
                })

        # ---------- 规则4：AOI声明文本中的PV面积与遥感特征比对 ----------
        if isinstance(claim_text, str) and len(claim_text) > 10:
            declared_pv, original_str = parse_chinese_area(claim_text)
            if declared_pv is not None and not pd.isna(end_pv) and end_pv > 0:
                diff_pct = (end_pv - declared_pv) / (declared_pv + 1e-6) * 100
                if abs(diff_pct) > THRESHOLDS["claim_pv_deviation_threshold_pct"]:
                    # 动态调整置信度：夸大（diff_pct < -20%）高置信，保守（diff_pct > 20%）低置信
                    if diff_pct < -20:
                        confidence = 0.75  # 夸大嫌疑
                    elif diff_pct > 20:
                        confidence = 0.50  # 保守披露
                    else:
                        confidence = CONFIDENCE_CONFIG.get("claim_pv_inflation", DEFAULT_CONFIDENCE)

                    submissions.append({
                        'submission_id': f"ANOM_{facility_id}_{int(start_year)}_{int(end_year)}_CLAIM",
                        'public_claim_id': public_claim_id,
                        'firm_id': firm_id,
                        'company_name': company_name,
                        'facility_id': facility_id,
                        'review_period': review_period,
                        'detection_module': 'text_remote_sensing_compare',
                        'suspected_statement_or_metric': 'Public claim PV area vs remote sensing',
                        'evidence_file': 'round2_public_blind_virtual_aoi_claims.csv; round2_public_blind_virtual_aoi_remote_sensing_features_2018_2025.csv',
                        'evidence_field_or_section': 'public_esg_claim_text; roof_pv_area_m2',
                        'detection_reasoning': f"声明称PV面积{original_str}，遥感显示{end_pv / 10000:.1f}万m²，偏差{diff_pct:.1f}%，{'夸大' if diff_pct < -20 else '保守'}。",
                        'suspected_issue_type': 'claim_pv_inflation',
                        'confidence_0_to_1': round(confidence, 2),
                        'recommended_follow_up': '核查PV安装验收报告。'
                    })

    submission_df = pd.DataFrame(submissions)
    return submission_df


if __name__ == "__main__":
    merged = pd.read_csv(MERGED_DATA_OUTPUT)
    print(f"合并数据行数: {len(merged)}")
    submission = detect_anomalies(merged)
    submission.to_csv(SUBMISSION_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"提交报告已保存至 {SUBMISSION_OUTPUT}")
    print(f"共生成 {len(submission)} 条检测记录。")

    # 打印阈值信息
    print("\n" + "=" * 70)
    print("当前 step5 检测规则配置汇总")
    print("=" * 70)
    print("\n【置信度分配 (confidence_0_to_1)】")
    for issue_type, conf in CONFIDENCE_CONFIG.items():
        print(f"  {issue_type:<40} -> {conf}")
    print(f"  {'默认兜底值 (未匹配规则)':<40} -> {DEFAULT_CONFIDENCE}")
    print("\n【规则触发阈值】")
    for key, val in THRESHOLDS.items():
        print(f"  {key:<40} -> {val}")
    print("\n【当前生效的检测规则说明】")
    print("  规则1: 建成区增长 > 20% 且 公司能耗增长 < 10% 或负增长")
    print("  规则2: PV面积增长 > 100% 且 公司可再生比例增长 < 10个百分点")
    print("  规则3: 夜光强度下降 且 公司产量增长 > 30%")
    print("  规则4: 文本声明PV面积 vs 遥感PV面积 偏差 > 20%")
    print("=" * 70)