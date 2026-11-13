#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_assemble_evidence.py (最终版)
====================================
功能：
  1. 读取遥感最终结果（final_results.csv）
  2. 读取企业披露数据（claims, metric_panel）
  3. 针对物理扩张显著的厂区，找出对应年份的声明，进行交叉验证
  4. 输出符合提交模板的 detection 报告

改进：
  - 年份匹配容差 ±2 年
  - 关键词匹配更灵活
  - 若无声明，则使用 metric_panel 数据作为替代证据
"""

import pandas as pd
import os
import re
from typing import Dict, List, Optional

# ================= 硬编码配置 =================
FINAL_RESULTS_PATH = r"/\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\results\final_results.csv"
CLAIMS_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_REAL_COMPANIES_20260718\01_send_to_group3_PUBLIC_BLIND\round2_public_blind_real_company_claims.csv"
METRIC_PANEL_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_REAL_COMPANIES_20260718\01_send_to_group3_PUBLIC_BLIND\round2_public_blind_real_company_metric_panel.csv"
GREEN_POWER_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_REAL_COMPANIES_20260718\01_send_to_group3_PUBLIC_BLIND\round2_public_blind_real_company_green_power_carbon_credit.csv"
OUTPUT_CSV = r"D:\Pycharm Projects\torch\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\submission_ready.csv"

# 厂区名称到公司名称的映射（用于匹配 claims 中的 company_name）
FACILITY_TO_COMPANY = {
    "BYD_Company_Limited_BYD_Pingshan_headquarters_manufacturing_campus": "BYD Company Limited",
    "BYD_Company_Limited_BYD_Xiaomo_Industrial_Park": "BYD Company Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Liyang_Production_Base": "Contemporary Amperex Technology Co., Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Ningde_headquarters_base": "Contemporary Amperex Technology Co., Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Yibin_Production_Base": "Contemporary Amperex Technology Co., Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Zhaoqing_Production_Base": "Contemporary Amperex Technology Co., Limited",
    "Midea_Group_Co._Ltd._Hefei_Laundry_Appliance_Factory": "Midea Group Co., Ltd.",
    "Midea_Group_Co._Ltd._Midea_Chongqing_Factory_Parks": "Midea Group Co., Ltd.",
    "Midea_Group_Co._Ltd._Midea_Shunde_headquarters_factory_area": "Midea Group Co., Ltd.",
}



# =============================================

def load_disclosure_data():
    claims = pd.read_csv(CLAIMS_PATH)
    metrics = pd.read_csv(METRIC_PANEL_PATH)
    green = pd.read_csv(GREEN_POWER_PATH) if os.path.exists(GREEN_POWER_PATH) else None
    return claims, metrics, green


def find_relevant_claims(claims_df: pd.DataFrame, company: str, facility_name: str, year: int) -> List[dict]:
    """
    查找与某厂区某年份相关的企业声明
    年份容差 ±2 年，关键词匹配不区分大小写
    """
    company_claims = claims_df[claims_df["company_name"] == company]
    relevant = []
    for _, row in company_claims.iterrows():
        text = str(row.get("public_disclosure_text", ""))
        claim_year_raw = row["year_or_period"]
        # 尝试提取年份
        try:
            claim_year_int = int(claim_year_raw)
        except:
            nums = re.findall(r'\d{4}', str(claim_year_raw))
            if nums:
                claim_year_int = int(nums[0])
            else:
                continue
        # 年份容差 ±2
        if abs(claim_year_int - year) <= 2:
            # 关键词匹配（不区分大小写）
            keywords = ["pingshan", "xiaomo", "ningde", "yibin", "zhaoqing", "liyang", "shunde", "hefei", "chongqing"]
            facility_lower = facility_name.lower()
            text_lower = text.lower()
            matched = False
            for kw in keywords:
                if kw in facility_lower or kw in text_lower:
                    matched = True
                    break
            # 如果关键词匹配或文本中包含厂区名的一部分（如 "Pingshan base"）
            if matched or any(part in text_lower for part in facility_name.lower().split('_')):
                relevant.append({
                    "item": row["disclosure_item"],
                    "value": row["reported_value"],
                    "unit": row["reported_unit"],
                    "text": text,
                    "year": claim_year_int,
                    "case_id": row["public_case_id"]
                })
    # 如果没找到相关声明，则返回该公司与目标年份最近的声明（不限关键词）
    if not relevant:
        # 按年份差值排序
        sorted_claims = []
        for _, row in company_claims.iterrows():
            text = str(row.get("public_disclosure_text", ""))
            claim_year_raw = row["year_or_period"]
            try:
                claim_year_int = int(claim_year_raw)
            except:
                nums = re.findall(r'\d{4}', str(claim_year_raw))
                if nums:
                    claim_year_int = int(nums[0])
                else:
                    continue
            sorted_claims.append({
                "item": row["disclosure_item"],
                "value": row["reported_value"],
                "unit": row["reported_unit"],
                "text": text,
                "year": claim_year_int,
                "case_id": row["public_case_id"],
                "diff": abs(claim_year_int - year)
            })
        if sorted_claims:
            sorted_claims.sort(key=lambda x: x["diff"])
            # 返回最近的一条
            return [{
                "item": sorted_claims[0]["item"],
                "value": sorted_claims[0]["value"],
                "unit": sorted_claims[0]["unit"],
                "text": sorted_claims[0]["text"],
                "year": sorted_claims[0]["year"],
                "case_id": sorted_claims[0]["case_id"]
            }]
    return relevant


def check_physical_consistency(metrics_df: pd.DataFrame, company: str, delta_ratio: float, year_t0: int,
                               year_t1: int) -> Dict:
    """
    根据物理变化量，检查与该厂区相关的披露是否合理
    返回：嫌疑类型、推理依据、置信度
    """
    company_metrics = metrics_df[metrics_df["company_name"] == company]

    def get_first_value(df, metric_name):
        subset = df[df["metric_name"] == metric_name]
        if not subset.empty:
            val = subset.iloc[0]["metric_value"]
            if val not in [None, "NA_public_not_found", "not_disclosed"]:
                try:
                    return float(val)
                except:
                    pass
        return None

    t0_data = company_metrics[company_metrics["year"] == year_t0]
    t1_data = company_metrics[company_metrics["year"] == year_t1]

    elec0 = get_first_value(t0_data, "electricity_consumption")
    elec1 = get_first_value(t1_data, "electricity_consumption")
    ener0 = get_first_value(t0_data, "total_energy_consumption")
    ener1 = get_first_value(t1_data, "total_energy_consumption")
    scope0 = get_first_value(t0_data, "scope1_2_emission")
    scope1 = get_first_value(t1_data, "scope1_2_emission")

    def delta_pct(v0, v1):
        if v0 is not None and v0 > 0 and v1 is not None:
            return (v1 - v0) / v0 * 100
        return None

    elec_delta = delta_pct(elec0, elec1)
    ener_delta = delta_pct(ener0, ener1)
    scope_delta = delta_pct(scope0, scope1)

    physical_growth = delta_ratio
    reasoning_parts = []
    issue_types = []
    confidence = 0.8

    if physical_growth > 0.01:
        if elec_delta is not None and elec_delta < 0:
            reasoning_parts.append(f"物理扩张 {physical_growth * 100:.2f}%，但用电量下降 {elec_delta:.1f}%，违背物理规律")
            issue_types.append("physical_mismatch_electricity")
            confidence -= 0.2
        if elec_delta is not None and 0 < elec_delta < 10 and physical_growth > 0.03:
            reasoning_parts.append(f"物理扩张 {physical_growth * 100:.2f}%，但用电量仅增 {elec_delta:.1f}%，增长不符")
            issue_types.append("underreported_energy")
            confidence -= 0.15
        if scope_delta is not None and scope_delta < 0:
            reasoning_parts.append(f"物理扩张但Scope1+2排放下降 {scope_delta:.1f}%，减排效率远超行业水平")
            issue_types.append("emission_reduction_suspicious")
            confidence -= 0.15
        if physical_growth > 0.05 and (elec_delta is None or ener_delta is None):
            reasoning_parts.append("物理扩张显著，但公司未披露相应年份的能耗数据，数据完整性存疑")
            issue_types.append("missing_disclosure")
            confidence -= 0.1
    else:
        reasoning_parts.append(f"物理变化不显著 ({physical_growth * 100:.2f}%)，未发现明显不一致")
        issue_types.append("no_obvious_issue")
        confidence = 0.6

    confidence = max(0.3, min(0.95, confidence))
    return {
        "reasoning": "; ".join(reasoning_parts),
        "issue_types": issue_types,
        "confidence": round(confidence, 2),
        "elec_delta": elec_delta,
        "ener_delta": ener_delta,
        "scope_delta": scope_delta,
    }


def main():
    final_df = pd.read_csv(FINAL_RESULTS_PATH)
    print(f"📂 加载遥感结果: {len(final_df)} 个厂区")
    claims, metrics, green = load_disclosure_data()
    print(f"📂 加载声明: {len(claims)} 条, 指标: {len(metrics)} 条")

    submissions = []
    for _, row in final_df.iterrows():
        facility_name = row["facility_name"]
        company = FACILITY_TO_COMPANY.get(facility_name, "unknown")
        t0_year = int(row["t0_year"])
        t1_year = int(row["t1_year"])
        delta = row["delta_ratio"]
        remote_conf = row["confidence"]

        # 查找相关声明（使用检测年份）
        relevant_claims = find_relevant_claims(claims, company, facility_name, t1_year)
        if not relevant_claims:
            # 尝试使用基准年份
            relevant_claims = find_relevant_claims(claims, company, facility_name, t0_year)

        # 如果没有声明，从 metric_panel 提取公司级别数据作为替代
        if not relevant_claims:
            suspected_statement = f"无直接匹配声明，引用公司 {company} 的能耗/排放指标"
            evidence_file = "round2_public_blind_real_company_metric_panel.csv"
            evidence_field = f"years {t0_year}-{t1_year}"
            # 尝试获取该公司的能耗数据
            company_metrics = metrics[metrics["company_name"] == company]
            t1_metrics = company_metrics[company_metrics["year"] == t1_year]
            if not t1_metrics.empty:
                metric_sample = t1_metrics.iloc[0]
                suspected_statement += f" (示例: {metric_sample['metric_name']} = {metric_sample['metric_value']} {metric_sample['metric_unit']})"
            detection_reasoning = f"遥感检测：厂区 {facility_name} 在 {t0_year}→{t1_year} 工业覆盖率从 {row['t0_ratio'] * 100:.2f}% 增至 {row['t1_ratio'] * 100:.2f}%，变化 {delta * 100:.2f}%。但未找到该厂区直接对应的声明，建议补充厂区级别数据。"
            issue_type = "missing_facility_disclosure"
            confidence = remote_conf * 0.7
            case_id = "N/A"
        else:
            claim = relevant_claims[0]
            suspected_statement = f"{claim['item']} = {claim['value']} {claim['unit']} (年份: {claim['year']})"
            evidence_file = "round2_public_blind_real_company_claims.csv"
            evidence_field = f"public_case_id: {claim['case_id']}"
            consistency = check_physical_consistency(metrics, company, delta, t0_year, t1_year)
            detection_reasoning = f"遥感检测：该厂区 {t0_year}→{t1_year} 工业覆盖率从 {row['t0_ratio'] * 100:.2f}% 增至 {row['t1_ratio'] * 100:.2f}%，变化 {delta * 100:.2f}%。"
            if consistency["reasoning"]:
                detection_reasoning += f" 交叉验证：{consistency['reasoning']}"
            issue_type = consistency["issue_types"][0] if consistency["issue_types"] else "no_obvious_issue"
            confidence = min(remote_conf, consistency["confidence"])
            case_id = claim["case_id"]

        submission = {
            "submission_id": f"RS_{facility_name[:20]}_{t1_year}",
            "public_case_id": case_id,
            "company_name": company,
            "year_or_period": f"{t0_year}-{t1_year}",
            "detection_module": "remote_sensing",
            "suspected_statement_or_metric": suspected_statement[:200],  # 截断过长
            "evidence_file": evidence_file,
            "evidence_field_or_section": evidence_field,
            "detection_reasoning": detection_reasoning[:500],
            "suspected_issue_type": issue_type,
            "confidence_0_to_1": confidence,
            "recommended_follow_up": "建议获取更高分辨率商业影像（如Planet）精确定位，并调取该厂区实际用电/产能数据做进一步验证。"
        }
        submissions.append(submission)

    out_df = pd.DataFrame(submissions)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 提交报告已生成: {OUTPUT_CSV}")
    print(f"   共 {len(out_df)} 条检测记录")
    if not out_df.empty:
        print("\n问题类型分布:")
        print(out_df["suspected_issue_type"].value_counts())


if __name__ == "__main__":
    main()



# ================= 硬编码配置 =================
FINAL_RESULTS_PATH = r"/\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\results\final_results.csv"
CLAIMS_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_REAL_COMPANIES_20260718\01_send_to_group3_PUBLIC_BLIND\round2_public_blind_real_company_claims.csv"
METRIC_PANEL_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_REAL_COMPANIES_20260718\01_send_to_group3_PUBLIC_BLIND\round2_public_blind_real_company_metric_panel.csv"
GREEN_POWER_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_REAL_COMPANIES_20260718\01_send_to_group3_PUBLIC_BLIND\round2_public_blind_real_company_green_power_carbon_credit.csv"
OUTPUT_CSV = r"D:\Pycharm Projects\torch\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\submission_ready.csv"

# 厂区名称到公司名称的映射（用于匹配 claims 中的 company_name）
FACILITY_TO_COMPANY = {
    "BYD_Company_Limited_BYD_Pingshan_headquarters_manufacturing_campus": "BYD Company Limited",
    "BYD_Company_Limited_BYD_Xiaomo_Industrial_Park": "BYD Company Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Liyang_Production_Base": "Contemporary Amperex Technology Co., Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Ningde_headquarters_base": "Contemporary Amperex Technology Co., Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Yibin_Production_Base": "Contemporary Amperex Technology Co., Limited",
    "Contemporary_Amperex_Technology_Co._Limited_CATL_Zhaoqing_Production_Base": "Contemporary Amperex Technology Co., Limited",
    "Midea_Group_Co._Ltd._Hefei_Laundry_Appliance_Factory": "Midea Group Co., Ltd.",
    "Midea_Group_Co._Ltd._Midea_Chongqing_Factory_Parks": "Midea Group Co., Ltd.",
    "Midea_Group_Co._Ltd._Midea_Shunde_headquarters_factory_area": "Midea Group Co., Ltd.",
}
