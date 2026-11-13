# step1_parse_facility_csv.py
"""
步骤1：解析Group1提供的厂区CSV文件
增强版：自动解析 optional 样本的时间窗口
输入：group3_t2_core_facility_coordinates_time_windows_PUBLIC.csv
      group3_t2_optional_facility_candidates_for_feasibility_PUBLIC.csv
输出：cleaned_facility_list.csv（包含完整时间窗口）
"""

import pandas as pd
import re
from typing import Tuple, Optional

# ================= 硬编码配置区 =================
CORE_CSV_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\Group3_T2_facility_coordinates_time_windows_PUBLIC_20260716\group3_t2_core_facility_coordinates_time_windows_PUBLIC.csv"
OPTIONAL_CSV_PATH = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\Group3_T2_facility_coordinates_time_windows_PUBLIC_20260716\group3_t2_optional_facility_candidates_for_feasibility_PUBLIC.csv"
OUTPUT_CSV_PATH = r"/\confrontation\data\cleaned_facility_list.csv"

# 不可处理的坐标精度（这些点无法使用）
SKIP_PRECISION = ["city_only", "municipality_centroid"]


# ===============================================


def parse_window_to_periods(window_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析 suggested_remote_sensing_window 字符串，提取基准期和检测期。
    支持多种格式：
      - "before smart-factory period: 2012-2014; after: 2016-present"
      - "pre-production comparison: 2019-2021; post-production: 2022-present"
      - "historical baseline before 2008; normal monitoring: 2018-present"
      - 简单的 "2018-2020; 2021-2025" 等

    返回: (baseline_period, detection_period)
    若无法解析，返回 (None, None)
    """
    if pd.isna(window_str) or window_str == "":
        return None, None

    # 尝试用分号分割，取两部分
    parts = [p.strip() for p in window_str.split(';')]
    if len(parts) >= 2:
        # 取第一段为基准期，第二段为检测期
        baseline_part = parts[0]
        detection_part = parts[1]
    elif len(parts) == 1:
        # 若只有一段，尝试提取两个年份范围，例如 "2015-2017; 2019-2025" 或 "2015-2017 and 2019-2025"
        # 使用正则提取所有年份范围
        ranges = re.findall(r'(\d{4})\s*[-–]\s*(\d{4})', window_str)
        if len(ranges) >= 2:
            baseline_part = f"{ranges[0][0]}-{ranges[0][1]}"
            detection_part = f"{ranges[1][0]}-{ranges[1][1]}"
        else:
            return None, None
    else:
        return None, None

    # 从每个部分中提取年份范围（格式：\d{4}-\d{4}）
    def extract_year_range(part: str) -> Optional[str]:
        match = re.search(r'(\d{4})\s*[-–]\s*(\d{4})', part)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        # 也处理 "present" 情况，但保留原字符串
        if 'present' in part.lower():
            # 尝试找开始的年份
            start_match = re.search(r'(\d{4})', part)
            if start_match:
                # 将 present 替换为 2025（固定上限，可根据需要调整）
                return f"{start_match.group(1)}-2025"
        return None

    baseline_period = extract_year_range(baseline_part)
    detection_period = extract_year_range(detection_part)

    return baseline_period, detection_period


def parse_core_facilities(csv_path: str) -> pd.DataFrame:
    """解析核心样本CSV"""
    df = pd.read_csv(csv_path)

    # 提取关键字段
    df_clean = df[[
        "company_name", "facility_id", "facility_name",
        "latitude_wgs84", "longitude_wgs84", "coordinate_precision",
        "suggested_baseline_period", "suggested_detection_period",
        "aoi_status"
    ]].copy()

    # 过滤不可处理的坐标精度
    df_clean = df_clean[~df_clean["coordinate_precision"].isin(SKIP_PRECISION)]

    print(f"核心样本：原始 {len(df)} 个，可用 {len(df_clean)} 个")
    return df_clean


def parse_optional_facilities(csv_path: str) -> pd.DataFrame:
    """解析可选候选样本CSV，并自动解析时间窗口"""
    df = pd.read_csv(csv_path)

    # 提取关键字段
    df_clean = df[[
        "company_or_entity", "facility_candidate_name",
        "latitude_wgs84", "longitude_wgs84", "coordinate_precision",
        "suggested_remote_sensing_window", "candidate_status_for_T2"
    ]].copy()

    # 过滤不可处理的坐标精度
    df_clean = df_clean[~df_clean["coordinate_precision"].isin(SKIP_PRECISION)]

    # 解析时间窗口
    baseline_list = []
    detection_list = []
    for idx, row in df_clean.iterrows():
        window = row["suggested_remote_sensing_window"]
        baseline, detection = parse_window_to_periods(window)
        baseline_list.append(baseline)
        detection_list.append(detection)

    df_clean["suggested_baseline_period"] = baseline_list
    df_clean["suggested_detection_period"] = detection_list

    # 只保留标记为 optional 的
    df_clean = df_clean[df_clean["candidate_status_for_T2"] == "optional_candidate_for_T2_feasibility_only"]

    print(f"可选样本：原始 {len(df)} 个，可用 {len(df_clean)} 个")
    return df_clean


def main():
    # 解析核心样本
    core_df = parse_core_facilities(CORE_CSV_PATH)

    # 解析可选样本
    optional_df = parse_optional_facilities(OPTIONAL_CSV_PATH)

    # 统一列名
    core_df = core_df.rename(columns={
        "company_name": "company",
        "facility_id": "facility_id",
        "facility_name": "facility_name",
        "latitude_wgs84": "lat",
        "longitude_wgs84": "lon",
    })

    optional_df = optional_df.rename(columns={
        "company_or_entity": "company",
        "facility_candidate_name": "facility_name",
        "latitude_wgs84": "lat",
        "longitude_wgs84": "lon",
    })
    optional_df["facility_id"] = "optional_" + optional_df.index.astype(str)
    optional_df["aoi_status"] = "point_seed_only"  # 给默认值

    # 合并
    all_df = pd.concat([core_df, optional_df], ignore_index=True)

    # 确保列顺序一致
    column_order = [
        "company", "facility_id", "facility_name", "lat", "lon",
        "coordinate_precision", "suggested_baseline_period", "suggested_detection_period",
        "aoi_status"
    ]
    all_df = all_df[column_order]

    # 保存
    all_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n合并后可用厂区总数：{len(all_df)}")
    print(f"已保存至：{OUTPUT_CSV_PATH}")

    # 打印摘要
    print("\n=== 可用厂区列表 ===")
    for idx, row in all_df.iterrows():
        print(f"  {row['facility_name']} ({row['company']})")
        print(f"   坐标: {row['lat']}, {row['lon']}")
        print(f"   精度: {row['coordinate_precision']}")
        print(f"   基准期: {row['suggested_baseline_period']} 检测期: {row['suggested_detection_period']}")
        print()


if __name__ == "__main__":
    main()




