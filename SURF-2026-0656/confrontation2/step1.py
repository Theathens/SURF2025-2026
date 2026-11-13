# step1_parse_facility.py
"""
步骤1：解析 Round 2 真实公司厂区 CSV（适配新数据结构）
输入：round2_public_blind_facility_coordinates_time_windows.csv
输出：cleaned_facility_list.csv（含动态 AOI 半径、基准年/检测年）
"""

import pandas as pd
import os
from typing import List, Tuple, Optional

# ================= 硬编码配置区 =================
INPUT_CSV = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_REAL_COMPANIES_20260718\01_send_to_group3_PUBLIC_BLIND\round2_public_blind_facility_coordinates_time_windows.csv"
OUTPUT_CSV = r"D:\Pycharm Projects\torch\confrontation2\data\cleaned_facility_list.csv"

# 坐标精度 -> AOI 缓冲区半径（米）映射
PRECISION_RADIUS_MAP = {
    "approximate_public_address": 500,
    "administrative_street_centroid": 1500,
    "district_centroid": 3000,
    "city_only": 5000,
    "municipality_centroid": 8000,
}

DEFAULT_RADIUS = 3000


# ===============================================


def aggregate_facility(facility_id: str, group: pd.DataFrame) -> Optional[dict]:
    """
    将同一个 facility_id 的多年数据聚合为一行
    """
    # 取第一条记录的静态信息
    first_row = group.iloc[0]

    # 收集所有可用年份（去重排序）
    years = sorted(group["year"].dropna().unique().astype(int).tolist())
    if not years:
        return None

    baseline_year = years[0]
    detection_year = years[-1]

    precision = first_row.get("location_precision", "unknown")
    radius = PRECISION_RADIUS_MAP.get(precision, DEFAULT_RADIUS)

    try:
        lat = float(first_row["latitude"])
        lon = float(first_row["longitude"])
    except (ValueError, KeyError):
        return None

    return {
        "company": first_row.get("company_name", "unknown"),
        "firm_id": first_row.get("firm_id", "unknown"),
        "facility_id": facility_id,  # 直接使用传入的键
        "facility_name": first_row.get("facility_name", "unnamed"),
        "lat": lat,
        "lon": lon,
        "location_precision": precision,
        "aoi_buffer_meters": radius,
        "available_years": years,
        "baseline_year": baseline_year,
        "detection_year": detection_year,
        "aoi_status": first_row.get("aoi_polygon_status", "point_seed_only"),
        "facility_type": first_row.get("facility_type", "unknown"),
        "operational_status": first_row.get("operational_status", "unknown"),
    }


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到输入文件: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"📂 读取原始数据: {len(df)} 行")

    aggregated_rows = []
    for facility_id, group in df.groupby("facility_id"):
        row_dict = aggregate_facility(facility_id, group)  # 传入 facility_id
        if row_dict is not None:
            aggregated_rows.append(row_dict)

    if not aggregated_rows:
        print("❌ 没有聚合出任何有效厂区")
        return

    out_df = pd.DataFrame(aggregated_rows)

    column_order = [
        "company", "firm_id", "facility_id", "facility_name",
        "lat", "lon", "location_precision", "aoi_buffer_meters",
        "available_years", "baseline_year", "detection_year",
        "aoi_status", "facility_type", "operational_status"
    ]
    out_df = out_df[column_order]

    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 聚合完成，共 {len(out_df)} 个厂区")
    print(f"📁 输出文件: {OUTPUT_CSV}")

    print("\n=== 厂区摘要 ===")
    for _, row in out_df.iterrows():
        years_str = ", ".join(map(str, row["available_years"]))
        print(f"🏭 {row['facility_name']} ({row['company']})")
        print(f"   坐标: ({row['lat']:.5f}, {row['lon']:.5f})")
        print(f"   精度: {row['location_precision']} → 缓冲区半径: {row['aoi_buffer_meters']}m")
        print(f"   年份: {years_str}")
        print(f"   基准年(T0): {row['baseline_year']}  检测年(T1): {row['detection_year']}")
        print()

    print("\n=== 精度分布统计 ===")
    print(out_df["location_precision"].value_counts())


if __name__ == "__main__":
    main()