# step1_load_remote_features.py
"""
Step 1: 加载虚拟遥感特征数据，过滤无效记录，保留有实际AOI的设施。
输出：清理后的设施-年度面板数据。
"""

import pandas as pd
from config import REMOTE_FEATURES_FILE


def load_remote_features(file_path):
    df = pd.read_csv(file_path)
    print(f"原始遥感特征记录数: {len(df)}")

    # 过滤无效记录
    df = df[df['aoi_area_ha'] > 0]
    df = df[df['satellite_data_source'] != 'not_applicable']
    print(f"过滤后记录数: {len(df)}")

    # 年份转为整数
    df['year'] = df['year'].astype(int)

    # ---------- 关键修复：将数值列转换为 float ----------
    numeric_cols = [
        'aoi_area_ha', 'built_up_area_ha', 'roof_pv_area_m2',
        'estimated_rooftop_pv_capacity_mw', 'annual_night_light_mean',
        'ndvi_buffer_mean', 'land_surface_heat_index',
        'production_capacity', 'built_up_area_change_yoy_ha',
        'roof_pv_area_change_yoy_m2', 'night_light_change_yoy'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # ----------------------------------------------------

    # 按设施和年份排序
    df = df.sort_values(['firm_id', 'facility_id', 'year'])
    return df


if __name__ == "__main__":
    df = load_remote_features(REMOTE_FEATURES_FILE)
    print("遥感特征加载完成。")
    print(df.head())