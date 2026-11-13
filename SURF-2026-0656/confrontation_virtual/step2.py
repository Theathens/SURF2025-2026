# step2_calc_facility_trends.py
"""
Step 2: 对每个设施，计算关键指标的年际变化（从最早有数据年份到2025）。
输出：每个设施的起始年、结束年、变化量和变化率。
"""

import pandas as pd
import numpy as np
from config import REMOTE_FEATURES_FILE, FACILITY_TRENDS_OUTPUT


def calculate_facility_trends(df):
    # 按设施分组
    trends = []
    for (firm_id, facility_id), group in df.groupby(['firm_id', 'facility_id']):
        # 确保按年份排序
        group = group.sort_values('year')
        if len(group) < 2:
            continue  # 需要至少两年数据

        first = group.iloc[0]
        last = group.iloc[-1]

        # 提取指标
        start_year = first['year']
        end_year = last['year']
        start_built = first['built_up_area_ha']
        end_built = last['built_up_area_ha']
        start_pv = first['roof_pv_area_m2']
        end_pv = last['roof_pv_area_m2']
        start_night = first['annual_night_light_mean']
        end_night = last['annual_night_light_mean']
        start_heat = first['land_surface_heat_index']
        end_heat = last['land_surface_heat_index']
        start_cap = first['production_capacity'] if 'production_capacity' in first else np.nan
        end_cap = last['production_capacity'] if 'production_capacity' in last else np.nan

        # 计算变化量和变化率
        built_delta = end_built - start_built
        built_pct = (built_delta / (start_built + 1e-6)) * 100  # 避免除以零

        pv_delta = end_pv - start_pv
        pv_pct = (pv_delta / (start_pv + 1e-6)) * 100 if start_pv > 0 else np.nan

        night_delta = end_night - start_night
        night_pct = (night_delta / (start_night + 1e-6)) * 100 if start_night > 0 else np.nan

        heat_delta = end_heat - start_heat
        heat_pct = (heat_delta / (start_heat + 1e-6)) * 100 if start_heat > 0 else np.nan

        # 判断扩张与否
        expansion = built_delta > 0  # 建成区增加视为扩张

        trends.append({
            'firm_id': firm_id,
            'facility_id': facility_id,
            'facility_name': group['facility_name'].iloc[0],
            'start_year': start_year,
            'end_year': end_year,
            'start_built_ha': start_built,
            'end_built_ha': end_built,
            'built_delta_ha': built_delta,
            'built_pct_change': built_pct,
            'start_pv_m2': start_pv,
            'end_pv_m2': end_pv,
            'pv_delta_m2': pv_delta,
            'pv_pct_change': pv_pct,
            'night_light_delta': night_delta,
            'heat_index_delta': heat_delta,
            'expansion_flag': expansion,
            'start_cap': start_cap,
            'end_cap': end_cap,
        })

    trend_df = pd.DataFrame(trends)
    trend_df.to_csv(FACILITY_TRENDS_OUTPUT, index=False)
    print(f"设施趋势已保存至 {FACILITY_TRENDS_OUTPUT}")
    return trend_df


if __name__ == "__main__":
    # 先运行 step1 加载数据
    from step1 import load_remote_features

    df_remote = load_remote_features(REMOTE_FEATURES_FILE)
    trends = calculate_facility_trends(df_remote)
    print(trends.head())