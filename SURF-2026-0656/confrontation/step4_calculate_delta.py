#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_calculate_delta.py
==========================
将 T0 和 T1 的推理结果配对，计算工业覆盖率的变化量，
生成趋势判断（扩张、收缩、稳定）。
"""

import pandas as pd
import os

# ================= 硬编码配置 =================
INPUT_CSV = r"D:\Pycharm Projects\torch\confrontation\data\results\inference_results.csv"
OUTPUT_CSV = r"D:\Pycharm Projects\torch\confrontation\data\results\final_results.csv"
# =============================================


def main():
    df = pd.read_csv(INPUT_CSV)
    # 按厂区分组，提取T0和T1
    # 如果有多个同厂区同一年份，保留第一个（实际应唯一）
    t0_df = df[df['year'].notna()].copy()
    # 需要确定T0和T1的年份，可根据文件名中的年份或默认逻辑
    # 这里简单做法：对每个厂区，取最小年份为T0，最大年份为T1
    # 但注意有些厂区可能只有一期，我们需要过滤
    results = []
    for facility in t0_df['facility_name'].unique():
        sub = t0_df[t0_df['facility_name'] == facility]
        if len(sub) < 2:
            print(f"⚠️ {facility} 只有一期数据，跳过")
            continue
        # 按年份排序
        sub_sorted = sub.sort_values('year')
        t0_row = sub_sorted.iloc[0]
        t1_row = sub_sorted.iloc[-1]
        if t0_row['year'] == t1_row['year']:
            print(f"⚠️ {facility} 两期年份相同，跳过")
            continue
        delta = t1_row['industrial_coverage'] - t0_row['industrial_coverage']
        # 趋势判断
        if delta > 0.05:
            trend = "扩张/增强 ↑"
        elif delta < -0.05:
            trend = "收缩/减弱 ↓"
        else:
            trend = "基本稳定 —"
        results.append({
            "facility_name": facility,
            "t0_year": t0_row['year'],
            "t0_coverage": round(t0_row['industrial_coverage'], 4),
            "t1_year": t1_row['year'],
            "t1_coverage": round(t1_row['industrial_coverage'], 4),
            "delta": round(delta, 4),
            "delta_pct": round(delta / (t0_row['industrial_coverage'] + 0.001) * 100, 2),
            "trend": trend
        })

    if not results:
        print("❌ 没有可配对的厂区数据")
        return

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ 变化量结果已保存: {OUTPUT_CSV}")
    print("\n趋势分布:")
    print(out_df['trend'].value_counts())


if __name__ == "__main__":
    main()