#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4_calculate_delta.py (重构版)
==================================
功能：
  1. 读取 step3 生成的 inference_results.csv
  2. 按 facility_name 分组配对 T0/T1
  3. 计算工业覆盖率变化量（绝对差值 + 相对变化率）
  4. 生成最终结果 CSV，供 step5 组装证据

输入：inference_results.csv
输出：final_results.csv（含每个厂区的双期对比结果）
"""

import pandas as pd
import os

# ================= 硬编码配置 =================
INPUT_CSV = r"D:\Pycharm Projects\torch\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\results\inference_results.csv"
OUTPUT_CSV = r"D:\Pycharm Projects\torch\confrontation2\data\01_send_to_group3_PUBLIC_BLIND\results\final_results.csv"


# =============================================


def main():
    # 读取推理结果
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 找不到推理结果文件: {INPUT_CSV}")
        print("请先运行 step3_run_unet_inference.py")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"📂 读取推理结果: {len(df)} 张影像")

    # 检查必要字段
    required_cols = ["facility_name", "year", "industrial_ratio", "industrial_pixels_scaled"]
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ 缺少必要列: {col}")
            return

    # 按厂区分组
    results = []
    for facility_name, group in df.groupby("facility_name"):
        # 过滤掉年份为空的记录
        group = group[group["year"].notna()]
        if len(group) < 2:
            print(f"⚠️  {facility_name} 只有 {len(group)} 期数据，跳过配对")
            continue

        # 按年份排序
        group_sorted = group.sort_values("year")
        t0_row = group_sorted.iloc[0]  # 最早年份（T0）
        t1_row = group_sorted.iloc[-1]  # 最晚年份（T1）

        t0_year = int(t0_row["year"])
        t1_year = int(t1_row["year"])

        # 如果两期年份相同，跳过（理论上不会发生）
        if t0_year == t1_year:
            print(f"⚠️  {facility_name} T0 和 T1 年份相同，跳过")
            continue

        # 提取覆盖率
        t0_ratio = t0_row["industrial_ratio"]
        t1_ratio = t1_row["industrial_ratio"]

        # 计算变化量
        delta_ratio = t1_ratio - t0_ratio
        delta_pct = (delta_ratio / (t0_ratio + 0.001)) * 100  # 相对变化（%）

        # 判断趋势（阈值 0.01 = 1% 的覆盖率变化）
        if delta_ratio > 0.01:
            trend = "扩张/增强 ↑"
        elif delta_ratio < -0.01:
            trend = "收缩/减弱 ↓"
        else:
            trend = "基本稳定 —"

        # 提取公司名（从 facility_name 中截取，因为命名规则是 {company}_{facility_name}）
        # 注意：我们的命名规则是 "公司名_厂区名_年份"，但这里 facility_name 字段是完整的 "公司名_厂区名"
        # 所以公司名实际上是 facility_name 中第一个下划线之前的部分
        parts = facility_name.split('_')
        if len(parts) >= 2:
            # 如果公司名本身包含下划线（如 "Contemporary_Amperex_Technology_Co._Limited"），需要特殊处理
            # 简单起见：如果长度>=4，取前3个作为公司名（仅限这种情况）
            if "Contemporary_Amperex_Technology_Co._Limited" in facility_name:
                company = "Contemporary Amperex Technology Co., Limited"
            elif "BYD_Company_Limited" in facility_name:
                company = "BYD Company Limited"
            elif "Midea_Group_Co._Ltd." in facility_name:
                company = "Midea Group Co., Ltd."
            else:
                company = parts[0]  # 备选
        else:
            company = facility_name

        # 置信度评估：如果工业覆盖率本身很低（< 0.5%），说明该厂区可能未被有效识别，
        # 因此变化量的可信度下降
        if t0_ratio < 0.005 or t1_ratio < 0.005:
            confidence = 0.3
            confidence_note = "低覆盖率，可能受缓冲区过大影响"
        elif abs(delta_ratio) < 0.01:
            confidence = 0.8
            confidence_note = "变化幅度小，检测可信度较高"
        else:
            confidence = 0.85
            confidence_note = "显著变化，检测可信度高"

        results.append({
            "company": company,
            "facility_name": facility_name,
            "t0_year": t0_year,
            "t0_ratio": round(t0_ratio, 6),
            "t1_year": t1_year,
            "t1_ratio": round(t1_ratio, 6),
            "delta_ratio": round(delta_ratio, 6),
            "delta_pct": round(delta_pct, 2),
            "trend": trend,
            "confidence": confidence,
            "confidence_note": confidence_note,
            "t0_pixels": int(t0_row["industrial_pixels_scaled"]),
            "t1_pixels": int(t1_row["industrial_pixels_scaled"]),
            "pixel_delta": int(t1_row["industrial_pixels_scaled"] - t0_row["industrial_pixels_scaled"]),
        })

    if not results:
        print("❌ 没有成功配对的厂区")
        return

    # 转换为 DataFrame
    out_df = pd.DataFrame(results)

    # 保存
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 配对完成，共 {len(out_df)} 个厂区")
    print(f"📁 输出文件: {OUTPUT_CSV}")

    # 打印摘要
    print("\n=== 变化趋势分布 ===")
    print(out_df["trend"].value_counts())

    print("\n=== 各厂区变化详情 ===")
    for _, row in out_df.iterrows():
        print(f"🏭 {row['facility_name']}")
        print(f"   {row['t0_year']} → {row['t1_year']}: {row['t0_ratio']:.2%} → {row['t1_ratio']:.2%}")
        print(f"   Δ = {row['delta_ratio']:.2%} ({row['delta_pct']:.1f}%)  {row['trend']}")
        print(f"   置信度: {row['confidence']:.2f} ({row['confidence_note']})")
        print()


if __name__ == "__main__":
    main()