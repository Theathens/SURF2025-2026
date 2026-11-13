# step4_company_trends.py
"""
Step 4: 公司层年报指标趋势。
输入: unified_annual_metrics.csv
输出: company_trends.csv   —— 每家 firm 一行, 首末年百分比变化 + 关键强度指标
"""
import os
import pandas as pd
import numpy as np

from config import UNIFIED_METRICS_OUTPUT, COMPANY_TRENDS_OUTPUT, DATA_DIR

# 新增：全 YoY 详细表输出（保留所有中间年异常）
YOY_DETAILS_OUTPUT = os.path.join(DATA_DIR, "company_yoy_details.csv")


def _pct(a, b):
    if pd.isna(a) or pd.isna(b) or a == 0:
        return np.nan
    return (b - a) / a * 100.0


def _first_last(g, col):
    valid = g[g[col].notna()]
    if len(valid) < 2:
        return np.nan, np.nan, np.nan, np.nan
    first, last = valid.iloc[0], valid.iloc[-1]
    return first[col], last[col], int(first["year"]), int(last["year"])


def compute(metrics: pd.DataFrame):
    """返回 (trends_df, yoy_details_df)。

    trends_df 保持原有首末年 + max_yoy 字段（向下兼容）。
    yoy_details_df 是新增的长表：每个 (firm, metric, year) 一行，
    保存 signed_pct（有符号变化）和 abs_pct（绝对变化）。
    """
    rows = []
    yoy_rows = []  # 新增：所有 YoY 详细记录（长表）
    for (cohort, firm), g in metrics.groupby(["cohort", "firm_id"]):
        g = g.sort_values("year")
        row = {
            "cohort": cohort,
            "firm_id": firm,
            "company_name": g["company_name"].iloc[0],
            "first_year_available": int(g["year"].min()) if pd.notna(g["year"].min()) else None,
            "last_year_available": int(g["year"].max()) if pd.notna(g["year"].max()) else None,
        }

        pairs = {
            "production_volume": "prod",
            "total_energy_mwh": "energy",
            "electricity_mwh": "elec",
            "purchased_electricity_mwh": "purchased_elec",
            "self_generated_pv_mwh": "self_pv",
            "green_power_purchased_mwh": "green_purchased",
            "green_certificates_mwh": "green_cert",
            "renewable_electricity_mwh": "renewable_mwh",
            "renewable_electricity_ratio_pct": "renewable_ratio",
            "scope1_emission_tco2e": "scope1",
            "scope2_market_based_tco2e": "scope2",
            "scope1_2_emission_tco2e": "scope12",
            "scope3_emission_tco2e": "scope3",
            "emissions_intensity_tco2e_per_production_unit": "emissions_intensity",
            "energy_intensity_mwh_per_production_unit": "energy_intensity",
            "water_withdrawal_m3": "water",
            "solid_waste_tonnes": "waste",
            "revenue_million_cny": "revenue",
        }
        for src, short in pairs.items():
            if src not in g.columns:
                continue
            a, b, ya, yb = _first_last(g, src)
            row[f"{short}_start"] = a
            row[f"{short}_end"] = b
            row[f"{short}_start_year"] = ya
            row[f"{short}_end_year"] = yb
            # 可再生比例用绝对差（百分点）而不是百分率
            if short == "renewable_ratio":
                row[f"{short}_delta_pp"] = (b - a) if pd.notna(a) and pd.notna(b) else np.nan
            else:
                row[f"{short}_pct_change"] = _pct(a, b)

        # 新增：year-over-year 全部变化检测（针对 LONGi 2023 类中间年造假）
        # 同时输出 max_yoy（向下兼容）和 yoy_details（保留所有年份 signed 值）
        for src, short in pairs.items():
            if src not in g.columns:
                continue
            valid = g[g[src].notna()].sort_values("year")
            if len(valid) < 2:
                row[f"{short}_max_yoy_pct"] = np.nan
                row[f"{short}_max_yoy_year"] = np.nan
                continue
            max_abs_pct = 0.0
            max_year = None
            for i in range(1, len(valid)):
                prev_val = valid.iloc[i-1][src]
                curr_val = valid.iloc[i][src]
                curr_year = int(valid.iloc[i]["year"])
                prev_year = int(valid.iloc[i-1]["year"])

                # renewable_ratio 用绝对差（pp），其他用百分比变化（%）
                if short == "renewable_ratio":
                    signed_pct = curr_val - prev_val  # 有符号百分点变化
                    abs_pct = abs(signed_pct)
                    unit_label = "pp"
                else:
                    if prev_val == 0:
                        continue
                    signed_pct = (curr_val - prev_val) / prev_val * 100.0
                    abs_pct = abs(signed_pct)
                    unit_label = "%"

                # 追加到长表（每个 YoY 一行，保留符号）
                yoy_rows.append({
                    "cohort": cohort,
                    "firm_id": firm,
                    "company_name": g["company_name"].iloc[0],
                    "metric": src,
                    "metric_short": short,
                    "prev_year": prev_year,
                    "curr_year": curr_year,
                    "prev_value": prev_val,
                    "curr_value": curr_val,
                    "signed_yoy": signed_pct,
                    "abs_yoy": abs_pct,
                    "unit": unit_label,
                })

                if abs_pct > max_abs_pct:
                    max_abs_pct = abs_pct
                    max_year = curr_year
            row[f"{short}_max_yoy_pct"] = max_abs_pct if max_year else np.nan
            row[f"{short}_max_yoy_year"] = max_year
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(yoy_rows)


def main():
    metrics = pd.read_csv(UNIFIED_METRICS_OUTPUT)
    df, yoy_df = compute(metrics)
    df.to_csv(COMPANY_TRENDS_OUTPUT, index=False, encoding="utf-8-sig")
    yoy_df.to_csv(YOY_DETAILS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[step4] YoY 详细表已保存: {YOY_DETAILS_OUTPUT}  行数={len(yoy_df)}")
    print(f"[step4] 公司趋势已保存: {COMPANY_TRENDS_OUTPUT}  行数={len(df)}")
    show_cols = [c for c in [
        "cohort", "firm_id", "prod_pct_change", "energy_pct_change",
        "renewable_ratio_delta_pp", "scope1_pct_change", "scope2_pct_change"
    ] if c in df.columns]
    print(df[show_cols].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
