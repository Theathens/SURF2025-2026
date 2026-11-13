# step4_company_trends.py
"""
Step 4: 公司层年报指标趋势。
输入: unified_annual_metrics.csv
输出: company_trends.csv   —— 每家 firm 一行, 首末年百分比变化 + 关键强度指标
"""
import pandas as pd
import numpy as np

from config import UNIFIED_METRICS_OUTPUT, COMPANY_TRENDS_OUTPUT


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


def compute(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    metrics = pd.read_csv(UNIFIED_METRICS_OUTPUT)
    df = compute(metrics)
    df.to_csv(COMPANY_TRENDS_OUTPUT, index=False, encoding="utf-8-sig")
    print(f"[step4] 公司趋势已保存: {COMPANY_TRENDS_OUTPUT}  行数={len(df)}")
    show_cols = [c for c in [
        "cohort", "firm_id", "prod_pct_change", "energy_pct_change",
        "renewable_ratio_delta_pp", "scope1_pct_change", "scope2_pct_change"
    ] if c in df.columns]
    print(df[show_cols].to_string(index=False))
    return df


if __name__ == "__main__":
    main()
