# step3_load_company_metrics.py
import pandas as pd
import numpy as np
from config import ANNUAL_METRICS_FILE, COMPANY_TRENDS_OUTPUT


def load_company_metrics(file_path):
    df = pd.read_csv(file_path)
    df = df[df['year'].notna()]
    df['year'] = df['year'].astype(int)

    # 转换数值列
    numeric_cols = [
        'revenue_million_cny', 'production_volume', 'employee_count',
        'factories_count', 'total_energy_mwh', 'electricity_mwh',
        'purchased_electricity_mwh', 'self_generated_pv_mwh',
        'green_power_purchased_mwh', 'green_certificates_mwh',
        'renewable_electricity_mwh', 'renewable_electricity_ratio_pct',
        'natural_gas_mwh', 'diesel_mwh', 'scope1_emission_tco2e',
        'scope2_market_based_tco2e', 'scope1_2_emission_tco2e',
        'scope3_emission_tco2e', 'emissions_intensity_tco2e_per_production_unit',
        'water_withdrawal_m3', 'solid_waste_tonnes'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calculate_company_trends(metrics_df):
    trends = []
    for firm_id, group in metrics_df.groupby('firm_id'):
        group = group.sort_values('year')
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        start_year = first['year']
        end_year = last['year']

        # 提取关键指标
        start_prod = first['production_volume']
        end_prod = last['production_volume']
        start_energy = first['total_energy_mwh']
        end_energy = last['total_energy_mwh']
        start_elec = first['electricity_mwh']
        end_elec = last['electricity_mwh']
        start_scope1 = first['scope1_emission_tco2e']
        end_scope1 = last['scope1_emission_tco2e']
        start_scope2 = first['scope2_market_based_tco2e']
        end_scope2 = last['scope2_market_based_tco2e']
        start_renewable = first['renewable_electricity_ratio_pct']
        end_renewable = last['renewable_electricity_ratio_pct']

        # 计算变化率（处理NaN）
        def pct_change(start, end):
            if pd.isna(start) or pd.isna(end) or start == 0:
                return np.nan
            return (end - start) / start * 100

        prod_pct = pct_change(start_prod, end_prod)
        energy_pct = pct_change(start_energy, end_energy)
        elec_pct = pct_change(start_elec, end_elec)
        scope1_pct = pct_change(start_scope1, end_scope1)
        scope2_pct = pct_change(start_scope2, end_scope2)
        renewable_delta = end_renewable - start_renewable if not pd.isna(start_renewable) and not pd.isna(
            end_renewable) else np.nan

        trends.append({
            'firm_id': firm_id,
            'company_name': group['company_name'].iloc[0],
            'start_year': start_year,
            'end_year': end_year,
            'prod_pct_change': prod_pct,
            'energy_pct_change': energy_pct,
            'elec_pct_change': elec_pct,
            'scope1_pct_change': scope1_pct,
            'scope2_pct_change': scope2_pct,
            'renewable_ratio_delta': renewable_delta,
            'start_energy': start_energy,
            'end_energy': end_energy,
            'start_scope1': start_scope1,
            'end_scope1': end_scope1,
        })
    trend_df = pd.DataFrame(trends)
    trend_df.to_csv(COMPANY_TRENDS_OUTPUT, index=False)
    print(f"公司趋势已保存至 {COMPANY_TRENDS_OUTPUT}")
    return trend_df


if __name__ == "__main__":
    metrics = load_company_metrics(ANNUAL_METRICS_FILE)
    company_trends = calculate_company_trends(metrics)
    print(company_trends.head())