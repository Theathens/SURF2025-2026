# step4_merge_and_align.py
"""
Step 4: 合并设施趋势和公司趋势，加载AOI声明文本，准备交叉验证。
输出：合并后的数据集，包含设施和公司变化率。
"""

import pandas as pd
from config import FACILITY_TRENDS_OUTPUT, COMPANY_TRENDS_OUTPUT, AOI_CLAIMS_FILE, MERGED_DATA_OUTPUT


def load_aoi_claims(file_path):
    df = pd.read_csv(file_path)
    # 提取相关字段：public_claim_id, firm_id, facility_id, review_period, public_esg_claim_text
    return df[['public_claim_id', 'firm_id', 'facility_id', 'facility_name', 'review_period', 'public_esg_claim_text']]


def merge_data(facility_trends, company_trends, aoi_claims):
    # 合并设施和公司趋势
    merged = pd.merge(facility_trends, company_trends, on='firm_id', how='left', suffixes=('_facility', '_company'))

    # 合并AOI声明（按facility_id）
    merged = pd.merge(merged, aoi_claims, on=['firm_id', 'facility_id'], how='left')

    # 保存
    merged.to_csv(MERGED_DATA_OUTPUT, index=False)
    print(f"合并数据已保存至 {MERGED_DATA_OUTPUT}")
    return merged


if __name__ == "__main__":
    facility_trends = pd.read_csv(FACILITY_TRENDS_OUTPUT)
    company_trends = pd.read_csv(COMPANY_TRENDS_OUTPUT)
    aoi_claims = load_aoi_claims(AOI_CLAIMS_FILE)
    merged = merge_data(facility_trends, company_trends, aoi_claims)
    print(merged.head())