# config.py
import os

DATA_DIR = r"C:\Users\25145\Desktop\SURF2026\T2 vs T4\ROUND2_GROUP3_PUBLIC_BLIND_3_VIRTUAL_COMPANIES_20260717\01_send_to_group3_PUBLIC_BLIND"

# 输入文件
REMOTE_FEATURES_FILE = os.path.join(DATA_DIR, "round2_public_blind_virtual_aoi_remote_sensing_features_2018_2025.csv")
ANNUAL_METRICS_FILE = os.path.join(DATA_DIR, "round2_public_blind_annual_esg_metrics_2018_2025.csv")
GREEN_POWER_FILE = os.path.join(DATA_DIR, "round2_public_blind_green_power_carbon_credit_2018_2025.csv")
AOI_CLAIMS_FILE = os.path.join(DATA_DIR, "round2_public_blind_virtual_aoi_claims.csv")
# ESG_REPORT_FILE 可选，不在脚本中使用，但可留作手工参考
ESG_REPORT_FILE = os.path.join(DATA_DIR, "round2_public_blind_esg_reports_2025.md")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 输出文件（也放在同一目录，也可调整）
FACILITY_TRENDS_OUTPUT = os.path.join(BASE_DIR, "data", "facility_trends.csv")
COMPANY_TRENDS_OUTPUT = os.path.join(BASE_DIR, "data", "company_trends.csv")
MERGED_DATA_OUTPUT = os.path.join(BASE_DIR, "data", "merged_data.csv")
SUBMISSION_OUTPUT = os.path.join(BASE_DIR, "data", "submission_ready_virtual.csv")