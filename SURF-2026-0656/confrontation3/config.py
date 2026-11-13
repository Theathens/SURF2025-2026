# config.py
"""
confrontation3 —— Round3 造假数据（virtual + real 两个 cohort）遥感层过滤侦测配置。
参考 confrontation_virtual / confrontation2 的目录约定。
"""
import os

# ==================== 输入根目录 ====================
PACKAGE_DIR = r"/\confrontation3\26.7.26\05_group3_public_blind_20260724"

# --- virtual cohort（数据齐备，无需抓取） ---
VIRTUAL_DIR = os.path.join(PACKAGE_DIR, "virtual_companies")
VIRTUAL_PROXY_FILE = os.path.join(VIRTUAL_DIR, "structured", "facility_satellite_proxy.csv")
VIRTUAL_METRICS_FILE = os.path.join(VIRTUAL_DIR, "structured", "annual_esg_metrics.csv")
VIRTUAL_GREEN_FILE = os.path.join(VIRTUAL_DIR, "structured", "green_power_carbon.csv")
VIRTUAL_REPORTS_DIR = os.path.join(VIRTUAL_DIR, "reports")

# --- real cohort（AOI 已就绪但遥感字段全空，需 GEE 抓取） ---
REAL_DIR = os.path.join(PACKAGE_DIR, "real_companies")
REAL_CONTEXT_FILE = os.path.join(REAL_DIR, "structured", "facility_satellite_context_real.csv")
REAL_METRICS_FILE = os.path.join(REAL_DIR, "structured", "annual_esg_metrics_real_public.csv")
REAL_GREEN_FILE = os.path.join(REAL_DIR, "structured", "green_power_carbon_real_public.csv")
REAL_REPORTS_DIR = os.path.join(REAL_DIR, "reports")

# --- 提交模板 ---
TEMPLATE_DIR = os.path.join(PACKAGE_DIR, "return_templates")
TEMPLATE_SATELLITE = os.path.join(TEMPLATE_DIR, "satellite_detection_return_template.csv")
TEMPLATE_STRUCTURED = os.path.join(TEMPLATE_DIR, "structured_detection_return_template.csv")
TEMPLATE_TEXT = os.path.join(TEMPLATE_DIR, "text_detection_return_template.csv")

# ==================== 输出目录 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# step1 汇总
UNIFIED_PROXY_OUTPUT = os.path.join(DATA_DIR, "unified_facility_proxy.csv")
UNIFIED_METRICS_OUTPUT = os.path.join(DATA_DIR, "unified_annual_metrics.csv")
# step2 GEE 抓取结果
REAL_GEE_OUTPUT = os.path.join(DATA_DIR, "real_gee_proxy.csv")
# step3/4 趋势
FACILITY_TRENDS_OUTPUT = os.path.join(DATA_DIR, "facility_trends.csv")
COMPANY_TRENDS_OUTPUT = os.path.join(DATA_DIR, "company_trends.csv")
MERGED_OUTPUT = os.path.join(DATA_DIR, "merged_trends.csv")
# step5 发现
FINDINGS_OUTPUT = os.path.join(DATA_DIR, "raw_findings.csv")
# step6 三张提交表
SUBMISSION_SATELLITE = os.path.join(DATA_DIR, "satellite_detection_submission.csv")
SUBMISSION_STRUCTURED = os.path.join(DATA_DIR, "structured_detection_submission.csv")
SUBMISSION_TEXT = os.path.join(DATA_DIR, "text_detection_submission.csv")

# ==================== 检测阈值 ====================
THRESHOLDS = {
    # 物理扩张 vs 能耗
    "built_expansion_pct": 10.0,        # 建成区累计增长 > 10% 视为显著扩张
    "energy_growth_pct": 5.0,           # 能耗增长 < 5%（或负）视为不匹配
    # PV 面积 vs 可再生比例
    "pv_growth_pct": 30.0,              # PV 面积累计增长 > 30%
    "renewable_delta_pp": 5.0,          # 可再生占比提升 < 5 个百分点视为不匹配
    # 夜光 vs 产量
    "nightlight_delta_min": 0.0,        # 夜光基本没涨（<= 0）
    "production_growth_pct": 20.0,      # 但产量增长 > 20%
    # 声明 PV 面积 vs 实测偏差
    "pv_claim_dev_pct": 20.0,
    # PV 名义产能推算一致性（面积 → MW，1MW ≈ 6000 m² 屋顶）
    "pv_area_to_mw_ratio": 15000.0,
    "pv_capacity_dev_pct": 25.0,
    # 单位产品能耗（能耗/产量）逆势下降
    "energy_intensity_drop_pct": -15.0,  # 强度下降超过 15% 需要遥感佐证
    # 夜光/热岛 与 排放 一致性
    "heat_index_delta_min": 0.5,
    "scope1_growth_pct": 0.0,           # 热岛升高但 scope1 未涨
}

CONFIDENCE_BASE = {
    "physical_expansion_energy_mismatch": 0.70,
    "pv_growth_renewable_mismatch": 0.65,
    "nightlight_production_mismatch": 0.55,
    "claim_pv_inflation": 0.70,
    "pv_capacity_area_mismatch": 0.60,
    "energy_intensity_suspicious_drop": 0.55,
    "heat_scope1_mismatch": 0.50,
}
DEFAULT_CONFIDENCE = 0.50

# ==================== GEE 抓取配置（仅 real cohort） ====================
GEE_CONFIG = {
    "project_id": "silent-elevator-500103-j4",   # 与 confrontation2 保持一致，如需可改
    "cloud_thresh": 30,
    "request_interval": 2.0,
    "output_dir": os.path.join(DATA_DIR, "real_gee_cache"),
}

COHORT_VIRTUAL = "controlled_company"
COHORT_REAL = "real_public_reference"


# 可独立控制图像导出（在 step2 中定义，也可移至 config）
IMAGE_EXPORT_CONFIG = {
    "enabled": True,          # 设为 False 可关闭图片导出（仅算统计值）
    "scale": 10,
    "image_size": 512,
    "cloud_thresh": 30,
}