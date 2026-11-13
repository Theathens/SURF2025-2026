# step6_write_submissions.py
"""
Step 6: 将 raw_findings.csv 分发到三张官方返回模板:
  - satellite_detection_return_template.csv
  - structured_detection_return_template.csv
  - text_detection_return_template.csv

保持列头与 return_templates/*.csv 完全一致。
"""
import os
import pandas as pd

from config import (
    FINDINGS_OUTPUT,
    TEMPLATE_SATELLITE, TEMPLATE_STRUCTURED, TEMPLATE_TEXT,
    SUBMISSION_SATELLITE, SUBMISSION_STRUCTURED, SUBMISSION_TEXT,
)


def _template_columns(path: str):
    return list(pd.read_csv(path, nrows=0).columns)


def _reindex(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols]


def dispatch():
    if not os.path.exists(FINDINGS_OUTPUT):
        raise FileNotFoundError(f"未找到 {FINDINGS_OUTPUT}, 请先运行 step5")

    findings = pd.read_csv(FINDINGS_OUTPUT)
    if findings.empty:
        print("[step6] findings 为空, 生成空的提交表")

    # ---- satellite ----
    sat_cols = _template_columns(TEMPLATE_SATELLITE)
    sat_df = findings[findings.get("detection_channel", "") == "satellite"].copy()
    # 确保关键字段存在
    for col in ["spatial_boundary", "data_source", "calculation", "confidence", "conclusion", "notes"]:
        if col not in sat_df.columns:
            sat_df[col] = ""
    sat_df = _reindex(sat_df, sat_cols)
    sat_df.to_csv(SUBMISSION_SATELLITE, index=False, encoding="utf-8-sig")
    print(f"[step6] satellite -> {SUBMISSION_SATELLITE}  行数={len(sat_df)}")

    # ---- structured ----
    struc_cols = _template_columns(TEMPLATE_STRUCTURED)
    struc_df = findings[findings.get("detection_channel", "") == "structured"].copy()
    # 派生 structured 模板需要的字段
    # table_name: 可从 variable 或 claim_type 派生
    if "table_name" not in struc_df.columns or struc_df["table_name"].isna().all():
        struc_df["table_name"] = struc_df.get("variable", "annual_esg_metrics")
    # field_name: 可用 variable
    if "field_name" not in struc_df.columns or struc_df["field_name"].isna().all():
        struc_df["field_name"] = struc_df.get("variable", "")
    # observed_value: 可用 reported_value
    if "observed_value" not in struc_df.columns or struc_df["observed_value"].isna().all():
        struc_df["observed_value"] = struc_df.get("reported_value", "")
    # unit: 可用 unit
    if "unit" not in struc_df.columns or struc_df["unit"].isna().all():
        struc_df["unit"] = struc_df.get("unit", "")
    # calculation_or_test: 可用 calculation
    if "calculation_or_test" not in struc_df.columns or struc_df["calculation_or_test"].isna().all():
        struc_df["calculation_or_test"] = struc_df.get("calculation", "")
    # evidence_file: 可从 data_source 或 report_file 派生
    if "evidence_file" not in struc_df.columns or struc_df["evidence_file"].isna().all():
        # 优先使用 report_file，否则用 data_source 或 default
        if "report_file" in struc_df.columns and not struc_df["report_file"].isna().all():
            struc_df["evidence_file"] = struc_df["report_file"]
        elif "data_source" in struc_df.columns:
            struc_df["evidence_file"] = struc_df["data_source"]
        else:
            struc_df["evidence_file"] = "unified_annual_metrics.csv"
    # evidence_location: 用 evidence_location 或 year
    if "evidence_location" not in struc_df.columns or struc_df["evidence_location"].isna().all():
        if "evidence_location" in struc_df.columns:
            struc_df["evidence_location"] = struc_df["evidence_location"]
        else:
            struc_df["evidence_location"] = "Year " + struc_df["year"].astype(str)
    # confidence
    if "confidence" not in struc_df.columns:
        struc_df["confidence"] = 0.5
    # conclusion
    if "conclusion" not in struc_df.columns:
        struc_df["conclusion"] = ""
    # notes
    if "notes" not in struc_df.columns:
        struc_df["notes"] = ""

    # 确保列顺序与模板一致
    struc_df = _reindex(struc_df, struc_cols)
    struc_df.to_csv(SUBMISSION_STRUCTURED, index=False, encoding="utf-8-sig")
    print(f"[step6] structured -> {SUBMISSION_STRUCTURED}  行数={len(struc_df)}")

    # ---- text ----
    text_cols = _template_columns(TEMPLATE_TEXT)
    text_df = findings[findings.get("detection_channel", "") == "text"].copy()
    # 派生 text 模板需要的字段
    # report_file
    if "report_file" not in text_df.columns or text_df["report_file"].isna().all():
        text_df["report_file"] = text_df.get("data_source", "")
    # section_or_paragraph: 可用 evidence_location
    if "section_or_paragraph" not in text_df.columns or text_df["section_or_paragraph"].isna().all():
        text_df["section_or_paragraph"] = text_df.get("evidence_location", "")
    # statement: 可用 calculation 或 variable
    if "statement" not in text_df.columns or text_df["statement"].isna().all():
        text_df["statement"] = text_df.get("calculation", text_df.get("variable", ""))
    # evidence_or_comparison: 可用 notes
    if "evidence_or_comparison" not in text_df.columns or text_df["evidence_or_comparison"].isna().all():
        text_df["evidence_or_comparison"] = text_df.get("notes", "")
    # confidence
    if "confidence" not in text_df.columns:
        text_df["confidence"] = 0.5
    # conclusion
    if "conclusion" not in text_df.columns:
        text_df["conclusion"] = ""
    # notes
    if "notes" not in text_df.columns:
        text_df["notes"] = ""

    text_df = _reindex(text_df, text_cols)
    text_df.to_csv(SUBMISSION_TEXT, index=False, encoding="utf-8-sig")
    print(f"[step6] text -> {SUBMISSION_TEXT}  行数={len(text_df)}")

    print("\n=== channel 分布 ===")
    if not findings.empty and "detection_channel" in findings.columns:
        print(findings["detection_channel"].value_counts())


if __name__ == "__main__":
    dispatch()