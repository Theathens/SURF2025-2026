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
    sat_df = _reindex(sat_df, sat_cols)
    sat_df.to_csv(SUBMISSION_SATELLITE, index=False, encoding="utf-8-sig")
    print(f"[step6] satellite -> {SUBMISSION_SATELLITE}  行数={len(sat_df)}")

    # ---- structured ----
    struc_cols = _template_columns(TEMPLATE_STRUCTURED)
    struc_df = findings[findings.get("detection_channel", "") == "structured"].copy()
    # 结构化模板的字段名与 findings 略有不同, 做一次映射
    field_map = {
        "table_name": struc_df.get("table_name", ""),
        "field_name": struc_df.get("field_name", ""),
        "observed_value": struc_df.get("observed_value", struc_df.get("reported_value", "")),
        "calculation_or_test": struc_df.get("calculation_or_test",
                                            struc_df.get("calculation", "")),
        "evidence_file": struc_df.get("evidence_file", ""),
        "evidence_location": struc_df.get("evidence_location", ""),
    }
    for k, v in field_map.items():
        if k not in struc_df.columns:
            struc_df[k] = v
    struc_df = _reindex(struc_df, struc_cols)
    struc_df.to_csv(SUBMISSION_STRUCTURED, index=False, encoding="utf-8-sig")
    print(f"[step6] structured -> {SUBMISSION_STRUCTURED}  行数={len(struc_df)}")

    # ---- text ----
    text_cols = _template_columns(TEMPLATE_TEXT)
    text_df = findings[findings.get("detection_channel", "") == "text"].copy()
    if "report_file" not in text_df.columns and "evidence_file" in text_df.columns:
        text_df["report_file"] = text_df["evidence_file"]
    if "section_or_paragraph" not in text_df.columns and "evidence_location" in text_df.columns:
        text_df["section_or_paragraph"] = text_df["evidence_location"]
    if "statement" not in text_df.columns and "calculation" in text_df.columns:
        text_df["statement"] = text_df["calculation"]
    if "evidence_or_comparison" not in text_df.columns:
        text_df["evidence_or_comparison"] = text_df.get("notes", "")
    text_df = _reindex(text_df, text_cols)
    text_df.to_csv(SUBMISSION_TEXT, index=False, encoding="utf-8-sig")
    print(f"[step6] text -> {SUBMISSION_TEXT}  行数={len(text_df)}")

    print("\n=== channel 分布 ===")
    if not findings.empty and "detection_channel" in findings.columns:
        print(findings["detection_channel"].value_counts())


if __name__ == "__main__":
    dispatch()
