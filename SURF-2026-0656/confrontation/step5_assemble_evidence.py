#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_assemble_evidence.py
============================
将最终结果组装成 EvidencePackage JSON 格式，
供后续模块化系统使用。
"""

import pandas as pd
import json
import os
from datetime import datetime

# ================= 硬编码配置 =================
INPUT_CSV = r"D:\Pycharm Projects\torch\confrontation\data\results\final_results.csv"
OUTPUT_JSON = r"D:\Pycharm Projects\torch\confrontation\data\results\evidence_packages.json"
# =============================================


def main():
    df = pd.read_csv(INPUT_CSV)
    packages = []
    for _, row in df.iterrows():
        package = {
            "facility_name": row['facility_name'],
            "evidence": {
                "t0_year": int(row['t0_year']),
                "t0_coverage": round(row['t0_coverage'], 4),
                "t1_year": int(row['t1_year']),
                "t1_coverage": round(row['t1_coverage'], 4),
                "delta": round(row['delta'], 4),
                "delta_pct": round(row['delta_pct'], 2),
                "trend": row['trend'],
            },
            "confidence": {
                "remote_sensing": 0.85,
                "data_quality": 0.90,
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source": "Sentinel-2 + U-Net Industrial Segmentation",
                "model": "unet_industrial_combine_GEE_BEN_3_best.pth"
            }
        }
        packages.append(package)

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(packages, f, ensure_ascii=False, indent=2)

    print(f"✅ EvidencePackage 已保存至: {OUTPUT_JSON}")
    print(f"   共 {len(packages)} 个厂区")


if __name__ == "__main__":
    main()