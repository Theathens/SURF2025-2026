# zonal_loader.py
import pandas as pd
from typing import Dict, Optional
from base_models import EvidencePackage

def load_patch_evidence(
    t0_excel_path: str,
    t1_excel_path: str,
    patch_name: str
) -> Optional[EvidencePackage]:
    df_t0 = pd.read_excel(t0_excel_path, sheet_name="逐Patch活动指数")
    df_t1 = pd.read_excel(t1_excel_path, sheet_name="逐Patch活动指数")

    row_t0 = df_t0[df_t0["Patch 文件名"] == patch_name]
    row_t1 = df_t1[df_t1["Patch 文件名"] == patch_name]

    if row_t0.empty or row_t1.empty:
        print(f"[警告] 两期数据中未找到 patch: {patch_name}")
        return None

    builtup_t0 = row_t0["工业像素数"].iloc[0]
    builtup_t1 = row_t1["工业像素数"].iloc[0]
    builtup_delta = builtup_t1 - builtup_t0

    evidence = EvidencePackage(
        builtup_delta=builtup_delta,
        claim_direction=1,      # 默认增长，后续可修改
        claim_category="OPERATION"
    )
    return evidence

def load_all_patches(
    t0_excel_path: str,
    t1_excel_path: str
) -> Dict[str, EvidencePackage]:
    df_t0 = pd.read_excel(t0_excel_path, sheet_name="逐Patch活动指数")
    all_patches = df_t0["Patch 文件名"].tolist()

    result = {}
    for patch in all_patches:
        ev = load_patch_evidence(t0_excel_path, t1_excel_path, patch)
        if ev is not None:
            result[patch] = ev
    return result