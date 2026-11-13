# -*- coding: utf-8 -*-
"""Merge per-company claim files into structured_claims.json and validate."""
import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(BASE)  # package root: 05_group3_public_blind_20260724

PARTS = [
    "claims_anchor.json",
    "claims_nova.json",
    "claims_orion.json",
    "claims_haier.json",
    "claims_longi.json",
    "claims_sungrow.json",
]

REQUIRED = [
    "claim_id", "cohort", "firm_id", "company_name", "report_file", "year",
    "section", "claim_type", "claim_category", "claim_direction",
    "claim_value", "claim_unit", "claim_text", "evidence_location", "confidence",
]

all_claims = []
problems = []

for p in PARTS:
    path = os.path.join(BASE, p)
    with open(path, "r", encoding="utf-8") as f:
        claims = json.load(f)
    assert isinstance(claims, list), f"{p}: not a JSON array"
    for c in claims:
        # qualitative claims and data-availability statements carry no numeric value
        non_numeric = c.get("claim_category") == "qualitative" or c.get("claim_type") == "data_availability_statement"
        for k in REQUIRED:
            if k not in c and not (k in ("claim_value", "claim_unit") and non_numeric):
                problems.append(f"{p}: {c.get('claim_id','?')} missing field {k}")
        if not non_numeric:
            if not isinstance(c.get("claim_value"), (int, float)) or isinstance(c.get("claim_value"), bool):
                problems.append(f"{p}: {c.get('claim_id','?')} claim_value not numeric")
            if not c.get("claim_unit"):
                problems.append(f"{p}: {c.get('claim_id','?')} missing claim_unit for quantitative claim")
        # normalize schema: add null value/unit for non-numeric claims
        if non_numeric:
            c.setdefault("claim_value", None)
            c.setdefault("claim_unit", None)
        if "claim_direction" in c and c["claim_direction"] not in ("increase", "decrease", "control", "low", "unknown"):
            problems.append(f"{p}: {c.get('claim_id','?')} bad claim_direction {c['claim_direction']}")
        if not c.get("evidence_location"):
            problems.append(f"{p}: {c.get('claim_id','?')} missing evidence_location")
    all_claims.extend(claims)

# claim_id uniqueness
ids = [c["claim_id"] for c in all_claims]
dups = sorted({i for i in ids if ids.count(i) > 1})
if dups:
    problems.append(f"duplicate claim_ids: {dups}")

# cohort consistency: cohort must match firm category
for c in all_claims:
    if c["firm_id"].startswith("v_") and c["cohort"] != "controlled_company":
        problems.append(f"{c['claim_id']}: virtual firm with cohort {c['cohort']}")
    if not c["firm_id"].startswith("v_") and c["cohort"] != "real_public_reference":
        problems.append(f"{c['claim_id']}: real firm with cohort {c['cohort']}")

if problems:
    print("VALIDATION FAILED:")
    for p in problems:
        print("  -", p)
    sys.exit(1)

out_path = os.path.join(PKG, "structured_claims.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_claims, f, ensure_ascii=False, indent=2)

# summary stats
from collections import Counter
by_firm = Counter(c["firm_id"] for c in all_claims)
by_cat = Counter(c["claim_category"] for c in all_claims)
by_year = Counter(c["year"] for c in all_claims)
by_type = Counter(c["claim_type"] for c in all_claims)

print(f"OK: {len(all_claims)} claims -> {out_path}")
print("per firm:", dict(by_firm))
print("per category:", dict(by_cat))
print("per year:", dict(by_year))
print("per type:", dict(by_type))
