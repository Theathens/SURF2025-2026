# modules/module_financial.py
# 财务恒等式与排放异常核验模块

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from base_models import BaseModule, EvidencePackage, ModuleResult, ModuleStatus

# ================= 该模块的硬编码配置 =================
CONFIG = {
    # 恒等式容忍度：当 |合计 - (Scope1+Scope2)| / 合计 超过该值，标记为可疑
    "reconciliation_tolerance": 0.05,  # 5%

    # 可再生占比异常阈值：当占比超过 100% 或小于 0% 时，直接严重警告
    "renewable_min": 0.0,
    "renewable_max": 100.0,

    # 排放强度同比恶化的容忍度（正数表示恶化）
    "intensity_worsening_tolerance": 0.02,  # 2%
}


class FinancialReconciliationModule(BaseModule):
    @property
    def name(self) -> str:
        return "财务数据一致性"

    @property
    def default_weight(self) -> float:
        return 0.35

    @property
    def required_fields(self) -> List[str]:
        # 至少需要 Scope1 和 Scope2 数据才能激活
        return ["scope1", "scope2"]

    def execute(self, evidence: EvidencePackage) -> ModuleResult:
        details = {}
        scores = []
        status = ModuleStatus.SUCCESS

        # ---------- 1. 检查 Scope1+Scope2 与合计的一致性 ----------
        if evidence.scope1_2_total is not None:
            expected = evidence.scope1 + evidence.scope2
            if expected == 0:
                rel_error = 0.0
            else:
                rel_error = abs(evidence.scope1_2_total - expected) / expected
            details["reconciliation_error"] = round(rel_error, 4)

            if rel_error > CONFIG["reconciliation_tolerance"]:
                # 误差大：得分低（可疑）
                score = -min(rel_error * 5, 1.0)  # 最大扣到 -1
                scores.append(score)
                details[
                    "reconciliation_issue"] = f"恒等式误差 {rel_error:.1%}，超出容忍度 {CONFIG['reconciliation_tolerance']:.0%}"
            else:
                scores.append(0.2)  # 少量加分
                details["reconciliation_issue"] = "恒等式通过"
        else:
            details["reconciliation_issue"] = "合计数据缺失，无法核验恒等式"

        # ---------- 2. 检查可再生占比的边界合法性 ----------
        if evidence.renewable_ratio is not None:
            if evidence.renewable_ratio < CONFIG["renewable_min"] or evidence.renewable_ratio > CONFIG["renewable_max"]:
                scores.append(-0.8)
                details["renewable_issue"] = f"可再生占比 {evidence.renewable_ratio:.1f}% 超出合理边界 [0%, 100%]"
            elif evidence.renewable_ratio > 90:
                scores.append(0.3)  # 高绿电占比，正能量
                details["renewable_issue"] = "可再生占比很高，积极信号"
            else:
                scores.append(0.0)
                details["renewable_issue"] = "可再生占比正常"
        else:
            details["renewable_issue"] = "可再生占比数据缺失"

        # ---------- 3. 排放强度同比趋势 (声称降低但强度上升) ----------
        if evidence.scope12_intensity is not None and evidence.total_energy_mwh is not None:
            # 这里简化：假设强度变化来自外部计算，或者由主程序预计算传入
            # 此处仅作占位逻辑
            pass

        # ---------- 4. 聚合最终得分 ----------
        if not scores:
            return ModuleResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                evidence="财务数据不足，无法评估",
                status=ModuleStatus.SKIPPED,
                details=details
            )

        avg_score = sum(scores) / len(scores)
        # 置信度：如果恒等式数据完整，置信度高，否则打折
        confidence = 0.8 if evidence.scope1_2_total is not None else 0.5

        evidence_text = f"核验 {len(scores)} 项财务指标，综合得分 {avg_score:.2f}"
        if avg_score < -0.3:
            evidence_text += "，发现显著的数据不一致异常！"
        else:
            evidence_text += "，未发现重大异常。"

        return ModuleResult(
            module_name=self.name,
            score=round(avg_score, 4),
            confidence=round(confidence, 4),
            evidence=evidence_text,
            status=status,
            details=details
        )