# modules/module_text_semantic.py
# 文本语义真实性模块：TTR / 3-gram覆盖率 / Benford检验

import sys
import os
PROJECT_ROOT = r"D:\Pycharm Projects\torch\greenwash_detection_orchestra"

sys.path.append(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from typing import List
from base_models import BaseModule, EvidencePackage, ModuleResult, ModuleStatus

# ================= 该模块的硬编码配置 =================
CONFIG = {
    # TTR 阈值：低于此值认为词汇贫瘠（可能是AI生成）
    "ttr_suspicious_threshold": 0.50,
    "ttr_high_confidence_threshold": 0.70,

    # 3-gram 模板覆盖率阈值：高于此值认为高度模板化（AI/套话）
    "ngram_suspicious_threshold": 0.60,

    # Benford 误差阈值：高于此值认为数字可能被篡改
    "benford_suspicious_threshold": 0.10,
}


class TextSemanticModule(BaseModule):
    @property
    def name(self) -> str:
        return "文本语义真实性"

    @property
    def default_weight(self) -> float:
        return 0.25

    @property
    def required_fields(self) -> List[str]:
        return ["ttr_score", "ngram_coverage"]

    def execute(self, evidence: EvidencePackage) -> ModuleResult:
        details = {}
        scores = []
        active_items = 0

        # ---------- 1. TTR (词汇多样性) ----------
        if evidence.ttr_score is not None:
            active_items += 1
            ttr = evidence.ttr_score
            details["ttr_score"] = ttr
            if ttr < CONFIG["ttr_suspicious_threshold"]:
                # 越低越可疑（词汇贫瘠）
                score = - (1 - ttr * 2)  # 0.4 -> -0.2, 0.2 -> -0.6
                scores.append(score)
                details["ttr_issue"] = f"TTR={ttr:.2f}，低于阈值 {CONFIG['ttr_suspicious_threshold']:.2f}，疑似AI生成"
            elif ttr > CONFIG["ttr_high_confidence_threshold"]:
                scores.append(0.4)
                details["ttr_issue"] = f"TTR={ttr:.2f}，词汇丰富，可信度高"
            else:
                scores.append(0.0)
                details["ttr_issue"] = f"TTR={ttr:.2f}，处于正常区间"

        # ---------- 2. 3-gram 模板覆盖率 ----------
        if evidence.ngram_coverage is not None:
            active_items += 1
            ngram = evidence.ngram_coverage
            details["ngram_coverage"] = ngram
            if ngram > CONFIG["ngram_suspicious_threshold"]:
                # 越高越可疑（套话模板）
                score = -min((ngram - 0.4) * 2, 1.0)
                scores.append(score)
                details[
                    "ngram_issue"] = f"3-gram覆盖率={ngram:.2f}，高于阈值 {CONFIG['ngram_suspicious_threshold']:.2f}，内容模板化严重"
            else:
                scores.append(0.2)
                details["ngram_issue"] = f"3-gram覆盖率={ngram:.2f}，处于正常范围"

        # ---------- 3. Benford 检验 ----------
        if evidence.benford_error is not None:
            active_items += 1
            benford = evidence.benford_error
            details["benford_error"] = benford
            if benford > CONFIG["benford_suspicious_threshold"]:
                score = -min(benford * 3, 1.0)
                scores.append(score)
                details[
                    "benford_issue"] = f"本福特误差={benford:.2f}，超过阈值 {CONFIG['benford_suspicious_threshold']:.2f}，数字可能被修饰"
            else:
                scores.append(0.1)
                details["benford_issue"] = "本福特检验通过"

        # ---------- 4. 聚合 ----------
        if active_items == 0 or not scores:
            return ModuleResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                evidence="文本维度数据缺失",
                status=ModuleStatus.SKIPPED,
                details=details
            )

        avg_score = sum(scores) / len(scores)
        confidence = min(0.5 + 0.15 * active_items, 1.0)

        evidence_text = f"基于 {active_items} 项文本特征，综合得分 {avg_score:.2f}"
        if avg_score < -0.3:
            evidence_text += "，文本呈现明显AI/模板化特征，语义可信度低。"
        else:
            evidence_text += "，文本特征正常。"

        return ModuleResult(
            module_name=self.name,
            score=round(avg_score, 4),
            confidence=round(confidence, 4),
            evidence=evidence_text,
            status=ModuleStatus.SUCCESS,
            details=details
        )