# modules/module_remote_sensing.py
# 遥感物理一致性模块：夜光 / 温度 / NDVI / 建成区

import sys
import os

# 添加父目录到路径，以便导入 base 模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from base_models import BaseModule, EvidencePackage, ModuleResult, ModuleStatus

# ================= 该模块的硬编码配置（仅该组员维护） =================
CONFIG = {
    # 各代理变量的权重 (内部权重，用于计算该模块最终得分)
    "weight_nightlight": 0.40,
    "weight_lst": 0.25,
    "weight_ndvi": 0.15,
    "weight_builtup": 0.20,

    # 数据质量阈值：当变化量超过该值时，认为有显著变化
    "min_nightlight_delta": 0.3,
    "min_lst_delta": 0.5,
    "min_ndvi_delta": 0.02,
    "min_builtup_delta": 50,
}


class RemoteSensingModule(BaseModule):
    @property
    def name(self) -> str:
        return "遥感物理一致性"

    @property
    def default_weight(self) -> float:
        # 该模块在总系统中的推荐权重（占比）
        return 0.40

    @property
    def required_fields(self) -> List[str]:
        # 只要存在至少 2 个遥感数据，该模块即可激活（这里要求全部存在，可放宽）
        return ["nightlight_delta", "lst_delta", "ndvi_delta", "builtup_delta"]

    def execute(self, evidence: EvidencePackage) -> ModuleResult:
        """
        核心逻辑：比较各遥感代理变化方向与声称方向是否一致。
        得分区间：-1 ~ 1
        """
        direction = evidence.claim_direction  # 1: 增长, -1: 下降

        # ---------- 1. 计算各维度得分 ----------
        def calc_dimension_score(delta: float, min_delta: float, weight: float) -> dict:
            if delta is None:
                return {"score": 0.0, "weight": 0.0, "active": False}

            # 如果变化量绝对值小于阈值，视为无变化 -> 中性 (0分)
            if abs(delta) < min_delta:
                return {"score": 0.0, "weight": 0.0, "active": False}

            # 判断方向是否匹配：增长声称(1) 应该 delta > 0；下降声称(-1) 应该 delta < 0
            if (direction == 1 and delta > 0) or (direction == -1 and delta < 0) or direction == 0:
                # 方向一致：得分 + (归一化幅度)
                normalized_magnitude = min(abs(delta) / (min_delta * 3), 1.0)
                score = normalized_magnitude
            else:
                # 方向相反：得分 - (归一化幅度)
                normalized_magnitude = min(abs(delta) / (min_delta * 3), 1.0)
                score = -normalized_magnitude

            return {"score": score, "weight": weight, "active": True}

        dims = {
            "nightlight": calc_dimension_score(
                evidence.nightlight_delta,
                CONFIG["min_nightlight_delta"],
                CONFIG["weight_nightlight"]
            ),
            "lst": calc_dimension_score(
                evidence.lst_delta,
                CONFIG["min_lst_delta"],
                CONFIG["weight_lst"]
            ),
            "ndvi": calc_dimension_score(
                evidence.ndvi_delta,
                CONFIG["min_ndvi_delta"],
                CONFIG["weight_ndvi"]
            ),
            "builtup": calc_dimension_score(
                evidence.builtup_delta,
                CONFIG["min_builtup_delta"],
                CONFIG["weight_builtup"]
            ),
        }

        # ---------- 2. 加权聚合 ----------
        total_weight = 0.0
        weighted_sum = 0.0
        active_count = 0
        details = {}

        for name, result in dims.items():
            if result["active"]:
                total_weight += result["weight"]
                weighted_sum += result["score"] * result["weight"]
                active_count += 1
                details[f"{name}_score"] = result["score"]
                details[f"{name}_weight"] = result["weight"]

        if total_weight == 0:
            return ModuleResult(
                module_name=self.name,
                score=0.0,
                confidence=0.0,
                evidence="无可用的遥感代理变化数据（变化量均低于阈值）",
                status=ModuleStatus.SKIPPED,
                details=details
            )

        final_score = weighted_sum / total_weight  # -1 ~ 1

        # 置信度：根据有效维度数量折算
        confidence = min(0.5 + 0.1 * active_count, 1.0)

        # 生成证据描述
        evidence_text = f"基于 {active_count} 个遥感维度，综合得分 {final_score:.2f}"
        if final_score > 0.3:
            evidence_text += "，物理趋势与声称方向一致。"
        elif final_score < -0.3:
            evidence_text += "，物理趋势与声称方向矛盾！"
        else:
            evidence_text += "，物理证据不足或方向不明确。"

        return ModuleResult(
            module_name=self.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            evidence=evidence_text,
            status=ModuleStatus.SUCCESS,
            details=details
        )