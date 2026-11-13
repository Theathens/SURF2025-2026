# orchestrator.py
# 主调度器：动态检测可用数据 -> 调用匹配的模块 -> 加权融合

import sys
from typing import List, Dict, Any
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_models import EvidencePackage, BaseModule, ModuleStatus, ModuleResult

# 导入所有已开发的模块
# 每个新模块在这里新增一行导入，并加入 MODULES_LIST
from modules.module_remote_sensing import RemoteSensingModule
from modules.module_financial import FinancialReconciliationModule
from modules.module_text_semantic import TextSemanticModule


# ================= 全局配置（硬编码） =================
class OrchestratorConfig:
    # 当某个模块执行报错时，是否让其静默跳过 (True=跳过继续, False=抛异常)
    SKIP_ON_MODULE_ERROR = True

    # 最终结果输出时，是否包含各模块的详细明细
    INCLUDE_DETAILS_IN_OUTPUT = True


# ================= 核心调度器 =================
class GreenwashOrchestrator:
    def __init__(self):
        # 所有已注册的模块列表（按需新增）
        self.modules: List[BaseModule] = [
            RemoteSensingModule(),
            FinancialReconciliationModule(),
            TextSemanticModule(),
            # TODO: 后续新增模块只需在这里追加一行
        ]

    def evaluate(self, evidence: EvidencePackage) -> Dict[str, Any]:
        """
        主入口：传入证据包，返回融合后的可信度与疑点详情
        """
        active_modules = []
        total_weight = 0.0

        # ---------- 阶段1：可用性检测 ----------
        for mod in self.modules:
            # 检查该模块的所有必需字段是否都有数据
            required_ok = True
            for field in mod.required_fields:
                if getattr(evidence, field, None) is None:
                    required_ok = False
                    break

            if required_ok:
                active_modules.append(mod)
                total_weight += mod.default_weight
            else:
                print(f"[调度] 模块 '{mod.name}' 所需数据缺失，已跳过。")

        # ---------- 阶段2：无可执行模块 ----------
        if not active_modules:
            return {
                "status": "无可用数据",
                "final_credibility": 0.0,
                "suspicion_score": 0.5,
                "active_modules": [],
                "module_results": [],
                "message": "所有维度数据均缺失，无法计算可信度。"
            }

        # ---------- 阶段3：执行各模块并动态加权 ----------
        final_score = 0.0
        weighted_confidence_sum = 0.0
        module_results = []

        for mod in active_modules:
            try:
                result = mod.execute(evidence)
                module_results.append(result)

                # 动态归一化权重: 该模块权重 / 所有可用模块权重之和
                adjusted_weight = mod.default_weight / total_weight
                final_score += result.score * adjusted_weight
                weighted_confidence_sum += result.confidence * adjusted_weight

                print(f"[调度] 模块 '{mod.name}' 执行成功，得分: {result.score:.3f}，权重占比: {adjusted_weight:.2%}")

            except Exception as e:
                if OrchestratorConfig.SKIP_ON_MODULE_ERROR:
                    print(f"[调度] 模块 '{mod.name}' 执行异常: {e}，已跳过。")
                    continue
                else:
                    raise e

        # ---------- 阶段4：计算最终可信度 ----------
        # 最终可信度 = 加权平均得分 × 加权平均置信度
        avg_confidence = weighted_confidence_sum  # 因为权重已归一化，直接等于加权平均置信度
        final_credibility = final_score * avg_confidence

        # 嫌疑度 = (1 - 可信度) / 2 ，转为 0~1，越高越可疑
        suspicion = (1 - final_credibility) / 2

        # 根据最终可信度给出定性标签
        if final_credibility > 0.5:
            level = "低嫌疑（证据支持声称）"
        elif final_credibility > 0.0:
            level = "中度嫌疑（需人工复核）"
        else:
            level = "高嫌疑（重点关注）"

        # ---------- 阶段5：组装返回结果 ----------
        output = {
            "status": "成功",
            "final_credibility": round(final_credibility, 4),  # -1 ~ 1
            "suspicion_score": round(suspicion, 4),  # 0 ~ 1
            "risk_level": level,
            "active_modules": [m.name for m in active_modules],
            "module_results": module_results,
        }

        if OrchestratorConfig.INCLUDE_DETAILS_IN_OUTPUT:
            output["message"] = f"共激活 {len(active_modules)} 个维度，综合置信度加权系数: {avg_confidence:.3f}"

        return output