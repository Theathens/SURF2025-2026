# 03_run.py
# 主运行入口：实例化数据 -> 调用调度器 -> 打印最终结果

from base_models import EvidencePackage
from orchestrator import GreenwashOrchestrator
import json


# ================= 用户硬编码配置区（所有手动输入都集中在这里） =================
class RunConfig:
    # ----- 测试数据1：遥感维度 + 财务维度 + 文本维度 全量输入 -----
    # 模拟一个 "产能提升" 声称 (OPERATION, direction=1)
    TEST_EVIDENCE = EvidencePackage(
        # 遥感数据 (来自 Zonal_stats2)
        nightlight_delta=-0.8,      # 夜光反而下降！可疑
        lst_delta=0.3,
        ndvi_delta=-0.01,
        builtup_delta=80,

        # 财务数据 (来自 ESG 年报)
        scope1=550932.0,
        scope2=7511038.0,
        scope1_2_total=8061970.0,   # 实际合计 8061970，与披露一致 -> 恒等式通过
        renewable_ratio=1.85,
        total_energy_mwh=10618556.88,
        scope12_intensity=0.7592,

        # 文本数据 (来自文本侦测)
        ttr_score=0.42,             # 低于0.5，词汇贫瘠，可疑
        ngram_coverage=0.70,        # 高于0.6，模板化严重，可疑

        # 声称元信息
        claim_direction=1,          # 声称产能提升
        claim_category="OPERATION"
    )

    # 如需测试其他案例，可在此新增多个测试用例，或从外部文件加载


# ================= 主程序 =================
if __name__ == "__main__":

    print("=" * 70)
    print("绿洗检测中心 (Greenwash Detection Hub) 启动")
    print("=" * 70)

    # 1. 实例化调度器
    orchestrator = GreenwashOrchestrator()

    # 2. 传入证据，执行评估
    evidence = RunConfig.TEST_EVIDENCE
    result = orchestrator.evaluate(evidence)

    # 3. 打印最终结果
    print("\n" + "=" * 70)
    print("【最终评估结果】")
    print("=" * 70)
    print(f"  最终可信度 (Credibility):    {result['final_credibility']:.4f}  (范围 -1 ~ 1, 负值表示可疑)")
    print(f"  嫌疑度 (Suspicion Score):    {result['suspicion_score']:.4f}  (范围 0 ~ 1, 越高越可疑)")
    print(f"  风险等级:                     {result['risk_level']}")
    print(f"  激活模块:                     {', '.join(result['active_modules'])}")
    print(f"  状态:                         {result['status']}")

    if "message" in result:
        print(f"  备注:                         {result['message']}")

    print("\n--- 各模块明细 ---")
    for mod_res in result.get("module_results", []):
        print(f"  [{mod_res.module_name}]")
        print(f"      得分: {mod_res.score:.4f} | 置信度: {mod_res.confidence:.4f}")
        print(f"      证据: {mod_res.evidence}")
        if mod_res.details:
            print(f"      详情: {json.dumps(mod_res.details, ensure_ascii=False, indent=6)}")
        print()