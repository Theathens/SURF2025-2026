# base_models.py
# 所有模块共用的数据结构和接口规范

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from abc import ABC, abstractmethod


# ---------- 状态与结果枚举 ----------
class ModuleStatus(str, Enum):
    SUCCESS = "成功执行"
    SKIPPED = "数据缺失，跳过"
    ERROR = "执行错误"


# ---------- 统一证据包（所有可能的数据字段） ----------
@dataclass
class EvidencePackage:
    """各维度的原始数据统一存放处。缺省值为 None，表示该维度未提供数据。"""

    # ===== 遥感维度 (来自 Zonal_stats2) =====
    nightlight_delta: Optional[float] = None      # 夜光变化量 (T1 - T0)
    lst_delta: Optional[float] = None             # 地表温度变化量
    ndvi_delta: Optional[float] = None            # 植被指数变化量
    builtup_delta: Optional[float] = None         # 建成区/工业像素变化量

    # ===== 财务维度 (来自 ESG 年报) =====
    scope1: Optional[float] = None                # Scope 1 排放量 (tCO2e)
    scope2: Optional[float] = None                # Scope 2 排放量 (tCO2e)
    scope1_2_total: Optional[float] = None        # 披露的 Scope1+2 合计
    renewable_ratio: Optional[float] = None       # 可再生能源占比 (%)
    total_energy_mwh: Optional[float] = None      # 总能耗 (MWh)
    scope12_intensity: Optional[float] = None     # 排放强度 (tCO2e/MWh)

    # ===== 文本维度 (来自文本侦测) =====
    ttr_score: Optional[float] = None             # 词汇多样性 (0~1, 越低越可疑)
    ngram_coverage: Optional[float] = None        # 3-gram 模板覆盖率 (0~1, 越高越可疑)
    benford_error: Optional[float] = None         # 本福特定律拟合误差 (0~1, 越高越可疑)

    # ===== 声称元信息 (用于判断方向/类别) =====
    claim_direction: int = 1                      # 声称方向: 1=增长, -1=下降, 0=稳定
    claim_category: str = "OPERATION"            # 声称类别: "OPERATION", "GREEN", "EXPANSION"


# ---------- 模块返回的统一结果 ----------
@dataclass
class ModuleResult:
    module_name: str
    score: float               # -1.0 (强烈可疑) 到 +1.0 (高度可信)
    confidence: float          # 0.0 ~ 1.0 (该结果的可信度)
    evidence: str              # 简短证据描述
    status: ModuleStatus
    details: Dict[str, Any] = field(default_factory=dict)  # 存放详细中间值


# ---------- 模块基类（所有子模块必须继承） ----------
class BaseModule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """模块显示名称"""
        pass

    @property
    @abstractmethod
    def default_weight(self) -> float:
        """该模块在完全可用时的推荐权重 (主程序会动态归一化)"""
        pass

    @property
    @abstractmethod
    def required_fields(self) -> List[str]:
        """必须存在哪些字段才能执行，否则直接跳过"""
        pass

    @abstractmethod
    def execute(self, evidence: EvidencePackage) -> ModuleResult:
        """核心执行逻辑"""
        pass