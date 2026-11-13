# -*- coding: utf-8 -*-
"""
greenwash_scorer.py
====================
SURF 2026 T2/T5 | ESG 真实性交叉验证模型
多维度加权可信度打分系统（"绿洗嫌疑度"评分器）

设计思路（对应之前讨论的四层管道）：
    声称输入 → 维度证据提取(z-score) → 单维度一致性打分(s_p) → 加权融合(Credibility) → 阈值分级

这是一个「可解释规则模型」，不是端到端机器学习分类器 —— 因为我们没有
"绿洗/未绿洗"的标签数据，规则模型的每一步都可以拆开检查、和领域专家权重对齐，
也方便向评委解释每个判断是怎么来的。
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ============================================================
# 第一层：声称结构化 (Claim)
# ============================================================

class ClaimCategory(str, Enum):
    """对应截图里 Group2 造假的四类披露声称"""
    OPERATION = "产能运营活动"       # 例：生产活动持续提升、产能利用率提高
    EXPANSION = "扩建新增设施"       # 例：新增厂房/产线/仓储
    GREEN = "光伏绿化"               # 例：屋顶光伏比例提高、绿化覆盖提升
    CONTRADICTION = "矛盾类"         # 例：产能提升 vs 排放大幅下降


class ClaimDirection(int, Enum):
    INCREASE = 1
    DECREASE = -1
    STABLE = 0


@dataclass
class Claim:
    """一条待验证的 ESG 披露声称"""
    claim_id: str
    category: ClaimCategory
    direction: ClaimDirection
    aoi_id: str                 # 对应的工厂/AOI 标识，用于关联卫星 proxy 数据
    t0_label: str = "T0"
    t1_label: str = "T1"
    text: str = ""               # 原始披露文本，仅作记录，不参与计算
    # 矛盾类专用：关联的另一条声称 id（例如"排放下降"声称的 id）
    paired_claim_id: Optional[str] = None


@dataclass
class ProxyObservation:
    """
    某个 AOI 在 T0/T1 两期的某个 proxy 原始观测值。
    quality 是数据质量置信度 in [0,1]：
      - 云覆盖高、NaN、GT缺失 等情况应该把 quality 设低（甚至0），
        这样该维度会在加权融合时自动"退出"，而不是被强行凑权重。
    """
    proxy_name: str          # "nightlight" / "lst" / "ndvi" / "builtup_area"
    value_t0: float
    value_t1: float
    quality: float = 1.0

    @property
    def delta(self) -> float:
        if any(v is None or (isinstance(v, float) and math.isnan(v))
               for v in (self.value_t0, self.value_t1)):
            return float("nan")
        return self.value_t1 - self.value_t0


# ============================================================
# 专家权重表 —— 体现"哪个proxy对哪类声称更靠谱"的物理先验
# 可以和 T5 / 项目负责人一起校准这些数字，这里给的是讨论中的初始值
# ============================================================

EXPERT_WEIGHTS: Dict[ClaimCategory, Dict[str, float]] = {
    ClaimCategory.OPERATION: {"nightlight": 0.45, "lst": 0.35, "builtup_area": 0.20},
    ClaimCategory.EXPANSION: {"nightlight": 0.25, "lst": 0.15, "builtup_area": 0.60},
    ClaimCategory.GREEN:     {"ndvi": 0.30, "builtup_area": 0.30},
}

# 弱验证维度的整体可信度上限：光伏/绿化在10m分辨率下证据本身就弱，
# 不能让弱证据推出高置信度的"强指控"（宁可"证据不足"也不能"错误指控"）
CATEGORY_CREDIBILITY_CAP: Dict[ClaimCategory, float] = {
    ClaimCategory.GREEN: 0.5,
}


# ============================================================
# 第二层 + 第三层：BaselineStats（z-score归一化） + 单维度打分
# ============================================================

class BaselineStats:
    """
    维护"同类型AOI在该proxy上的变化量(Δ)分布"，用于把原始变化量转成z-score。
    不用固定阈值的原因：不同工厂类型/规模的基线波动差异很大，
    用同类样本的分布做归一化更公平。
    """

    def __init__(self):
        self._deltas: Dict[str, List[float]] = {}   # proxy_name -> list of Δ (拟合用)
        self._mu: Dict[str, float] = {}
        self._sigma: Dict[str, float] = {}
        self._k: Dict[str, float] = {}               # 缩放常数：|Δz| 的95分位数

    def fit(self, proxy_name: str, deltas: List[float]):
        """用历史/同类AOI的变化量样本拟合均值、标准差和缩放常数k"""
        arr = np.array([d for d in deltas if d is not None and not math.isnan(d)])
        if len(arr) < 2:
            raise ValueError(f"[{proxy_name}] 拟合样本太少（{len(arr)}个），无法计算稳定的均值/标准差")
        mu, sigma = float(np.mean(arr)), float(np.std(arr))
        sigma = sigma if sigma > 1e-8 else 1e-8   # 防止除0
        z = (arr - mu) / sigma
        k = float(np.percentile(np.abs(z), 95)) or 1.0
        self._deltas[proxy_name] = list(arr)
        self._mu[proxy_name] = mu
        self._sigma[proxy_name] = sigma
        self._k[proxy_name] = k

    def zscore(self, proxy_name: str, delta: float) -> float:
        if proxy_name not in self._mu:
            raise KeyError(f"proxy '{proxy_name}' 还没有 fit 过基线分布，请先调用 fit()")
        if delta is None or math.isnan(delta):
            return float("nan")
        return (delta - self._mu[proxy_name]) / self._sigma[proxy_name]

    def scale_k(self, proxy_name: str) -> float:
        return self._k.get(proxy_name, 2.0)   # 没fit过就退回默认k=2.0


def single_dimension_score(direction: ClaimDirection, z: float, k: float) -> float:
    """
    s_p = clip( sign(声称方向) * z / k , -1, 1 )
      +1  → proxy 强烈支持声称
       0  → 证据不足以判断（噪声区间内）
      -1  → proxy 明显和声称矛盾（真正的"绿洗信号"）
    STABLE(方向=0) 的声称：只看变化幅度是否显著偏离0，偏离越大越"不支持稳定"的声称。
    """
    if math.isnan(z):
        return float("nan")   # 数据缺失，交给上层用 quality=0 处理
    if direction == ClaimDirection.STABLE:
        # "保持稳定"的声称：|z|越大越矛盾
        return float(np.clip(1 - abs(z) / k, -1, 1))
    return float(np.clip(direction.value * z / k, -1, 1))


# ============================================================
# 第四层：加权融合 —— 可信度 Credibility
# ============================================================

@dataclass
class DimensionResult:
    proxy_name: str
    delta: float
    z: float
    s_p: float
    weight: float
    quality: float

    @property
    def effective_weight(self) -> float:
        return self.weight * self.quality

    @property
    def vote(self) -> str:
        if math.isnan(self.s_p):
            return "数据缺失"
        if self.s_p > 0.3:
            return "支持声称"
        if self.s_p < -0.3:
            return "矛盾"
        return "中性/证据不足"


@dataclass
class ScoreResult:
    claim: Claim
    dimensions: List[DimensionResult]
    credibility: float          # in [-1, 1]，越低代表越可疑
    suspicion: float            # in [0, 1] = (1 - credibility) / 2
    capped: bool = False        # 是否因为弱验证类别被截断可信度幅度


def compute_credibility(claim: Claim,
                         observations: Dict[str, ProxyObservation],
                         baseline: BaselineStats) -> ScoreResult:
    """
    对一条声称，融合它类别下所有相关proxy的单维度打分，得到加权可信度。
    观测数据缺失/quality=0的维度会自动被排除出加权和的分母，不会拉低整体权重占比。
    """
    weights = EXPERT_WEIGHTS[claim.category]
    dims: List[DimensionResult] = []

    for proxy_name, w in weights.items():
        obs = observations.get(proxy_name)
        if obs is None:
            # 完全没提供这个proxy的数据 —— 记为质量0，参与展示但不参与计算
            dims.append(DimensionResult(proxy_name, float("nan"), float("nan"),
                                         float("nan"), w, 0.0))
            continue
        z = baseline.zscore(proxy_name, obs.delta)
        k = baseline.scale_k(proxy_name)
        s_p = single_dimension_score(claim.direction, z, k)
        dims.append(DimensionResult(proxy_name, obs.delta, z, s_p, w, obs.quality))

    numer = sum(d.effective_weight * d.s_p for d in dims if not math.isnan(d.s_p))
    denom = sum(d.effective_weight for d in dims if not math.isnan(d.s_p))

    if denom < 1e-8:
        # 所有维度都缺失/低质量 —— 无法判断，credibility记为0（中性），并标记出来
        credibility = 0.0
    else:
        credibility = numer / denom

    capped = False
    cap = CATEGORY_CREDIBILITY_CAP.get(claim.category)
    if cap is not None and abs(credibility) > cap:
        credibility = math.copysign(cap, credibility)
        capped = True

    suspicion = (1 - credibility) / 2
    return ScoreResult(claim=claim, dimensions=dims,
                        credibility=credibility, suspicion=suspicion, capped=capped)


# ============================================================
# 矛盾类（第4类）：声称 vs 声称 的逻辑一致性检查
# proxy在这里是"仲裁票"：先看两条声称本身逻辑是否自洽，
# 如果本身就矛盾，直接标记；否则用proxy判断哪条更可能是真的。
# ============================================================

# 简化的物理先验：某类声称的"增强"是否应该伴随另一类声称的"增强"
# True 表示"两者理应同向"，False 表示"理应反向"
LOGICAL_CONSISTENCY_RULES: Dict[Tuple[str, str], bool] = {
    ("产能", "排放"): True,     # 产能提升通常伴随排放/能耗上升（除非有独立能效证据）
    ("产能", "能效"): False,    # 产能提升本身不必然伴随能效改善
}


def check_contradiction(claim_a_topic: str, claim_a_dir: ClaimDirection,
                         claim_b_topic: str, claim_b_dir: ClaimDirection,
                         arbiter_score: Optional[ScoreResult] = None) -> Dict:
    """
    检查一对声称 (如 "产能提升" vs "排放下降") 是否自相矛盾。
    若 arbiter_score 提供了（比如用产能声称算出的 credibility），
    则用它辅助判断"哪条更可能是真的"。
    """
    key = (claim_a_topic, claim_b_topic)
    reversed_key = (claim_b_topic, claim_a_topic)
    expect_same_dir = LOGICAL_CONSISTENCY_RULES.get(key)
    if expect_same_dir is None:
        expect_same_dir = LOGICAL_CONSISTENCY_RULES.get(reversed_key)
    if expect_same_dir is None:
        return {"flag": "无预设规则，需人工定义", "detail": key}

    same_dir = (claim_a_dir.value * claim_b_dir.value) > 0
    is_logically_contradictory = (same_dir != expect_same_dir) and (claim_a_dir.value != 0 and claim_b_dir.value != 0)

    result = {
        "claim_pair": (claim_a_topic, claim_b_topic),
        "expect_same_direction": expect_same_dir,
        "observed_same_direction": same_dir,
        "logically_contradictory": is_logically_contradictory,
    }
    if is_logically_contradictory and arbiter_score is not None:
        result["arbiter_credibility_on_claim_a"] = arbiter_score.credibility
        result["note"] = ("两条声称本身逻辑矛盾；已用proxy仲裁其中一条声称的可信度，"
                           "可信度越低代表这条声称越可能是被粉饰的那一方。")
    return result


# ============================================================
# 第五层：阈值分级 —— 用「基线对照组」校准，而不是拍脑袋定死数字
# ============================================================

@dataclass
class ThresholdBands:
    mean: float
    std: float

    @property
    def low_cut(self) -> float:
        return self.mean + 1 * self.std

    @property
    def high_cut(self) -> float:
        return self.mean + 2 * self.std

    def classify(self, suspicion: float) -> str:
        if suspicion < self.low_cut:
            return "低嫌疑（证据支持或中性）"
        elif suspicion < self.high_cut:
            return "中度嫌疑（建议人工复核）"
        else:
            return "高度嫌疑（重点标记）"


def calibrate_thresholds(baseline_suspicion_scores: List[float]) -> ThresholdBands:
    """
    用一批"已知为正常/已确认真实"的对照组声称的嫌疑度分布，
    校准出 mean/std，再用 mean+1σ / mean+2σ 分级。
    没有标签数据时，可以先用人工构造的几条"明显真实"的声称作为对照组种子。
    """
    arr = np.array(baseline_suspicion_scores)
    return ThresholdBands(mean=float(np.mean(arr)), std=float(np.std(arr)) or 1e-6)


# ============================================================
# 可解释输出
# ============================================================

def explain(result: ScoreResult, bands: Optional[ThresholdBands] = None) -> str:
    lines = [f"声称: \"{result.claim.text or result.claim.claim_id}\" "
             f"[{result.claim.category.value} / {result.claim.direction.name}]"]
    for d in result.dimensions:
        z_str = f"z={d.z:.2f}" if not math.isnan(d.z) else "z=N/A"
        s_str = f"s_p={d.s_p:.2f}" if not math.isnan(d.s_p) else "s_p=N/A"
        lines.append(f"  ├─ {d.proxy_name}: {z_str}, {s_str} → {d.vote} "
                      f"[权重{d.weight:.2f}, 数据质量{d.quality:.2f}]")
    cap_note = "（已按弱验证类别截断幅度）" if result.capped else ""
    lines.append(f"  └─ 综合可信度 = {result.credibility:.2f}{cap_note} "
                 f"→ 嫌疑度 = {result.suspicion:.2f}")
    if bands is not None:
        lines.append(f"     分级: 【{bands.classify(result.suspicion)}】"
                      f" (基线均值{bands.mean:.2f}, low_cut={bands.low_cut:.2f}, high_cut={bands.high_cut:.2f})")
    return "\n".join(lines)


def batch_score_to_dataframe(results: List[ScoreResult],
                              bands: Optional[ThresholdBands] = None) -> pd.DataFrame:
    """把一批打分结果整理成表格，方便导出Excel给T5/T6"""
    rows = []
    for r in results:
        row = {
            "claim_id": r.claim.claim_id,
            "category": r.claim.category.value,
            "direction": r.claim.direction.name,
            "aoi_id": r.claim.aoi_id,
            "credibility": round(r.credibility, 3),
            "suspicion": round(r.suspicion, 3),
            "capped": r.capped,
        }
        if bands is not None:
            row["level"] = bands.classify(r.suspicion)
        for d in r.dimensions:
            row[f"{d.proxy_name}_delta"] = d.delta
            row[f"{d.proxy_name}_z"] = d.z
            row[f"{d.proxy_name}_s_p"] = d.s_p
            row[f"{d.proxy_name}_vote"] = d.vote
        rows.append(row)
    return pd.DataFrame(rows)
