"""
Skill-MAS Evaluation Metrics Extraction Module
===============================================
从 Skill-MAS 官方代码库 (https://github.com/linhh29/Skill-MAS) 提取的全部评测指标计算代码。

指标分类（7类，对应论文 22 个评测维度）:
  A. 主性能指标 (Main Performance)          — assemble_select.py
  B. 分布统计指标 (Distributional Stats)     — elbow_selection.py
  C. 选择性反思指标 (Selective Reflection)    — elbow_selection.py + contrastive_reflect.py
  D. 迁移性指标 (Transferability)           — contrastive_reflect.py + schemas.py
  E. 成本指标 (Cost Metrics)                 — llm_cost.py
  F. 演化追踪指标 (Evolution Tracking)       — bank_optimizer.py + assemble_select.py
  G. 各Benchmark评分 (Per-Benchmark Scoring)  — dataset/*/score.py + agent_metrics.py

源文件映射:
  evolution/elbow_selection.py          → B, C 类核心
  evolution/assemble_select.py           → A, F 类核心
  evolution/contrastive_reflect.py      → C, D 类
  evolution/bank_optimizer.py           → F 类
  evolution/schemas.py                 → 数据结构定义
  utils/llm_cost.py                     → E 类核心
  dataset/hlemath/score.py              → G-hlemath
  dataset/BrowseComp-Plus/score.py      → G-bcp
  dataset/deep_research_bench/utils/score_calculator.py → G-drb
  dataset/vitabench/src/vita/metrics/agent_metrics.py  → G-vitabench
"""

from __future__ import annotations

import math
import re
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from pathlib import Path


# ============================================================================
# 数据结构定义 (from evolution/schemas.py)
# ============================================================================

@dataclass
class PhaseSnapshot:
    """单个构建阶段的快照 (from schemas.py)."""
    phase: str
    instruction: str = ""
    output_preview: str = ""


@dataclass
class TrajectoryRecord:
    """
    单条轨迹记录 (from schemas.py).
    
    每个任务在每个 round 跑 K 条轨迹, 每条产出一个 TrajectoryRecord.
    """
    schema: str
    bench_backend: str          # "vitabench" | "drb" | "hlemath" | "bcp"
    round_idx: int
    task_id: str
    trajectory_idx: int         # 0..K-1
    trajectory_tag: str        # e.g. "task_42_traj_00"
    score: float               # [0, 1]
    score_source: str          # "vitabench_nl_assertion_ratio" | "drb_race_per_task" | "hlemath_sympy" | "bcp_exact_match" | "bcp_llm_judge"
    log_path: str
    raw_result_path: str
    phase_snapshots: list[PhaseSnapshot] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["phase_snapshots"] = [asdict(x) for x in self.phase_snapshots]
        return out


@dataclass
class DomainPatch:
    """
    候选 Skill 补丁 (from schemas.py).
    
    对比反思阶段产出: 从高/低分组轨迹中提取的改进约束.
    """
    schema: str
    task_id: str
    phase: str                 # 目标阶段 ("任务分解" | "Agent工程" | "工作流编排")
    constraint: str            # 具体约束规则
    rationale: str             # 实现机制 + 预期影响
    source_gap: float          # high_score - low_score
    source_high_traj: str
    source_low_traj: str
    frequency: int = 1
    status: str = "candidate"  # "candidate" | "accepted" | "rejected"

    def key(self) -> tuple[str, str]:
        return (self.phase.strip().lower(), self.constraint.strip().lower())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# A. 主性能指标 (from evolution/assemble_select.py)
# ============================================================================

class MainPerformanceMetrics:
    """
    主性能指标计算器.
    
    来源: evolution/assemble_select.py
    """

    @staticmethod
    def compute_round_score(by_task: dict[str, list[TrajectoryRecord]]) -> float:
        """
        Round 级别主分数 (Avg.Perf).
        
        计算方式:
          1. 对每个 task, 取所有 K 条轨迹的平均分
          2. 对所有 task 的平均分再取算术平均
        
        公式:
          round_score = (1/|T|) * Σ_{t∈T} [ (1/K) * Σ_{k=1}^{K} score(t, k) ]
        
        Args:
            by_task: {task_id: [TrajectoryRecord, ...]}
        
        Returns:
            float: round 平均分, [0, 1]
        """
        vals: list[float] = []
        for rows in by_task.values():
            if not rows:
                continue
            vals.append(sum(float(r.score) for r in rows) / len(rows))
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    @staticmethod
    def per_task_mean_scores(by_task: dict[str, list[TrajectoryRecord]]) -> dict[str, float]:
        """
        每个任务的平均分 (Per-Benchmark Score).
        
        Returns:
            {task_id: mean_score}
        """
        out: dict[str, float] = {}
        for tid, rows in by_task.items():
            if not rows:
                out[tid] = 0.0
                continue
            out[tid] = sum(float(r.score) for r in rows) / len(rows)
        return out

    @staticmethod
    def per_task_all_scores(by_task: dict[str, list[TrajectoryRecord]]) -> dict[str, list[float]]:
        """
        每个任务的全部轨迹分数 (用于分布分析).
        
        Returns:
            {task_id: [score_k0, score_k1, ...]}
        """
        return {tid: [float(r.score) for r in rows] for tid, rows in by_task.items()}

    @staticmethod
    def select_best_round(
        rounds: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        选择最优 round (from finalize_best_round).
        
        选择规则 (三级 tie-break):
          1. 分数最高
          2. 同分 → 优先 skill 复杂度低
          3. 仍同分 → 优先 stability risk 低
        
        Args:
            rounds: [{"round_idx": int, "round_score": float, "skill_round_path": str}, ...]
        
        Returns:
            最优 round 的完整信息
        """
        if not rounds:
            return {
                "best_round_idx": 0,
                "best_round_score": 0.0,
                "skill_path": "",
            }

        def _complexity(skill_round_path: str) -> int:
            p = Path(skill_round_path or "")
            if not p.is_dir():
                return 10**9
            return 1 if (p / "SKILL.md").is_file() else 10**9

        def _rank_key(row: dict[str, Any]) -> tuple[float, int, int]:
            ridx = int(row.get("round_idx", 0))
            round_path = str(row.get("skill_round_path") or "")
            complexity = _complexity(round_path)
            return (
                float(row.get("round_score", 0.0)),
                -complexity,
                -ridx,
            )

        best = max(rounds, key=_rank_key)
        return {
            "best_round_idx": int(best.get("round_idx", 0)),
            "best_round_score": float(best.get("round_score", 0.0)),
            "skill_path": str(best.get("skill_path") or ""),
            "skill_round_path": str(best.get("skill_round_path") or ""),
        }


# ============================================================================
# B. 分布统计指标 (from evolution/elbow_selection.py)
# ============================================================================

class DistributionalMetrics:
    """
    分布统计指标: 多轨迹方差评估.
    
    来源: evolution/elbow_selection.py
    
    核心思想:
      - 对同一任务跑 K 条轨迹, 得到 K 个分数
      - uncertainty (std) = 分数分布的离散程度
      - difficulty (-mean) = 任务难度 (分数越低越难)
      - priority = 0.5 * (norm(uncertainty) + norm(difficulty))
    """

    @staticmethod
    def population_std(xs: list[float]) -> float:
        """
        总体标准差 (ddof=0), 匹配 numpy.std(xs, ddof=0).
        
        公式: σ = sqrt( (1/N) * Σ (x_i - μ)² )
        """
        n = len(xs)
        if n <= 1:
            return 0.0
        mean = sum(xs) / n
        return math.sqrt(sum((x - mean) ** 2 for x in xs) / n)

    @staticmethod
    def normalize_minmax_1d(values: list[float]) -> list[float]:
        """
        Min-Max 归一化 (跨任务, within round).
        
        公式: norm(x) = (x - min) / (max - min)
        如果 max - min < 1e-8, 返回全 0.5
        """
        if not values:
            return []
        min_s = min(values)
        max_s = max(values)
        if max_s - min_s < 1e-8:
            return [0.5] * len(values)
        return [(v - min_s) / (max_s - min_s) for v in values]

    @staticmethod
    def compute_priority_vectors(
        samples_scores: list[list[float]]
    ) -> dict[str, list[float]]:
        """
        计算 per-task 的 uncertainty / difficulty / priority 向量.
        
        这是 Skill-MAS 分布统计的核心函数.
        
        Args:
            samples_scores: [[score_k0, score_k1, ...], ...]  (n_tasks × K)
        
        Returns:
            {
                "uncertainties_raw": [std_0, std_1, ...],
                "difficulties_raw": [-mean_0, -mean_1, ...],
                "uncertainties_normalized": [...],
                "difficulties_normalized": [...],
                "priorities": [p_0, p_1, ...],
            }
        
        公式:
          uncertainty_i = std(scores_i)          # 轨迹分数的标准差
          difficulty_i  = -mean(scores_i)        # 负的均值 (分数越低→难度越大)
          u_norm        = minmax(uncertainties)  # 跨任务归一化
          d_norm        = minmax(difficulties)
          priority_i    = (u_norm_i + d_norm_i) / 2
        """
        uncertainties: list[float] = []
        difficulties: list[float] = []
        for row in samples_scores:
            if not row:
                uncertainties.append(0.0)
                difficulties.append(0.0)
                continue
            mean = sum(row) / len(row)
            difficulties.append(-mean)
            uncertainties.append(DistributionalMetrics.population_std(row))

        u_norm = DistributionalMetrics.normalize_minmax_1d(uncertainties)
        d_norm = DistributionalMetrics.normalize_minmax_1d(difficulties)
        priorities = [(u_norm[i] + d_norm[i]) / 2.0 for i in range(len(uncertainties))]
        return {
            "uncertainties_raw": uncertainties,
            "difficulties_raw": difficulties,
            "uncertainties_normalized": u_norm,
            "difficulties_normalized": d_norm,
            "priorities": priorities,
        }

    @staticmethod
    def compute_priority_scores(samples_scores: list[list[float]]) -> list[float]:
        """
        Per-task priority score in [0, 1].
        
        对应论文公式:
          p_i = (1/2) * (norm(u_i) + norm(d_i))
        
        其中:
          u_i = std(trajectory_scores_i)     # uncertainty
          d_i = -mean(trajectory_scores_i)   # difficulty
          norm = minmax_normalization_across_tasks
        """
        if not samples_scores:
            return []
        return DistributionalMetrics.compute_priority_vectors(samples_scores)["priorities"]


# ============================================================================
# C. 选择性反思指标 (from evolution/elbow_selection.py + contrastive_reflect.py)
# ============================================================================

class SelectiveReflectionMetrics:
    """
    选择性反思指标: 任务选择 + 肘部检测.
    
    来源: evolution/elbow_selection.py
    
    核心思想:
      - 不对所有任务做反思 (太贵), 而是按 priority 排序
      - 用 second finite difference 检测 priority 曲线的 "肘部"
      - 肘部之前 = 高 priority 任务, 需要反思
    """

    @staticmethod
    def adaptive_elbow_count(
        sorted_scores_desc: list[float],
        sensitivity: float = 1.0
    ) -> int:
        """
        二阶差分肘部检测: 估计降序排列中前多少个在高优先级区域.
        
        原理:
          diffs[i] = sorted[i] - sorted[i+1]            # 一阶差分
          second_diffs[i] = diffs[i] - diffs[i+1]       # 二阶差分
          elbow_idx = argmax(|second_diffs|) + 1        # 最大拐点
          count = int(elbow_idx * sensitivity)
        
        Args:
            sorted_scores_desc: 降序排列的 priority scores
            sensitivity: 灵敏度参数, >1 选更多, <1 选更少
        
        Returns:
            int: 选择的任务数量, [1, n]
        """
        scores = sorted_scores_desc
        n = len(scores)
        if n == 0:
            return 0
        if n <= 2:
            return n

        diffs = [scores[i] - scores[i + 1] for i in range(n - 1)]
        second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
        if not second_diffs:
            return n

        argmax_i = max(range(len(second_diffs)), key=lambda i: abs(second_diffs[i]))
        elbow_idx = argmax_i + 1
        count = int(elbow_idx * float(sensitivity))
        count = max(1, min(n, count))
        return count

    @staticmethod
    def second_diff_elbow_detail(
        sorted_scores_desc: list[float],
        sensitivity: float = 1.0
    ) -> dict:
        """
        肘部检测的详细中间结果 (用于日志和可视化).
        
        Returns:
            {
                "n": int,
                "diffs": [...],               # 一阶差分
                "second_diffs": [...],         # 二阶差分
                "second_diff_argmax_index": int,
                "elbow_idx_before_sensitivity": int,
                "sensitivity": float,
                "selected_count": int,
            }
        """
        scores = sorted_scores_desc
        n = len(scores)
        out: dict = {
            "n": n,
            "diffs": [],
            "second_diffs": [],
            "second_diff_argmax_index": None,
            "elbow_idx_before_sensitivity": None,
            "sensitivity": float(sensitivity),
            "selected_count": 0,
        }
        if n == 0:
            return out
        if n <= 2:
            out["selected_count"] = n
            return out

        diffs = [scores[i] - scores[i + 1] for i in range(n - 1)]
        second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
        out["diffs"] = diffs
        out["second_diffs"] = second_diffs

        if not second_diffs:
            out["selected_count"] = n
            return out

        argmax_i = max(range(len(second_diffs)), key=lambda i: abs(second_diffs[i]))
        elbow_idx = argmax_i + 1
        out["second_diff_argmax_index"] = int(argmax_i)
        out["elbow_idx_before_sensitivity"] = int(elbow_idx)
        count = int(elbow_idx * float(sensitivity))
        count = max(1, min(n, count))
        out["selected_count"] = int(count)
        return out

    @staticmethod
    def compute_reflection_task_selection(
        task_rows: list[tuple[str, list[float]]],
        max_reflection_cases: int,
        sensitivity: float = 1.0
    ) -> tuple[list[str], dict[str, Any]]:
        """
        完整的选择性反思任务选择 pipeline.
        
        步骤:
          1. 按 task_id 排序 (确保稳定)
          2. 计算 per-task uncertainty/difficulty/priority
          3. 按 priority 降序排列
          4. 用二阶差分肘部检测选择 top-K
          5. 用 max_reflection_cases 截断
        
        Args:
            task_rows: [(task_id, [score_k0, score_k1, ...]), ...]
            max_reflection_cases: 最大反思任务数
            sensitivity: 肘部检测灵敏度
        
        Returns:
            (selected_task_ids, full_report)
        """
        rows = sorted(task_rows, key=lambda kv: str(kv[0]))
        if not rows:
            return [], {
                "schema": "skill_mas_priority_selection_v1",
                "selection_mode": "second_diff_elbow",
                "tasks": [],
                "reflection_selected_task_ids": [],
                "reflection_selected_count": 0,
            }

        samples_scores = [s for _, s in rows]
        task_ids = [str(t) for t, _ in rows]
        vec = DistributionalMetrics.compute_priority_vectors(samples_scores)

        tasks_out: list[dict[str, Any]] = []
        for i, tid in enumerate(task_ids):
            sc = samples_scores[i]
            mean = sum(sc) / len(sc) if sc else 0.0
            tasks_out.append({
                "task_id": tid,
                "num_trajectories": len(sc),
                "mean_score": float(mean),
                "trajectory_scores": [float(x) for x in sc],
                "uncertainty_raw": float(vec["uncertainties_raw"][i]),
                "difficulty_raw": float(vec["difficulties_raw"][i]),
                "uncertainty_normalized": float(vec["uncertainties_normalized"][i]),
                "difficulty_normalized": float(vec["difficulties_normalized"][i]),
                "priority": float(vec["priorities"][i]),
            })

        ranked_indices = sorted(
            range(len(tasks_out)),
            key=lambda i: tasks_out[i]["priority"],
            reverse=True,
        )
        ranking_desc = []
        for rank, idx in enumerate(ranked_indices):
            entry = dict(tasks_out[idx])
            entry["rank"] = int(rank)
            ranking_desc.append(entry)

        prio_desc = [tasks_out[i]["priority"] for i in ranked_indices]
        elbow_detail = SelectiveReflectionMetrics.second_diff_elbow_detail(
            prio_desc, sensitivity=sensitivity
        )
        elbow_k = SelectiveReflectionMetrics.adaptive_elbow_count(
            prio_desc, sensitivity=sensitivity
        )
        cap = int(max(1, max_reflection_cases))
        elbow_k = min(elbow_k, cap, len(ranked_indices))

        selected_ids = [str(ranking_desc[r]["task_id"]) for r in range(elbow_k)]

        report = {
            "schema": "skill_mas_priority_selection_v1",
            "selection_mode": "second_diff_elbow",
            "priority_definition": {
                "uncertainty": "population_std_of_trajectory_scores",
                "difficulty": "negative_mean_trajectory_score",
                "blend": "(minmax_norm(u)+minmax_norm(d))/2 within_round",
            },
            "normalization": "min_max_across_tasks_in_round",
            "sensitivity": float(sensitivity),
            "max_reflection_cases_cap": cap,
            "num_tasks": len(tasks_out),
            "tasks": tasks_out,
            "ranking_descending_priority": ranking_desc,
            "elbow_method_detail": elbow_detail,
            "reflection_selected_task_ids": selected_ids,
            "reflection_selected_count": int(elbow_k),
        }
        return selected_ids, report


# ============================================================================
# D. 迁移性指标 (from evolution/contrastive_reflect.py + schemas.py)
# ============================================================================

class TransferabilityMetrics:
    """
    迁移性指标: 跨任务/跨模型对比分析.
    
    来源: evolution/contrastive_reflect.py
    
    核心思想:
      - source_gap = high_score - low_score (within-task contrast)
      - 跨 LLM 迁移: 在 LLM_A 上进化的 Skill → LLM_B 上测试
      - 跨 Task 迁移: 在 Task_Set_A 上进化 → Task_Set_B 上测试
    """

    @staticmethod
    def compute_source_gap(
        trajectories: list[TrajectoryRecord]
    ) -> tuple[float, str, str]:
        """
        Within-task contrastive gap: 高分轨迹与低分轨迹的差距.
        
        公式: source_gap = max(scores) - min(scores)
        
        Returns:
            (gap, high_traj_tag, low_traj_tag)
        """
        if not trajectories:
            return 0.0, "", ""
        sorted_trajs = sorted(trajectories, key=lambda x: x.score)
        low = sorted_trajs[0]
        high = sorted_trajs[-1]
        return float(high.score) - float(low.score), high.trajectory_tag, low.trajectory_tag

    @staticmethod
    def cross_llm_transfer_delta(
        source_llm_scores: list[float],
        target_llm_scores: list[float]
    ) -> dict[str, float]:
        """
        跨 LLM 迁移增益.
        
        公式:
          source_avg = mean(source_llm_scores)
          target_avg = mean(target_llm_scores)
          delta = target_avg - source_avg
          relative = delta / source_avg * 100  (percentage points)
        
        Returns:
            {"source_avg": float, "target_avg": float, "delta": float, "relative_pp": float}
        """
        s = sum(source_llm_scores) / len(source_llm_scores) if source_llm_scores else 0.0
        t = sum(target_llm_scores) / len(target_llm_scores) if target_llm_scores else 0.0
        delta = t - s
        relative_pp = (delta / s * 100) if s > 1e-8 else 0.0
        return {
            "source_avg": s,
            "target_avg": t,
            "delta": delta,
            "relative_pp": relative_pp,
        }

    @staticmethod
    def cross_task_transfer_matrix(
        source_task_scores: dict[str, list[float]],
        target_task_scores: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        """
        跨任务迁移矩阵 (heatmap 数据).
        
        Args:
            source_task_scores: {task_id: [scores...]} (进化时使用的任务)
            target_task_scores: {task_id: [scores...]} (迁移后测试的任务)
        
        Returns:
            {"per_task": {task_id: delta}, "overall": float, "num_source": int, "num_target": int}
        """
        per_task: dict[str, float] = {}
        for tid in target_task_scores:
            t_scores = target_task_scores[tid]
            s_scores = source_task_scores.get(tid, [])
            if not s_scores:
                per_task[tid] = 0.0
                continue
            s_avg = sum(s_scores) / len(s_scores)
            t_avg = sum(t_scores) / len(t_scores)
            per_task[tid] = t_avg - s_avg

        all_deltas = list(per_task.values())
        overall = sum(all_deltas) / len(all_deltas) if all_deltas else 0.0
        return {
            "per_task": per_task,
            "overall": overall,
            "num_source": len(source_task_scores),
            "num_target": len(target_task_scores),
        }


# ============================================================================
# E. 成本指标 (from utils/llm_cost.py)
# ============================================================================

@dataclass
class UsageTotals:
    """LLM usage 总量 (from llm_cost.py empty_usage_totals)."""
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def empty() -> "UsageTotals":
        return UsageTotals()

    def add(self, src: dict[str, Any]) -> None:
        """合并另一条 usage 记录 (from add_usage_totals)."""
        s = dict(src or {})
        self.prompt_tokens += int(s.get("prompt_tokens", 0) or 0)
        out = int(s.get("output_tokens", s.get("completion_tokens", 0)) or 0)
        self.output_tokens += out
        self.total_tokens += int(s.get("total_tokens", 0) or 0)
        self.estimated_cost_usd += float(s.get("estimated_cost_usd", 0.0) or 0.0)


class CostMetrics:
    """
    成本指标计算器.
    
    来源: utils/llm_cost.py
    
    分为两类:
      1. Inference Cost: rollout 阶段所有 LLM 调用的 token 消耗
      2. Evolution Cost: 反思 + bank 优化阶段的 LLM 调用消耗
    """

    # 默认定价 (per 1M tokens, USD) — 示例值, 实际从 model_config.json 加载
    DEFAULT_PRICING = {
        "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
        "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
        "deepseek-chat": {"input_per_1m": 0.14, "output_per_1m": 0.28},
        "deepseek-reasoner": {"input_per_1m": 0.55, "output_per_1m": 2.19},
        "gemini-2.0-flash": {"input_per_1m": 0.10, "output_per_1m": 0.40},
        "qwen-max": {"input_per_1m": 1.60, "output_per_1m": 6.40},
    }

    @staticmethod
    def estimate_cost(
        prompt_tokens: int,
        output_tokens: int,
        model: str,
        pricing_table: dict[str, Any] | None = None
    ) -> float:
        """
        估算单次 LLM 调用的 USD 成本.
        
        公式:
          cost = (prompt_tokens / 1M) * input_rate + (output_tokens / 1M) * output_rate
        
        Args:
            prompt_tokens: 输入 token 数
            output_tokens: 输出 token 数
            model: 模型名称
            pricing_table: 定价表 (None 则用默认)
        """
        table = pricing_table or CostMetrics.DEFAULT_PRICING
        rates = table.get(model, table.get("gpt-4o", {"input_per_1m": 2.50, "output_per_1m": 10.00}))
        input_cost = (prompt_tokens / 1_000_000) * rates["input_per_1m"]
        output_cost = (output_tokens / 1_000_000) * rates["output_per_1m"]
        return round(input_cost + output_cost, 10)

    @staticmethod
    def build_rollout_cost_report(
        per_trajectory_usage: list[dict[str, Any]],
        model: str,
        pricing_table: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        构建 rollout 阶段成本报告 (from vita_rollout_cost_report).
        
        Args:
            per_trajectory_usage: [{"prompt_tokens": int, "output_tokens": int, "total_tokens": int}, ...]
            model: 模型名称
            pricing_table: 定价表
        
        Returns:
            {
                "phase": "rollout",
                "model": str,
                "aggregate_usage": UsageTotals,
                "per_trajectory": [...],
                "estimated_cost_usd": float,
            }
        """
        grand = UsageTotals.empty()
        per_traj: list[dict[str, Any]] = []

        for traj_usage in per_trajectory_usage:
            pt = int(traj_usage.get("prompt_tokens", 0) or 0)
            ot = int(traj_usage.get("output_tokens", 0) or 0)
            tt = int(traj_usage.get("total_tokens", pt + ot) or 0)
            cost = CostMetrics.estimate_cost(pt, ot, model, pricing_table)

            entry = {
                "prompt_tokens": pt,
                "output_tokens": ot,
                "total_tokens": tt,
                "estimated_cost_usd": cost,
            }
            entry["estimated_cost_usd"] = cost
            grand.add(entry)
            per_traj.append(entry)

        return {
            "phase": "rollout",
            "model": model,
            "aggregate_usage": grand.to_dict(),
            "per_trajectory": per_traj,
            "estimated_cost_usd": grand.estimated_cost_usd,
        }

    @staticmethod
    def build_evolution_cost_report(
        optimizer_calls: list[dict[str, Any]],
        pricing_table: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        构建进化阶段成本报告 (from build_round_cost_document).
        
        Args:
            optimizer_calls: [{"phase": str, "model": str, "usage": {"prompt_tokens": int, ...}}, ...]
            pricing_table: 定价表
        
        Returns:
            {
                "phase": "evolution",
                "sections": [...],
                "round_total_estimated_cost_usd": float,
            }
        """
        sections: list[dict[str, Any]] = []
        total = 0.0

        for call in optimizer_calls:
            model = call.get("model", "gpt-4o")
            usage = call.get("usage", {})
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ot = int(usage.get("output_tokens", 0) or 0)
            cost = CostMetrics.estimate_cost(pt, ot, model, pricing_table)

            section = {
                "phase": call.get("phase", "unknown"),
                "model": model,
                "prompt_tokens": pt,
                "output_tokens": ot,
                "total_tokens": int(usage.get("total_tokens", pt + ot) or 0),
                "estimated_cost_usd": cost,
            }
            sections.append(section)
            total += cost

        return {
            "phase": "evolution",
            "sections": sections,
            "round_total_estimated_cost_usd": round(total, 10),
        }

    @staticmethod
    def cumulative_cost_summary(
        round_costs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        累积成本汇总 (from merge_cumulative_summary).
        
        Args:
            round_costs: [{"round_idx": int, "round_total_estimated_cost_usd": float}, ...]
        
        Returns:
            {
                "rounds": [...],
                "cumulative_estimated_cost_usd": float,
            }
        """
        cum = sum(float(r.get("round_total_estimated_cost_usd", 0.0) or 0.0) for r in round_costs)
        return {
            "schema": "skill_mas_evolve_llm_cost_cumulative_v1",
            "rounds": round_costs,
            "cumulative_estimated_cost_usd": round(cum, 10),
        }


# ============================================================================
# F. 演化追踪指标 (from evolution/bank_optimizer.py + assemble_select.py)
# ============================================================================

class EvolutionTrackingMetrics:
    """
    演化追踪指标: 跨 round 的 Skill 进化监控.
    
    来源: evolution/bank_optimizer.py (_write_knee_artifacts)
          evolution/assemble_select.py (finalize_best_round)
    """

    @staticmethod
    def compute_round_priority_report(
        by_task: dict[str, list[TrajectoryRecord]],
        round_idx: int,
        max_reflection_cases: int = 10,
        sensitivity: float = 1.0
    ) -> dict[str, Any]:
        """
        单 round 的优先级选择报告 (from _write_knee_artifacts).
        
        包含:
          - per-task uncertainty/difficulty/priority
          - 二阶差分肘部检测结果
          - knee index 和 knee task
        """
        task_rows: list[tuple[str, list[float]]] = []
        for task_id, rows in sorted(by_task.items(), key=lambda kv: str(kv[0])):
            if not rows:
                continue
            task_rows.append((str(task_id), [float(r.score) for r in rows]))

        ranked: list[tuple[str, float]] = []
        if task_rows:
            samples_scores = [s for _, s in task_rows]
            priorities = DistributionalMetrics.compute_priority_scores(samples_scores)
            ranked = sorted(
                zip([t for t, _ in task_rows], priorities),
                key=lambda x: x[1],
                reverse=True,
            )

        payload: dict[str, Any] = {
            "round_idx": int(round_idx),
            "method": "second_diff_elbow",
            "priority_metric": "uncertainty_std_mean_difficulty_blend",
            "num_tasks": len(ranked),
            "priorities_desc": [{"task_id": t, "priority": float(p)} for t, p in ranked],
        }

        if ranked:
            values = [p for _, p in ranked]
            detail = SelectiveReflectionMetrics.second_diff_elbow_detail(values, sensitivity=sensitivity)
            payload.update({k: v for k, v in detail.items() if k != "n"})

            sdi = detail.get("second_diff_argmax_index")
            if sdi is not None:
                kidx = min(int(sdi) + 1, len(ranked) - 1)
            else:
                kidx = min(max(int(detail.get("selected_count") or 1) - 1, 0), len(ranked) - 1)
            payload["knee_index"] = int(kidx)
            payload["knee_task_id"] = ranked[kidx][0]
            payload["knee_priority"] = float(ranked[kidx][1])
            payload["knee_gap"] = float(ranked[kidx][1])
        else:
            payload["knee_index"] = None
            payload["knee_task_id"] = None
            payload["knee_priority"] = None
            payload["knee_gap"] = None

        # Full priority selection report
        _, priority_report = SelectiveReflectionMetrics.compute_reflection_task_selection(
            task_rows,
            max_reflection_cases=max_reflection_cases,
            sensitivity=sensitivity,
        )
        payload["priority_selection"] = priority_report

        return payload

    @staticmethod
    def track_skill_evolution(
        round_scores: list[dict[str, Any]],
        bank_meta_history: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        追踪 Skill 进化轨迹.
        
        Args:
            round_scores: [{"round_idx": int, "round_score": float}, ...]
            bank_meta_history: [{"round_idx": int, "updated_skill_name": str, "selected_task_ids": [...]}, ...]
        
        Returns:
            {
                "best_round_idx": int,
                "best_round_score": float,
                "convergence_round": int,       # 分数首次不再显著提升的 round
                "score_trajectory": [...],
                "total_rounds": int,
                "improvement_from_baseline": float,
            }
        """
        if not round_scores:
            return {
                "best_round_idx": 0,
                "best_round_score": 0.0,
                "convergence_round": 0,
                "score_trajectory": [],
                "total_rounds": 0,
                "improvement_from_baseline": 0.0,
            }

        sorted_rounds = sorted(round_scores, key=lambda x: int(x["round_idx"]))
        scores = [float(r["round_score"]) for r in sorted_rounds]
        baseline = scores[0] if scores else 0.0

        best_idx = 0
        best_score = baseline
        for i, s in enumerate(scores):
            if s > best_score:
                best_score = s
                best_idx = i

        # Convergence: 首次连续3轮提升 < 0.5pp
        convergence_round = len(scores) - 1
        for i in range(2, len(scores)):
            recent_gains = [scores[j] - scores[j-1] for j in range(i-2, i+1)]
            if all(abs(g) < 0.005 for g in recent_gains):
                convergence_round = i
                break

        return {
            "best_round_idx": int(sorted_rounds[best_idx]["round_idx"]),
            "best_round_score": best_score,
            "convergence_round": int(sorted_rounds[convergence_round]["round_idx"]),
            "score_trajectory": [
                {"round_idx": int(r["round_idx"]), "round_score": float(r["round_score"])}
                for r in sorted_rounds
            ],
            "total_rounds": len(scores),
            "improvement_from_baseline": best_score - baseline,
            "baseline_score": baseline,
            "best_improvement_pp": (best_score - baseline) * 100,
        }


# ============================================================================
# G. 各 Benchmark 评分 (from dataset/ scoring files)
# ============================================================================

class HLEMATHScorer:
    """
    HLE-Math 评分器.
    
    来源: dataset/hlemath/score.py
    
    评分方式: 提取 \\boxed{} 答案, 用 sympy 判断数学等价性
    输出: 0 (错误) 或 1 (正确)
    """

    @staticmethod
    def extract_model_answer(text: str) -> str:
        """提取模型输出的答案 (优先 \\boxed{})."""
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()
        # Fallback: 最后一句话
        sentences = re.split(r"(?<!\d)[.!?]\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    @staticmethod
    def calculate_score(expected_output: str, prediction: str) -> tuple[int, str]:
        """
        计算 HLE-Math 分数.
        
        Returns:
            (0|1, extracted_prediction)
        """
        expected = HLEMATHScorer.extract_model_answer(expected_output)
        predicted = HLEMATHScorer.extract_model_answer(prediction)

        if str(predicted) == str(expected):
            return 1, predicted

        # 尝试数值比较
        try:
            p = float(str(predicted).replace(",", ""))
            r = float(str(expected).replace(",", ""))
            if abs(p - r) < 1e-3:
                return 1, predicted
        except (ValueError, TypeError):
            pass

        # 尝试符号比较 (简化版, 完整版需 sympy)
        try:
            if str(predicted).replace(" ", "") == str(expected).replace(" ", ""):
                return 1, predicted
        except Exception:
            pass

        return 0, predicted


class BrowseCompScorer:
    """
    BrowseComp-Plus 评分器.
    
    来源: dataset/BrowseComp-Plus/score.py
    
    评分方式: 归一化后精确匹配 (或 LLM Judge)
    输出: 0 (错误) 或 1 (正确)
    """

    @staticmethod
    def _normalize(text: str) -> str:
        s = (text or "").strip().lower()
        s = re.sub(r"\s+", " ", s)
        s = s.strip(" \t\r\n.,;:!?\"'`")
        return s

    @staticmethod
    def calculate_score(expected_output: str, prediction: str) -> tuple[int, str]:
        """
        计算 BrowseComp 分数.
        
        Returns:
            (0|1, raw_prediction)
        """
        gold = BrowseCompScorer._normalize(expected_output)
        pred_raw = (prediction or "").strip()
        pred = BrowseCompScorer._normalize(pred_raw)
        return (1 if pred == gold else 0), pred_raw


class VitaBenchMetrics:
    """
    VitaBench 评分指标.
    
    来源: dataset/vitabench/src/vita/metrics/agent_metrics.py
    
    评分方式:
      1. NL Assertion Ratio: satisfied_rubrics / total_rubrics
      2. pass^k = C(c,k) / C(n,k)
      3. pass@k = 1 - C(n-c,k) / C(n,k)
      4. average@k = mean(rewards)
    """

    @staticmethod
    def nl_assertion_ratio(reward_rubrics: list[dict[str, Any]]) -> float:
        """
        NL Assertion Ratio: 满足的 rubric 比例.
        
        来源: rollout_multi.py _vita_nl_assertion_score
        
        Args:
            reward_rubrics: [{"met": True/False, ...}, ...]
        
        Returns:
            float: [0, 1], satisfied / total
        """
        if not reward_rubrics:
            return 0.0
        total = 0
        met = 0
        for item in reward_rubrics:
            if not isinstance(item, dict) or "met" not in item:
                continue
            total += 1
            if bool(item.get("met")):
                met += 1
        return float(met) / float(total) if total > 0 else 0.0

    @staticmethod
    def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
        """
        pass^k 指标 (from arXiv:2406.12045).
        
        公式: pass^k = C(c, k) / C(n, k)
        
        其中:
          n = 总试验数
          c = 成功数
          k = 选取数
        
        含义: 随机选 k 个试验全部成功的概率
        """
        if num_trials < k:
            raise ValueError(f"trials {num_trials} < k {k}")
        return math.comb(success_count, k) / math.comb(num_trials, k)

    @staticmethod
    def pass_at_k(num_trials: int, success_count: int, k: int) -> float:
        """
        pass@k 指标.
        
        公式: pass@k = 1 - C(n-c, k) / C(n, k)
        
        含义: 至少 k 个试验中有 1 个成功的概率
        """
        if num_trials < k:
            return 0.0
        if success_count > num_trials:
            return 0.0
        if num_trials - success_count >= k:
            return 1.0 - (math.comb(num_trials - success_count, k) / math.comb(num_trials, k))
        else:
            return 1.0

    @staticmethod
    def average_at_k(rewards: list[float], k: int) -> float:
        """average@k = mean(all rewards)."""
        if len(rewards) < k or k == 0:
            return 0.0
        return sum(rewards) / len(rewards)

    @staticmethod
    def compute_all_metrics(
        task_rewards: dict[str, list[float]],
        num_trials: int
    ) -> dict[str, Any]:
        """
        计算 VitaBench 全部指标.
        
        Args:
            task_rewards: {task_id: [reward_1, reward_2, ...]}
            num_trials: 每个任务的试验次数
        
        Returns:
            {
                "avg_reward": float,
                "pass_hat_ks": {k: float},
                "pass_at_n": {k: float},
                "average_at_n": {k: float},
            }
        """
        all_rewards = []
        for tid, rewards in task_rewards.items():
            all_rewards.extend(rewards)

        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

        pass_hat_ks: dict[int, float] = {}
        pass_at_n: dict[int, float] = {}
        average_at_n: dict[int, float] = {}

        for k in range(1, num_trials + 1):
            phk_values = []
            pak_values = []
            aak_values = []

            for tid, rewards in task_rewards.items():
                n = len(rewards)
                c = sum(1 for r in rewards if r == 1.0)

                if n >= k:
                    phk_values.append(VitaBenchMetrics.pass_hat_k(n, c, k))
                    pak_values.append(VitaBenchMetrics.pass_at_k(n, c, k))
                    aak_values.append(VitaBenchMetrics.average_at_k(rewards, k))

            if phk_values:
                pass_hat_ks[k] = sum(phk_values) / len(phk_values)
            if pak_values:
                pass_at_n[k] = sum(pak_values) / len(pak_values)
            if aak_values:
                average_at_n[k] = sum(aak_values) / len(aak_values)

        return {
            "avg_reward": avg_reward,
            "pass_hat_ks": pass_hat_ks,
            "pass_at_n": pass_at_n,
            "average_at_n": average_at_n,
        }


class DRBScorer:
    """
    DeepResearch Bench RACE 评分器.
    
    来源: dataset/deep_research_bench/utils/score_calculator.py
    
    评分方式: 加权多维度评分 (LLM Judge)
    输出: [0, 10] 的加权总分
    """

    @staticmethod
    def calculate_weighted_scores(
        llm_output_json: dict[str, list[dict[str, Any]]],
        criteria_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        计算 DRB 加权分数.
        
        Args:
            llm_output_json: {dimension: [{"criterion": str, "target_score": float, "article_2_score": float}, ...]}
            criteria_data: {"dimension_weight": {dim: weight}, "criterions": {dim: [{criterion, weight}]}}
        
        Returns:
            {
                "target": {"dims": {dim: weighted_avg}, "total": float},
                "reference": {"dims": {dim: weighted_avg}, "total": float},
            }
        """
        results = {
            "target": {"dims": {}, "total": 0.0},
            "reference": {"dims": {}, "total": 0.0}
        }
        total_target = 0.0
        total_reference = 0.0

        dimension_weights = criteria_data.get("dimension_weight", {})

        # Build criterion -> weight mapping per dimension
        criterion_weights: dict[str, dict[str, float]] = {}
        for dim, criterions in criteria_data.get("criterions", {}).items():
            criterion_weights[dim] = {c["criterion"]: c["weight"] for c in criterions}

        for dim, scores_list in llm_output_json.items():
            if dim not in dimension_weights:
                continue
            if dim not in criterion_weights:
                continue

            dim_target_sum = 0.0
            dim_reference_sum = 0.0
            dim_total_weight = 0.0

            for score_item in scores_list:
                criterion_text = str(score_item.get("criterion", "")).strip()
                target_score = score_item.get("target_score") or score_item.get("article_1_score")
                ref_score = score_item.get("article_2_score")

                if not criterion_text or target_score is None:
                    continue

                weight = criterion_weights[dim].get(criterion_text, 1.0)

                dim_target_sum += float(target_score) * weight
                dim_total_weight += weight

                if ref_score is not None:
                    dim_reference_sum += float(ref_score) * weight

            if dim_total_weight > 0:
                dim_target_avg = dim_target_sum / dim_total_weight
                dim_ref_avg = dim_reference_sum / dim_total_weight if ref_score is not None else 0
            else:
                dim_target_avg = 0
                dim_ref_avg = 0

            results["target"]["dims"][f"{dim}_weighted_avg"] = dim_target_avg
            results["reference"]["dims"][f"{dim}_weighted_avg"] = dim_ref_avg

            dim_weight = dimension_weights.get(dim, 0)
            total_target += dim_target_avg * dim_weight
            total_reference += dim_ref_avg * dim_weight

        results["target"]["total"] = total_target
        results["reference"]["total"] = total_reference
        return results

    @staticmethod
    def citation_stats(citations_data: list[dict[str, Any]]) -> dict[str, float]:
        """
        引用统计 (from utils/stat.py).
        
        Returns:
            {
                "avg_citations_per_task": float,
                "avg_valid_citations_per_task": float,
                "valid_rate": float,
            }
        """
        total_citations = 0
        total_valid = 0
        total_num = 0

        for d in citations_data:
            if not d.get("citations"):
                continue
            for c in d.get("citations_deduped", {}).values():
                if c.get("validate_error") is not None:
                    continue
                for _c in c.get("validate_res", []):
                    if _c["result"] != "unknown":
                        total_citations += 1
                        if _c["result"] == "supported":
                            total_valid += 1
            total_num += 1

        return {
            "avg_citations_per_task": total_citations / max(total_num, 1),
            "avg_valid_citations_per_task": total_valid / max(total_num, 1),
            "valid_rate": total_valid / max(total_citations, 1),
        }


# ============================================================================
# 综合指标报告生成器
# ============================================================================

class SkillMASMetricsReport:
    """
    综合指标报告生成器: 整合所有 7 类指标.
    """

    @staticmethod
    def generate_full_report(
        by_task: dict[str, list[TrajectoryRecord]],
        round_idx: int,
        optimizer_usage: list[dict[str, Any]] | None = None,
        rollout_usage: list[dict[str, Any]] | None = None,
        model: str = "gpt-4o",
        max_reflection_cases: int = 10,
        previous_rounds: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        生成单 round 的完整指标报告.
        
        Returns:
            包含全部 7 类指标的字典
        """
        # A. 主性能
        round_score = MainPerformanceMetrics.compute_round_score(by_task)
        per_task = MainPerformanceMetrics.per_task_mean_scores(by_task)
        all_scores = MainPerformanceMetrics.per_task_all_scores(by_task)

        # B. 分布统计
        samples_scores = [all_scores[tid] for tid in sorted(all_scores.keys())]
        priority_vectors = DistributionalMetrics.compute_priority_vectors(samples_scores)

        # C. 选择性反思
        task_rows = [(tid, all_scores[tid]) for tid in sorted(all_scores.keys())]
        selected_ids, selection_report = SelectiveReflectionMetrics.compute_reflection_task_selection(
            task_rows, max_reflection_cases=max_reflection_cases
        )

        # D. 迁移性 (within-task contrast)
        contrasts = {}
        for tid, trajs in by_task.items():
            gap, high_tag, low_tag = TransferabilityMetrics.compute_source_gap(trajs)
            contrasts[tid] = {
                "source_gap": gap,
                "high_traj": high_tag,
                "low_traj": low_tag,
            }

        # E. 成本
        rollout_cost = None
        if rollout_usage:
            rollout_cost = CostMetrics.build_rollout_cost_report(rollout_usage, model)

        evolution_cost = None
        if optimizer_usage:
            evolution_cost = CostMetrics.build_evolution_cost_report(optimizer_usage)

        # F. 演化追踪
        evolution_report = EvolutionTrackingMetrics.compute_round_priority_report(
            by_task, round_idx, max_reflection_cases
        )

        # G. Benchmark 特定指标
        bench_backend = list(by_task.values())[0][0].bench_backend if by_task else "unknown"
        bench_metrics = {}
        if bench_backend == "vitabench":
            task_rewards = {tid: [r.score for r in trajs] for tid, trajs in by_task.items()}
            k_traj = max(len(trajs) for trajs in by_task.values()) if by_task else 1
            bench_metrics = VitaBenchMetrics.compute_all_metrics(task_rewards, k_traj)
        elif bench_backend == "hlemath":
            bench_metrics = {"scoring_method": "sympy_symbolic_equality", "output_range": [0, 1]}
        elif bench_backend == "bcp":
            bench_metrics = {"scoring_method": "normalized_exact_match_or_llm_judge", "output_range": [0, 1]}
        elif bench_backend == "drb":
            bench_metrics = {"scoring_method": "weighted_multi_dimension_llm_judge", "output_range": [0, 10]}

        return {
            "schema": "skill_mas_full_metrics_report_v1",
            "round_idx": round_idx,
            "bench_backend": bench_backend,
            "num_tasks": len(by_task),
            "A_main_performance": {
                "round_score": round_score,
                "per_task_mean_scores": per_task,
                "per_task_all_scores": all_scores,
            },
            "B_distributional": {
                "uncertainties_raw": priority_vectors["uncertainties_raw"],
                "difficulties_raw": priority_vectors["difficulties_raw"],
                "uncertainties_normalized": priority_vectors["uncertainties_normalized"],
                "difficulties_normalized": priority_vectors["difficulties_normalized"],
                "priorities": priority_vectors["priorities"],
            },
            "C_selective_reflection": {
                "selected_task_ids": selected_ids,
                "selected_count": len(selected_ids),
                "selection_report": selection_report,
            },
            "D_transferability": {
                "within_task_contrasts": contrasts,
            },
            "E_cost": {
                "rollout_cost": rollout_cost,
                "evolution_cost": evolution_cost,
            },
            "F_evolution_tracking": evolution_report,
            "G_benchmark_specific": bench_metrics,
        }


# ============================================================================
# Mock 数据生成 + 自测
# ============================================================================

def _generate_mock_data(
    n_tasks: int = 8,
    k_trajectories: int = 5,
    round_idx: int = 3,
    bench_backend: str = "vitabench"
) -> dict[str, list[TrajectoryRecord]]:
    """生成 mock 轨迹数据用于测试."""
    import random
    random.seed(42 + round_idx)

    by_task: dict[str, list[TrajectoryRecord]] = {}
    for t in range(n_tasks):
        tid = f"task_{t:03d}"
        # 不同任务有不同的难度
        base_score = random.uniform(0.2, 0.8)
        scores = [max(0.0, min(1.0, base_score + random.gauss(0, 0.15))) for _ in range(k_trajectories)]

        by_task[tid] = [
            TrajectoryRecord(
                schema="skill_mas_trajectory_record_v1",
                bench_backend=bench_backend,
                round_idx=round_idx,
                task_id=tid,
                trajectory_idx=k,
                trajectory_tag=f"task_{t:03d}_traj_{k:02d}",
                score=score,
                score_source="vitabench_nl_assertion_ratio" if bench_backend == "vitabench" else "mock",
                log_path=f"/mock/{tid}/traj_{k:02d}.json",
                raw_result_path=f"/mock/{tid}/traj_{k:02d}.json",
                phase_snapshots=[
                    PhaseSnapshot(phase="任务分解", instruction="...", output_preview="..."),
                    PhaseSnapshot(phase="Agent工程", instruction="...", output_preview="..."),
                    PhaseSnapshot(phase="工作流编排", instruction="...", output_preview="..."),
                ],
            )
            for k, score in enumerate(scores)
        ]
    return by_task


if __name__ == "__main__":
    print("=" * 80)
    print("Skill-MAS Evaluation Metrics — Self Test")
    print("Source: https://github.com/linhh29/Skill-MAS")
    print("=" * 80)

    # 生成 mock 数据
    by_task = _generate_mock_data(n_tasks=8, k_trajectories=5, round_idx=3, bench_backend="vitabench")

    # 生成完整报告
    mock_rollout_usage = [
        {"prompt_tokens": 12000, "output_tokens": 3000, "total_tokens": 15000},
        {"prompt_tokens": 15000, "output_tokens": 4000, "total_tokens": 19000},
    ]
    mock_optimizer_usage = [
        {"phase": "contrastive_reflection_phase1", "model": "gpt-4o", "usage": {"prompt_tokens": 8000, "output_tokens": 2000}},
        {"phase": "contrastive_reflection_phase2", "model": "gpt-4o", "usage": {"prompt_tokens": 12000, "output_tokens": 3000}},
        {"phase": "bank_optimizer_three_stage", "model": "gpt-4o", "usage": {"prompt_tokens": 10000, "output_tokens": 5000}},
    ]

    report = SkillMASMetricsReport.generate_full_report(
        by_task=by_task,
        round_idx=3,
        optimizer_usage=mock_optimizer_usage,
        rollout_usage=mock_rollout_usage,
        model="gpt-4o",
        max_reflection_cases=5,
    )

    # 打印报告
    print(f"\n{'─' * 60}")
    print(f"Round {report['round_idx']} | Backend: {report['bench_backend']} | Tasks: {report['num_tasks']}")
    print(f"{'─' * 60}")

    # A. 主性能
    print(f"\n[A] 主性能指标 (Main Performance)")
    print(f"    Round Score (Avg.Perf):      {report['A_main_performance']['round_score']:.4f}")
    per_task = report['A_main_performance']['per_task_mean_scores']
    for tid, score in sorted(per_task.items()):
        print(f"      {tid}: {score:.4f}")

    # B. 分布统计
    print(f"\n[B] 分布统计指标 (Distributional Stats)")
    u_raw = report['B_distributional']['uncertainties_raw']
    d_raw = report['B_distributional']['difficulties_raw']
    prios = report['B_distributional']['priorities']
    for i, tid in enumerate(sorted(per_task.keys())):
        print(f"    {tid}: uncertainty={u_raw[i]:.4f}  difficulty={d_raw[i]:.4f}  priority={prios[i]:.4f}")

    # C. 选择性反思
    print(f"\n[C] 选择性反思指标 (Selective Reflection)")
    sel = report['C_selective_reflection']
    print(f"    Selected tasks: {sel['selected_count']}/{report['num_tasks']}")
    for tid in sel['selected_task_ids']:
        print(f"      → {tid}")
    elbow = report['C_selective_reflection']['selection_report']['elbow_method_detail']
    print(f"    Elbow method: second_diff_argmax={elbow.get('second_diff_argmax_index')}")
    print(f"    Elbow idx (pre-sensitivity): {elbow.get('elbow_idx_before_sensitivity')}")
    print(f"    Final selected count: {elbow.get('selected_count')}")

    # D. 迁移性
    print(f"\n[D] 迁移性指标 (Transferability — Within-task Contrast)")
    contrasts = report['D_transferability']['within_task_contrasts']
    for tid, c in sorted(contrasts.items()):
        print(f"    {tid}: gap={c['source_gap']:.4f}  high={c['high_traj']}  low={c['low_traj']}")

    # E. 成本
    print(f"\n[E] 成本指标 (Cost Metrics)")
    if report['E_cost']['rollout_cost']:
        rc = report['E_cost']['rollout_cost']
        print(f"    Rollout Cost:")
        print(f"      Total tokens: {rc['aggregate_usage']['total_tokens']:,}")
        print(f"      Estimated cost: ${rc['aggregate_usage']['estimated_cost_usd']:.6f}")
    if report['E_cost']['evolution_cost']:
        ec = report['E_cost']['evolution_cost']
        print(f"    Evolution Cost:")
        for s in ec['sections']:
            print(f"      {s['phase']}: ${s['estimated_cost_usd']:.6f}")
        print(f"      Round total: ${ec['round_total_estimated_cost_usd']:.6f}")

    # F. 演化追踪
    print(f"\n[F] 演化追踪指标 (Evolution Tracking)")
    evo = report['F_evolution_tracking']
    print(f"    Method: {evo['method']}")
    print(f"    Priority metric: {evo['priority_metric']}")
    if evo.get('knee_task_id'):
        print(f"    Knee task: {evo['knee_task_id']} (priority={evo['knee_priority']:.4f})")
    print(f"    Priorities (desc):")
    for p in evo['priorities_desc'][:5]:
        print(f"      {p['task_id']}: {p['priority']:.4f}")

    # G. Benchmark 特定
    print(f"\n[G] Benchmark 特定指标 ({report['bench_backend']})")
    g = report['G_benchmark_specific']
    if "avg_reward" in g:
        print(f"    Avg Reward: {g['avg_reward']:.4f}")
        print(f"    Pass^k: {g['pass_hat_ks']}")
        print(f"    Pass@k: {g['pass_at_n']}")
        print(f"    Average@k: {g['average_at_n']}")
    else:
        print(f"    {g}")

    # 测试多 round 演化追踪
    print(f"\n{'─' * 60}")
    print(f"[Extra] 多 Round 演化追踪测试")
    print(f"{'─' * 60}")
    mock_round_scores = [
        {"round_idx": 0, "round_score": 0.35, "skill_path": "/mock/round_00/SKILL.md"},
        {"round_idx": 1, "round_score": 0.52, "skill_path": "/mock/round_01/SKILL.md"},
        {"round_idx": 2, "round_score": 0.61, "skill_path": "/mock/round_02/SKILL.md"},
        {"round_idx": 3, "round_score": 0.63, "skill_path": "/mock/round_03/SKILL.md"},
        {"round_idx": 4, "round_score": 0.64, "skill_path": "/mock/round_04/SKILL.md"},
    ]
    evo_track = EvolutionTrackingMetrics.track_skill_evolution(
        mock_round_scores, []
    )
    print(f"    Best round: {evo_track['best_round_idx']} (score={evo_track['best_round_score']:.4f})")
    print(f"    Convergence round: {evo_track['convergence_round']}")
    print(f"    Baseline: {evo_track['baseline_score']:.4f}")
    print(f"    Improvement: +{evo_track['best_improvement_pp']:.2f}pp")
    print(f"    Score trajectory:")
    for s in evo_track['score_trajectory']:
        print(f"      Round {s['round_idx']}: {s['round_score']:.4f}")

    # 测试 Benchmark 评分器
    print(f"\n{'─' * 60}")
    print(f"[Extra] Benchmark 评分器测试")
    print(f"{'─' * 60}")

    # HLE-Math
    score, pred = HLEMATHScorer.calculate_score("\\boxed{42}", "The answer is \\boxed{42}")
    print(f"    HLE-Math: gold=42, pred=42 → score={score}")
    score, pred = HLEMATHScorer.calculate_score("\\boxed{\\frac{1}{2}}", "Answer: \\boxed{0.5}")
    print(f"    HLE-Math: gold=1/2, pred=0.5 → score={score}")

    # BrowseComp
    score, pred = BrowseCompScorer.calculate_score("Paris", "Paris")
    print(f"    BrowseComp: gold=Paris, pred=Paris → score={score}")
    score, pred = BrowseCompScorer.calculate_score("Paris", "  paris. ")
    print(f"    BrowseComp: gold=Paris, pred='  paris. ' → score={score}")

    # VitaBench NL assertions
    rubrics = [{"met": True}, {"met": True}, {"met": False}, {"met": True}]
    ratio = VitaBenchMetrics.nl_assertion_ratio(rubrics)
    print(f"    VitaBench NL assertions: 3/4 met → ratio={ratio:.4f}")

    # VitaBench pass^k
    phk = VitaBenchMetrics.pass_hat_k(5, 3, 2)
    print(f"    VitaBench pass^2 (n=5, c=3): {phk:.4f}")
    pak = VitaBenchMetrics.pass_at_k(5, 3, 2)
    print(f"    VitaBench pass@2 (n=5, c=3): {pak:.4f}")

    print(f"\n{'=' * 80}")
    print("All metrics computed successfully!")
    print(f"{'=' * 80}")
