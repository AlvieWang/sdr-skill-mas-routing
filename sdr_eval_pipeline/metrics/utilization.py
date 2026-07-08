"""
SDR Evaluation Pipeline - Metric C: Utilization & Stability

Measures routing diversity, utilization balance, and collapse detection,
inspired by SkillOrchestra's findings on RL routing collapse.

Metrics:
- Skill-level Utilization Balance: How evenly are models used per skill?
- Routing Entropy: Shannon entropy of model selection distribution
- Routing Collapse Rate: Fraction of steps where one model dominates
- Skill-level Cost-Effectiveness: Pass rate per unit cost per skill
- Pareto Frontier Coverage: Multi-objective trade-off quality
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List

import numpy as np

from core.types import Trajectory, MetricResult, EvaluationConfig, StepResult


class UtilizationMetrics:
    """Category C: Utilization balance and stability metrics."""
    
    CATEGORY = "C"
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(self, trajectories: List[Trajectory]) -> List[MetricResult]:
        results = []
        
        # C1: Skill-level Utilization Balance
        balance = self._utilization_balance(trajectories)
        results.append(MetricResult(
            name="Skill-level Utilization Balance",
            value=balance,
            category=self.CATEGORY,
            description="How evenly models are used per skill (1.0=perfect balance)",
            baseline_value=0.02,  # RL-PER collapse: 98% one model
            detail={"source": "SkillOrchestra"},
        ))
        
        # C2: Routing Entropy
        entropy = self._routing_entropy(trajectories)
        results.append(MetricResult(
            name="Routing Entropy",
            value=entropy,
            category=self.CATEGORY,
            description="Shannon entropy of model selection (higher=more diverse)",
            baseline_value=0.1,  # RL-PER collapse: entropy near 0
            detail={"source": "SkillOrchestra", "unit": "bits"},
        ))
        
        # C3: Routing Collapse Rate
        collapse = self._routing_collapse_rate(trajectories)
        results.append(MetricResult(
            name="Routing Collapse Rate",
            value=collapse,
            category=self.CATEGORY,
            description="Fraction of steps where single model > 95% of selections",
            baseline_value=0.98,  # SkillOrchestra: RL 98% collapse
            detail={"source": "SkillOrchestra (RL 98% collapse)"},
        ))
        
        # C4: Skill-level Cost-Effectiveness
        ce = self._skill_cost_effectiveness(trajectories)
        results.append(MetricResult(
            name="Skill-level Cost-Effectiveness",
            value=ce,
            category=self.CATEGORY,
            description="Average pass rate per unit token cost per skill",
            baseline_value=None,
            detail={"source": "RL-PER + SkillOrchestra"},
        ))
        
        # C5: Pareto Frontier Coverage
        pareto = self._pareto_frontier_coverage(trajectories)
        results.append(MetricResult(
            name="Pareto Frontier Coverage",
            value=pareto,
            category=self.CATEGORY,
            description="Fraction of Pareto-optimal points covered (0-1)",
            baseline_value=None,
            detail={"source": "SkillOrchestra"},
        ))
        
        return results
    
    def _utilization_balance(self, trajectories: List[Trajectory]) -> float:
        """Measure how evenly models are used (1.0 = perfect balance)."""
        all_models = []
        for traj in trajectories:
            for dec in traj.decisions:
                all_models.append(dec.selected_model)
        
        if not all_models:
            return 0.0
        
        counts = Counter(all_models)
        total = len(all_models)
        n_models = len(self.config.models)
        
        # Ideal: each model used 1/n_models of the time
        ideal = 1.0 / n_models
        actual = [counts.get(m, 0) / total for m in self.config.models]
        
        # Balance = 1 - normalized variance
        variance = np.var(actual)
        max_variance = (1 - ideal) ** 2 * n_models  # Worst case: all on one model
        balance = 1.0 - (variance / max_variance if max_variance > 0 else 0)
        
        return round(balance, 3)
    
    def _routing_entropy(self, trajectories: List[Trajectory]) -> float:
        """Shannon entropy of model selection distribution."""
        all_models = []
        for traj in trajectories:
            for dec in traj.decisions:
                all_models.append(dec.selected_model)
        
        if not all_models:
            return 0.0
        
        counts = Counter(all_models)
        total = len(all_models)
        
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return round(entropy, 3)
    
    def _routing_collapse_rate(self, trajectories: List[Trajectory]) -> float:
        """Fraction of steps where one model dominates > threshold."""
        all_models = []
        for traj in trajectories:
            for dec in traj.decisions:
                all_models.append(dec.selected_model)
        
        if not all_models:
            return 0.0
        
        counts = Counter(all_models)
        total = len(all_models)
        max_freq = max(counts.values()) / total
        
        if max_freq > self.config.collapse_threshold:
            return round(max_freq, 3)
        else:
            return round(max(0.0, max_freq - self.config.collapse_threshold), 3)
    
    def _skill_cost_effectiveness(self, trajectories: List[Trajectory]) -> float:
        """Average pass rate per unit token cost per skill."""
        skill_data = defaultdict(lambda: {"success": 0, "total": 0, "tokens": 0})
        
        for traj in trajectories:
            for result in traj.results:
                for skill in result.skills_involved:
                    skill_data[skill]["total"] += 1
                    skill_data[skill]["tokens"] += result.token_cost
                    if result.success:
                        skill_data[skill]["success"] += 1
        
        if not skill_data:
            return 0.0
        
        ce_values = []
        for skill, data in skill_data.items():
            if data["tokens"] > 0:
                pass_rate = data["success"] / data["total"] if data["total"] > 0 else 0
                ce = pass_rate / (data["tokens"] / data["total"]) * 1000  # per 1K tokens
                ce_values.append(ce)
        
        return round(np.mean(ce_values), 3) if ce_values else 0.0
    
    def _pareto_frontier_coverage(self, trajectories: List[Trajectory]) -> float:
        """Fraction of Pareto-optimal (cost, quality) points covered."""
        points = []
        for traj in trajectories:
            if traj.total_tokens > 0:
                quality = 1.0 if traj.task_success else 0.0
                cost = traj.total_tokens
                points.append((cost, quality))
        
        if len(points) < 2:
            return 1.0 if points else 0.0
        
        # Sort by cost
        points.sort(key=lambda p: p[0])
        
        # Find Pareto frontier (maximize quality, minimize cost)
        pareto = []
        max_quality = -1
        for cost, quality in points:
            if quality > max_quality:
                pareto.append((cost, quality))
                max_quality = quality
        
        # Coverage = pareto points / total points
        return round(len(pareto) / len(points), 3)
