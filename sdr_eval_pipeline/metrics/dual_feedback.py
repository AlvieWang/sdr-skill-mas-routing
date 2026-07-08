"""
SDR Evaluation Pipeline - Metric E: Dual Feedback

Measures pre-execution and post-execution feedback quality,
inspired by ToolTree's dual feedback mechanism.

Metrics:
- Pre-execution Skill Match: Accuracy of pre-execution predictions
- Post-execution Skill Contribution: Skill-level contribution scores
- Feedback Gap: Difference between pre and post execution scores
- Skill-level Plan F1: Quality of skill sequence planning
- Skill-level Exec F1: Quality of skill execution
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

from core.types import Trajectory, MetricResult, EvaluationConfig, StepResult


class DualFeedbackMetrics:
    """Category E: Dual feedback (pre + post execution) metrics."""
    
    CATEGORY = "E"
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(self, trajectories: List[Trajectory]) -> List[MetricResult]:
        results = []
        
        # E1: Pre-execution Skill Match
        pre_match = self._pre_execution_match(trajectories)
        results.append(MetricResult(
            name="Pre-execution Skill Match",
            value=pre_match,
            category=self.CATEGORY,
            description="Correlation between pre-execution predictions and actual outcomes",
            baseline_value=None,  # Rubar has no pre-execution prediction
            detail={"source": "ToolTree r_pre"},
        ))
        
        # E2: Post-execution Skill Contribution
        post_contribution = self._post_execution_contribution(trajectories)
        results.append(MetricResult(
            name="Post-execution Skill Contribution",
            value=post_contribution,
            category=self.CATEGORY,
            description="Average skill-level contribution score after execution",
            baseline_value=None,
            detail={"source": "ToolTree r_post"},
        ))
        
        # E3: Feedback Gap
        gap = self._feedback_gap(trajectories)
        results.append(MetricResult(
            name="Feedback Gap",
            value=gap,
            category=self.CATEGORY,
            description="Absolute difference between pre and post execution scores",
            baseline_value=1.0,  # No pre-execution = gap = 1.0
            detail={"source": "ToolTree (ablation -7.50)"},
        ))
        
        # E4: Skill-level Plan F1
        plan_f1 = self._skill_plan_f1(trajectories)
        results.append(MetricResult(
            name="Skill-level Plan F1",
            value=plan_f1,
            category=self.CATEGORY,
            description="F1 of planned skill sequence vs actual required skills",
            baseline_value=None,
            detail={"source": "ToolTree Plan F1"},
        ))
        
        # E5: Skill-level Exec F1
        exec_f1 = self._skill_exec_f1(trajectories)
        results.append(MetricResult(
            name="Skill-level Exec F1",
            value=exec_f1,
            category=self.CATEGORY,
            description="F1 of executed skill outcomes vs expected outcomes",
            baseline_value=None,
            detail={"source": "ToolTree Exec F1"},
        ))
        
        return results
    
    def _pre_execution_match(self, trajectories: List[Trajectory]) -> float:
        """Correlation between pre-execution predictions and actual outcomes."""
        pre_scores = []
        actual_outcomes = []
        
        for traj in trajectories:
            for result in traj.results:
                pre_scores.append(result.pre_execution_score)
                actual_outcomes.append(1.0 if result.success else 0.0)
        
        if len(pre_scores) < 2:
            return 0.0
        
        # Binary accuracy of pre-execution predictions
        correct = 0
        for pre, actual in zip(pre_scores, actual_outcomes):
            predicted = 1.0 if pre > 0.5 else 0.0
            if predicted == actual:
                correct += 1
        
        return round(correct / len(pre_scores), 3)
    
    def _post_execution_contribution(self, trajectories: List[Trajectory]) -> float:
        """Average post-execution contribution score."""
        scores = []
        for traj in trajectories:
            for result in traj.results:
                scores.append(result.post_execution_score)
        
        return round(np.mean(scores), 3) if scores else 0.0
    
    def _feedback_gap(self, trajectories: List[Trajectory]) -> float:
        """Average absolute difference between pre and post execution scores."""
        gaps = []
        for traj in trajectories:
            for result in traj.results:
                gap = abs(result.pre_execution_score - result.post_execution_score)
                gaps.append(gap)
        
        return round(np.mean(gaps), 3) if gaps else 1.0
    
    def _skill_plan_f1(self, trajectories: List[Trajectory]) -> float:
        """F1 of planned skill sequence vs ground-truth required skills."""
        all_precisions = []
        all_recalls = []
        
        for traj in trajectories:
            for dec, ctx in zip(traj.decisions, traj.steps):
                if not ctx.gt_skills:
                    continue
                
                predicted = set(dec.predicted_skills)
                gt = set(ctx.gt_skills)
                
                if predicted and gt:
                    tp = len(predicted & gt)
                    precision = tp / len(predicted)
                    recall = tp / len(gt)
                    all_precisions.append(precision)
                    all_recalls.append(recall)
        
        if not all_precisions or not all_recalls:
            return 0.0
        
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        
        if avg_precision + avg_recall == 0:
            return 0.0
        
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        return round(f1, 3)
    
    def _skill_exec_f1(self, trajectories: List[Trajectory]) -> float:
        """F1 of executed skill outcomes vs expected success."""
        skill_outcomes = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for traj in trajectories:
            for result in traj.results:
                for skill_name in result.skills_involved:
                    if result.success:
                        skill_outcomes[skill_name]["tp"] += 1
                    else:
                        skill_outcomes[skill_name]["fn"] += 1
        
        f1_values = []
        for skill, counts in skill_outcomes.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                f1_values.append(f1)
        
        return round(np.mean(f1_values), 3) if f1_values else 0.0
