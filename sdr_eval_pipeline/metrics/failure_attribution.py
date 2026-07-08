"""
SDR Evaluation Pipeline - Metric F: Failure Attribution

Measures skill-level failure attribution and diagnostic quality,
inspired by PawBench's 6-dimensional failure analysis.

Metrics:
- Skill-level Failure Attribution: Fraction of failures attributable to specific skills
- Skill Discovery Failure Rate: Failures due to weak skill discovery
- Skill-Model Mismatch Rate: Failures due to skill-model capability mismatch
- Harness-Skill Interaction Score: Model x harness performance on specific skills
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List

import numpy as np

from core.types import (
    Trajectory, MetricResult, EvaluationConfig,
    StepResult, FailureSource
)


class FailureAttributionMetrics:
    """Category F: Failure attribution and diagnostic metrics."""
    
    CATEGORY = "F"
    
    # PawBench capability scores (for reference)
    PAWBENCH_BASELINE = {
        "Skill_Use": 47.2,  # Most difficult capability
        "Tool_Use": 58.5,
        "Planning": 62.3,
        "Self_Verification": 55.1,
        "Logic_Math": 64.7,
        "Code": 60.2,
    }
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate(self, trajectories: List[Trajectory]) -> List[MetricResult]:
        results = []
        
        # F1: Skill-level Failure Attribution
        attribution_rate = self._failure_attribution_rate(trajectories)
        results.append(MetricResult(
            name="Skill-level Failure Attribution",
            value=attribution_rate,
            category=self.CATEGORY,
            description="Fraction of failures attributable to specific skills",
            baseline_value=0.0,  # Rubar/RL-PER have no attribution
            detail={"source": "PawBench 6-dimensional"},
        ))
        
        # F2: Skill Discovery Failure Rate
        discovery_fail = self._skill_discovery_failure_rate(trajectories)
        results.append(MetricResult(
            name="Skill Discovery Failure Rate",
            value=discovery_fail,
            category=self.CATEGORY,
            description="Fraction of failures caused by weak skill discovery",
            baseline_value=None,
            detail={"source": "PawBench (Skill_Use=47.2)"},
        ))
        
        # F3: Skill-Model Mismatch Rate
        mismatch = self._skill_model_mismatch_rate(trajectories)
        results.append(MetricResult(
            name="Skill-Model Mismatch Rate",
            value=mismatch,
            category=self.CATEGORY,
            description="Failures where skill requirements didn't match model capabilities",
            baseline_value=None,
            detail={"source": "PawBench"},
        ))
        
        # F4: Harness-Skill Interaction Score
        interaction = self._harness_skill_interaction(trajectories)
        results.append(MetricResult(
            name="Harness-Skill Interaction Score",
            value=interaction,
            category=self.CATEGORY,
            description="Model x harness performance variation on specific skills",
            baseline_value=11.5,  # PawBench harness gap max
            detail={"source": "PawBench Harness Gap"},
        ))
        
        return results
    
    def _failure_attribution_rate(self, trajectories: List[Trajectory]) -> float:
        """Fraction of failures that can be attributed to specific skills."""
        total_failures = 0
        attributed_failures = 0
        
        for traj in trajectories:
            for result in traj.results:
                if not result.success:
                    total_failures += 1
                    if result.failure_source is not None:
                        attributed_failures += 1
                    elif result.skills_involved:
                        # Can attribute to skills involved
                        attributed_failures += 1
        
        return round(attributed_failures / total_failures, 3) if total_failures > 0 else 0.0
    
    def _skill_discovery_failure_rate(self, trajectories: List[Trajectory]) -> float:
        """Failures caused by weak skill discovery."""
        total_failures = 0
        discovery_failures = 0
        
        for traj in trajectories:
            for result in traj.results:
                if not result.success:
                    total_failures += 1
                    if result.failure_source == FailureSource.SKILL_DISCOVERY_WEAK:
                        discovery_failures += 1
        
        return round(discovery_failures / total_failures, 3) if total_failures > 0 else 0.0
    
    def _skill_model_mismatch_rate(self, trajectories: List[Trajectory]) -> float:
        """Failures where skill needs didn't match model capabilities."""
        mismatch_count = 0
        total_steps = 0
        
        for traj in trajectories:
            for result in traj.results:
                total_steps += 1
                if not result.success and result.skills_involved:
                    # Check if model was mismatched (e.g., 4B used for debug skill)
                    for skill_name in result.skills_involved:
                        if "debug" in skill_name.lower() and result.model_used == "4B":
                            mismatch_count += 1
                            break
                        elif "plan" in skill_name.lower() and result.model_used == "4B":
                            mismatch_count += 1
                            break
        
        return round(mismatch_count / total_steps, 3) if total_steps > 0 else 0.0
    
    def _harness_skill_interaction(self, trajectories: List[Trajectory]) -> float:
        """Model x skill performance variation (harness gap)."""
        skill_model_performance = defaultdict(lambda: {"success": 0, "total": 0})
        
        for traj in trajectories:
            for result in traj.results:
                for skill_name in result.skills_involved:
                    key = f"{result.model_used}_{skill_name}"
                    skill_model_performance[key]["total"] += 1
                    if result.success:
                        skill_model_performance[key]["success"] += 1
        
        # Calculate max variation across models for same skill
        skill_rates = defaultdict(dict)
        for key, data in skill_model_performance.items():
            parts = key.split("_", 1)
            if len(parts) == 2:
                model, skill = parts
                rate = data["success"] / data["total"] if data["total"] > 0 else 0
                skill_rates[skill][model] = rate
        
        max_gaps = []
        for skill, model_rates in skill_rates.items():
            if len(model_rates) >= 2:
                rates = list(model_rates.values())
                gap = max(rates) - min(rates)
                max_gaps.append(gap * 100)  # Convert to percentage points
        
        return round(np.mean(max_gaps), 1) if max_gaps else 0.0
