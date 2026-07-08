"""
SDR Evaluation Pipeline - Metric D: Skill Evolution & Quality

Tracks skill lifecycle events (split, merge, creation) and quality metrics,
inspired by SkillOpt's Selection Gate and SkillOrchestra's Skill Refinement.

Metrics:
- Skill Refinement Rate: Splits + merges per unit time
- Skill Stability: Variance of capability profile over time
- Skill Coverage: Fraction of skill types with registered skills
- Skill Quality Gate Pass Rate: Fraction of new skills passing validation
- Skill Velocity: Steps to convergence for new skills
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np

from core.types import Trajectory, MetricResult, EvaluationConfig
from core.skill_registry import SkillRegistry


class SkillEvolutionMetrics:
    """Category D: Skill evolution and quality metrics."""
    
    CATEGORY = "D"
    
    def __init__(self, config: EvaluationConfig, skill_registry: SkillRegistry):
        self.config = config
        self.skill_registry = skill_registry
    
    def evaluate(self, trajectories: List[Trajectory]) -> List[MetricResult]:
        results = []
        
        total_steps = sum(len(t.steps) for t in trajectories)
        
        # D1: Skill Refinement Rate
        evolution = self.skill_registry.get_evolution_summary()
        rate = (evolution["splits"] + evolution["merges"]) / max(1, total_steps / 100)
        results.append(MetricResult(
            name="Skill Refinement Rate",
            value=rate,
            category=self.CATEGORY,
            description="Skill splits + merges per 100 steps",
            baseline_value=None,  # Rubar/RL-PER don't track this
            detail={
                "source": "SkillOrchestra + SkillOpt",
                "splits": evolution["splits"],
                "merges": evolution["merges"],
            },
        ))
        
        # D2: Skill Stability (variance of Beta-Bernoulli posteriors)
        stability = self._skill_stability()
        results.append(MetricResult(
            name="Skill Stability",
            value=stability,
            category=self.CATEGORY,
            description="1 - average variance of skill capability profiles (1.0=stable)",
            baseline_value=None,
            detail={"source": "SkillOrchestra Beta-Bernoulli"},
        ))
        
        # D3: Skill Coverage
        coverage = self.skill_registry.get_skill_coverage()
        results.append(MetricResult(
            name="Skill Coverage",
            value=coverage,
            category=self.CATEGORY,
            description="Fraction of skill types with registered skills",
            baseline_value=0.70,  # Rubar 7/10 dimensions
            detail={"source": "SkillRouter (80K skills)"},
        ))
        
        # D4: Skill Quality Gate Pass Rate
        pass_rate = self._quality_gate_pass_rate()
        results.append(MetricResult(
            name="Skill Quality Gate Pass Rate",
            value=pass_rate,
            category=self.CATEGORY,
            description="Fraction of new skills passing Selection Gate validation",
            baseline_value=None,
            detail={"source": "SkillOpt Selection Gate"},
        ))
        
        # D5: Skill Velocity (convergence speed)
        velocity = self._skill_velocity(trajectories)
        results.append(MetricResult(
            name="Skill Velocity",
            value=velocity,
            category=self.CATEGORY,
            description="Average steps for new skill to converge (lower=faster)",
            baseline_value=None,
            detail={"source": "SkillOpt (2-4 epoch convergence)"},
        ))
        
        return results
    
    def _skill_stability(self) -> float:
        """1 - average variance of capability profiles across models."""
        variances = []
        for skill in self.skill_registry.skills.values():
            probs = [skill.success_prob(m) for m in self.config.models]
            if len(probs) > 1:
                variances.append(np.var(probs))
        
        if not variances:
            return 1.0
        
        avg_var = np.mean(variances)
        return round(1.0 - avg_var, 3)
    
    def _quality_gate_pass_rate(self) -> float:
        """Fraction of skills that passed the Selection Gate."""
        if not self.skill_registry.skills:
            return 0.0
        
        passed = sum(
            1 for s in self.skill_registry.skills.values()
            if s.selection_gate_score > 0.5
        )
        total = len(self.skill_registry.skills)
        return round(passed / total, 3) if total > 0 else 0.0
    
    def _skill_velocity(self, trajectories: List[Trajectory]) -> float:
        """Average steps for a skill to converge (stabilize posterior)."""
        # Track when each skill's posterior stabilizes
        skill_updates = defaultdict(list)
        
        for traj in trajectories:
            for i, result in enumerate(traj.results):
                for skill_name in result.skills_involved:
                    skill_updates[skill_name].append((i, result.success))
        
        convergence_steps = []
        for skill_name, updates in skill_updates.items():
            if len(updates) < 5:
                continue
            
            # Find step where success rate stabilizes
            window = 3
            for i in range(window, len(updates)):
                recent = [u[1] for u in updates[i-window:i]]
                recent_rate = np.mean(recent)
                if abs(recent_rate - 0.5) > 0.2:  # Converged (not 50/50)
                    convergence_steps.append(updates[i][0])
                    break
        
        return round(np.mean(convergence_steps), 1) if convergence_steps else 0.0
