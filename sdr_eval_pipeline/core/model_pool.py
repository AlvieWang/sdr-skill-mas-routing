"""
SDR Evaluation Pipeline - Model Pool Manager

Manages the 4B/7B/14B model pool with capability and cost profiles.
Based on Rubar and RL-PER's model pool design.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .types import ModelId, EvaluationConfig


@dataclass
class ModelProfile:
    """Profile for a single model in the pool."""
    model_id: str
    params: str  # "4B", "7B", "14B"
    cost_ratio: float  # Relative cost (4B=1.0, 7B=2.5, 14B=6.0)
    base_latency_ms: float
    pass_at_1: float  # Base Pass@1 on SWE-bench
    
    # Rubar skill matrix (7 dimensions from the formalization)
    skill_matrix: dict[str, float] = field(default_factory=dict)
    
    # RL-PER capability
    capability_value: float = 0.0
    cost_coefficient: float = 0.0


class ModelPool:
    """
    Manages the 4B/7B/14B model pool.
    
    Provides:
    - Model selection based on skill requirements
    - Cost calculation
    - Capability lookup (Rubar's 7-dimension matrix)
    """
    
    # Rubar's 7-dimension skill matrix (from the formalization document)
    RUBAR_SKILL_MATRIX = {
        "4B": {
            "retrieve": 0.88, "code_gen": 0.55, "debug": 0.45,
            "plan": 0.50, "verify": 0.90, "test": 0.60, "doc": 0.70,
        },
        "7B": {
            "retrieve": 0.91, "code_gen": 0.87, "debug": 0.82,
            "plan": 0.80, "verify": 0.92, "test": 0.85, "doc": 0.86,
        },
        "14B": {
            "retrieve": 0.93, "code_gen": 0.94, "debug": 0.91,
            "plan": 0.90, "verify": 0.94, "test": 0.92, "doc": 0.91,
        },
    }
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.profiles: dict[str, ModelProfile] = {}
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize model profiles from Rubar/RL-PER formalization."""
        specs = [
            ("4B", "4B", 1.0, 15.0, 0.375),   # Qwen3-4B
            ("7B", "7B", 2.5, 25.0, 0.475),   # Qwen2.5-7B
            ("14B", "14B", 6.0, 45.0, 0.545),  # Qwen2.5-14B
        ]
        
        for model_id, params, cost, latency, pass1 in specs:
            profile = ModelProfile(
                model_id=model_id,
                params=params,
                cost_ratio=cost,
                base_latency_ms=latency,
                pass_at_1=pass1,
                skill_matrix=self.RUBAR_SKILL_MATRIX.get(model_id, {}),
                capability_value=pass1,
                cost_coefficient=cost * 0.1,
            )
            self.profiles[model_id] = profile
    
    def get_model(self, model_id: str) -> Optional[ModelProfile]:
        return self.profiles.get(model_id)
    
    def get_all_models(self) -> list[ModelProfile]:
        return list(self.profiles.values())
    
    def get_skill_capability(self, model_id: str, skill_name: str) -> float:
        """Get model's capability for a specific skill (from Rubar matrix)."""
        profile = self.profiles.get(model_id)
        if profile is None:
            return 0.5
        return profile.skill_matrix.get(skill_name, 0.5)
    
    def get_cost(self, model_id: str, token_count: int = 1000) -> float:
        """Estimate cost for a model call."""
        profile = self.profiles.get(model_id)
        if profile is None:
            return 0.0
        return profile.cost_ratio * token_count / 1000.0
    
    def get_latency(self, model_id: str) -> float:
        """Get base latency for a model."""
        profile = self.profiles.get(model_id)
        if profile is None:
            return 30.0
        return profile.base_latency_ms
    
    def select_best_model_for_skill(
        self,
        skill_name: str,
        budget_constraint: float = 1.0,
        min_capability: float = 0.0
    ) -> str:
        """
        Select the best model for a given skill under budget constraint.
        
        Uses Rubar's Specificity priority: choose the cheapest model
        that meets the minimum capability threshold.
        """
        best_model = "7B"  # Default
        best_cost_effectiveness = -1.0
        
        for model_id, profile in self.profiles.items():
            capability = profile.skill_matrix.get(skill_name, 0.5)
            if capability < min_capability:
                continue
            
            cost = profile.cost_ratio
            if cost > budget_constraint * 10:
                continue
            
            # Cost-effectiveness: capability / cost
            ce = capability / cost
            if ce > best_cost_effectiveness:
                best_cost_effectiveness = ce
                best_model = model_id
        
        return best_model
