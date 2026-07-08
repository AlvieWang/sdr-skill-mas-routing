"""
SDR Evaluation Pipeline - Core Type Definitions

Defines the fundamental data structures used throughout the evaluation pipeline:
- Skill: A unit of capability that can be routed, measured, and transferred
- StepContext: The context of a single execution step
- RoutingDecision: The output of a router for a given step
- Trajectory: A complete task execution trace
- EvaluationResult: The result of evaluating a single step
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


# ============================================================
# Enumerations
# ============================================================

class ModelId(str, Enum):
    """Model identifiers in the 4B/7B/14B pool."""
    M_T = "4B"   # Tiny  - Qwen3-4B
    M_S = "7B"   # Small - Qwen2.5-7B
    M_M = "14B"  # Medium- Qwen2.5-14B


class SkillType(str, Enum):
    """Skill categories mapped from Rubar's 7-dimension matrix + SWE/WebArena extensions."""
    RETRIEVE = "retrieve"        # Search, grep, file lookup
    CODE_GEN = "code_gen"        # Code generation
    DEBUG = "debug"             # Debugging and error fixing
    PLAN = "plan"                # Planning and architecture
    VERIFY = "verify"            # Verification and validation
    TEST = "test"                # Test generation and execution
    DOC = "doc"                  # Documentation understanding
    # WebArena extensions
    NAVIGATE = "navigate"        # Web navigation
    FORM_FILL = "form_fill"      # Form filling
    DATA_EXTRACT = "data_extract"  # Data extraction from web


class FeedbackType(str, Enum):
    """Types of feedback signals."""
    PRE_EXECUTION = "pre"    # Forward-looking prediction
    POST_EXECUTION = "post"  # Backward-looking evaluation


class FailureSource(str, Enum):
    """Failure attribution sources (from PawBench 6-dimensional scheme)."""
    MODEL_REASONING = "model_reasoning"
    TOOL_MISSING = "tool_missing"
    SKILL_DISCOVERY_WEAK = "skill_discovery_weak"
    WORKSPACE_PERCEPTION = "workspace_perception"
    NETWORK_FRAGILE = "network_fragile"
    COMPLETION_CHECK_LOOSE = "completion_check_loose"


# ============================================================
# Core Data Structures
# ============================================================

@dataclass
class Skill:
    """
    A skill is a unit of capability that can be routed, measured, and transferred.
    
    Inspired by SkillOrchestra's Beta-Bernoulli modeling and SkillOpt's text-space optimization.
    Each skill tracks:
    - Capability profile: per-model success probability (Beta-Bernoulli posterior)
    - Cost profile: per-model token cost and latency
    - Evolution state: whether the skill is stable, pending split, or newly created
    """
    name: str
    description: str
    when_to_apply: str
    skill_type: SkillType
    
    # Beta-Bernoulli posterior parameters for each model
    # alpha = successes + 1, beta = failures + 1
    capability_profile: dict[str, tuple[float, float]] = field(default_factory=dict)
    
    # Cost profile: {model_id: {"avg_tokens": int, "avg_latency_ms": float}}
    cost_profile: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Evolution tracking
    creation_step: int = 0
    last_modified_step: int = 0
    split_count: int = 0
    merge_count: int = 0
    
    # Quality gate (from SkillOpt)
    selection_gate_score: float = 0.0
    rejection_count: int = 0

    def success_prob(self, model_id: str) -> float:
        """Posterior mean of Beta(alpha, beta) = alpha / (alpha + beta)."""
        if model_id not in self.capability_profile:
            return 0.5  # Uninformative prior
        alpha, beta = self.capability_profile[model_id]
        return alpha / (alpha + beta)

    def update_posterior(self, model_id: str, success: bool) -> None:
        """Update Beta-Bernoulli posterior after observing an outcome."""
        if model_id not in self.capability_profile:
            self.capability_profile[model_id] = (1.0, 1.0)  # Uniform prior
        alpha, beta = self.capability_profile[model_id]
        if success:
            self.capability_profile[model_id] = (alpha + 1, beta)
        else:
            self.capability_profile[model_id] = (alpha, beta + 1)


@dataclass
class StepContext:
    """
    The context of a single execution step in a task trajectory.
    Contains all information the router needs to make a decision.
    """
    step_id: int
    task_id: str
    query: str
    history: list[dict] = field(default_factory=list)
    
    # Ground-truth skill annotations (for evaluation only)
    gt_skills: list[str] = field(default_factory=list)
    gt_model: Optional[str] = None  # Oracle model choice
    
    # Context features (from Rubar/OpenSquilla)
    token_count: int = 0
    has_code: bool = False
    programming_lang: Optional[str] = None
    complexity_score: float = 0.0
    budget_remaining: float = 1.0
    previous_step_failed: bool = False
    
    # Benchmark metadata
    benchmark: str = "swe_bench"  # or "webarena"


@dataclass
class RoutingDecision:
    """
    The output of a router for a given step.
    Captures both the skill identification and model selection.
    """
    step_id: int
    # Skill identification (SDR layer)
    predicted_skills: list[str] = field(default_factory=list)
    skill_scores: dict[str, float] = field(default_factory=dict)  # skill_name -> confidence
    
    # Model selection (Rubar/RL-PER layer)
    selected_model: str = ""
    model_distribution: dict[str, float] = field(default_factory=dict)
    
    # Routing metadata
    uncertainty: float = 0.0
    router_type: str = "sdr"  # "rubar", "rl_per", "sdr"
    latency_ms: float = 0.0


@dataclass
class StepResult:
    """The result of executing a single step."""
    step_id: int
    success: bool
    quality_score: float  # 0.0 - 1.0
    token_cost: int
    latency_ms: float
    model_used: str
    skills_involved: list[str] = field(default_factory=list)
    
    # Dual feedback (from ToolTree)
    pre_execution_score: float = 0.0   # Predicted usefulness
    post_execution_score: float = 0.0  # Actual contribution
    
    # Failure attribution (from PawBench)
    failure_source: Optional[FailureSource] = None
    failure_detail: str = ""


@dataclass
class Trajectory:
    """A complete task execution trace."""
    task_id: str
    benchmark: str
    steps: list[StepContext] = field(default_factory=list)
    decisions: list[RoutingDecision] = field(default_factory=list)
    results: list[StepResult] = field(default_factory=list)
    task_success: bool = False
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    
    def add_step(self, context: StepContext, decision: RoutingDecision, result: StepResult) -> None:
        self.steps.append(context)
        self.decisions.append(decision)
        self.results.append(result)
        self.total_tokens += result.token_cost
        self.total_latency_ms += result.latency_ms


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""
    # Model pool
    models: list[str] = field(default_factory=lambda: ["4B", "7B", "14B"])
    model_costs: dict[str, float] = field(default_factory=lambda: {"4B": 1.0, "7B": 2.5, "14B": 6.0})
    model_latencies: dict[str, float] = field(default_factory=lambda: {"4B": 15.0, "7B": 25.0, "14B": 45.0})
    
    # Skill retrieval
    top_k_skills: int = 10
    
    # Routing collapse threshold
    collapse_threshold: float = 0.95  # If single model >95% → collapse
    
    # Dual feedback
    feedback_gap_threshold: float = 0.15
    
    # Pareto frontier
    pareto_points: int = 50
    
    # Skill evolution
    split_variance_threshold: float = 0.15
    merge_indistinguishable_threshold: float = 0.05
    
    # Output
    output_dir: str = "output"
    verbose: bool = True


@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: float
    category: str  # A-F
    description: str
    baseline_value: Optional[float] = None  # For comparison
    detail: dict[str, Any] = field(default_factory=dict)
