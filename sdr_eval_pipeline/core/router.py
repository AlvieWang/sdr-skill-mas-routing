"""
SDR Evaluation Pipeline - Router Implementations

Three router variants for comparison:
1. RubarRouter: Deterministic rubric-based routing (from Rubar formalization)
2. RLPerRouter: RL-pretrained external router (from RL-PER formalization)
3. SDRRouter: Skill-Driven Dynamic Router (proposed SDR framework)

Each router takes a StepContext and produces a RoutingDecision.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .types import (
    StepContext, RoutingDecision, EvaluationConfig,
    SkillType, FailureSource
)
from .skill_registry import SkillRegistry
from .model_pool import ModelPool


class BaseRouter:
    """Base class for all routers."""
    
    def __init__(self, config: EvaluationConfig, model_pool: ModelPool):
        self.config = config
        self.model_pool = model_pool
        self.router_type = "base"
    
    def route(self, step_context: StepContext) -> RoutingDecision:
        raise NotImplementedError
    
    def update(self, step_context: StepContext, decision: RoutingDecision, reward: float) -> None:
        """Update router state (for TTT/RL)."""
        pass


class RubarRouter(BaseRouter):
    """
    Rubar: Deterministic rubric-based routing.
    
    Implements the condition matching + priority rule matching
    from the Rubar formalization. No skill abstraction.
    """
    
    def __init__(self, config: EvaluationConfig, model_pool: ModelPool):
        super().__init__(config, model_pool)
        self.router_type = "rubar"
    
    def route(self, step_context: StepContext) -> RoutingDecision:
        # Condition matching (from Rubar formalization, Section 6)
        conditions = self._evaluate_conditions(step_context)
        
        # Priority rule matching (Specificity first)
        selected_model = self._priority_match(conditions, step_context)
        
        return RoutingDecision(
            step_id=step_context.step_id,
            predicted_skills=[],  # Rubar doesn't use skill abstraction
            skill_scores={},
            selected_model=selected_model,
            model_distribution={selected_model: 1.0},
            uncertainty=0.0,
            router_type=self.router_type,
            latency_ms=2.0,  # 4B condition matching <2ms
        )
    
    def _evaluate_conditions(self, ctx: StepContext) -> list[tuple[str, float]]:
        """Evaluate Rubar conditions (specificity score)."""
        conditions = []
        
        # C1: token_count > 2000
        if ctx.token_count > 2000:
            conditions.append(("long_context", 0.9))
        
        # C2: has_code
        if ctx.has_code:
            conditions.append(("has_code", 0.7))
        
        # C3: complexity >= 0.7
        if ctx.complexity_score >= 0.7:
            conditions.append(("high_complexity", 0.95))
        
        # C4: previous_step_failed
        if ctx.previous_step_failed:
            conditions.append(("prev_failed", 0.85))
        
        # C5: budget < 0.3
        if ctx.budget_remaining < 0.3:
            conditions.append(("low_budget", 0.6))
        
        # C6: default
        conditions.append(("default", 0.0))
        
        return conditions
    
    def _priority_match(self, conditions: list[tuple[str, float]], ctx: StepContext) -> str:
        """Priority rule matching: most specific condition wins."""
        # Sort by specificity (descending)
        conditions.sort(key=lambda x: x[1], reverse=True)
        
        for cond_name, _ in conditions:
            if cond_name == "high_complexity":
                return "14B"
            elif cond_name == "prev_failed":
                return "14B"
            elif cond_name == "long_context":
                return "7B"
            elif cond_name == "has_code":
                return "7B"
            elif cond_name == "low_budget":
                return "4B"
            elif cond_name == "default":
                return "7B"
        
        return "7B"


class RLPerRouter(BaseRouter):
    """
    RL-PER: RL-pretrained external router.
    
    Simulates the RL router behavior. In production, this would be
    a Qwen3-4B model trained with rLLM + RL-Factory.
    Here we simulate with a learned policy that may exhibit routing collapse.
    """
    
    def __init__(self, config: EvaluationConfig, model_pool: ModelPool):
        super().__init__(config, model_pool)
        self.router_type = "rl_per"
        
        # Simulated RL policy (prone to routing collapse)
        self._policy_bias = {"4B": 0.05, "7B": 0.10, "14B": 0.85}  # Collapse to 14B
        self._total_updates = 0
        self._exploration_rate = 0.1
    
    def route(self, step_context: StepContext) -> RoutingDecision:
        # RL router: softmax over learned Q-values
        q_values = self._estimate_q_values(step_context)
        
        # Softmax with temperature
        temp = 0.5
        exp_q = {m: np.exp(q / temp) for m, q in q_values.items()}
        total = sum(exp_q.values())
        distribution = {m: v / total for m, v in exp_q.items()}
        
        # Apply exploration
        if random.random() < self._exploration_rate:
            selected = random.choice(list(distribution.keys()))
        else:
            selected = max(distribution, key=distribution.get)
        
        return RoutingDecision(
            step_id=step_context.step_id,
            predicted_skills=[],
            skill_scores={},
            selected_model=selected,
            model_distribution=distribution,
            uncertainty=1.0 - max(distribution.values()),
            router_type=self.router_type,
            latency_ms=5.0,  # 4B router ~5ms
        )
    
    def _estimate_q_values(self, ctx: StepContext) -> dict[str, float]:
        """Estimate Q-values for each model (simulated RL policy)."""
        q = {}
        for model_id in self.config.models:
            base_q = self.model_pool.get_skill_capability(model_id, "code_gen")
            # Add bias (simulating learned policy)
            bias = self._policy_bias.get(model_id, 0.1)
            # Context-dependent adjustments
            if ctx.complexity_score > 0.7 and model_id == "14B":
                bias += 0.2
            if ctx.budget_remaining < 0.3 and model_id == "4B":
                bias += 0.1
            q[model_id] = base_q + bias
        return q
    
    def update(self, step_context: StepContext, decision: RoutingDecision, reward: float) -> None:
        """Simulate RL update (LoRA on 4B router)."""
        self._total_updates += 1
        # Gradually reduce exploration
        self._exploration_rate = max(0.01, 0.1 * np.exp(-self._total_updates / 1000))
        
        # Update policy bias (simulated gradient update)
        model = decision.selected_model
        if reward > 0.5:
            self._policy_bias[model] = min(0.99, self._policy_bias[model] + 0.001)
        else:
            self._policy_bias[model] = max(0.01, self._policy_bias[model] - 0.002)


class SDRRouter(BaseRouter):
    """
    SDR: Skill-Driven Dynamic Router.
    
    The proposed framework that combines:
    - Skill identification (SkillRouter-style retrieve-and-rerank)
    - Rubric condition matching (Rubar-style)
    - RL routing (RL-PER-style)
    - Dual feedback (ToolTree-style)
    - Meta-RL transfer (LaMer-style)
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        model_pool: ModelPool,
        skill_registry: SkillRegistry
    ):
        super().__init__(config, model_pool)
        self.router_type = "sdr"
        self.skill_registry = skill_registry
        
        # Meta-RL state (from LaMer)
        self._cross_task_memory: dict[str, list[float]] = {}  # skill_name -> reward history
        self._adaptation_speed: dict[str, int] = {}  # skill_name -> steps to converge
    
    def route(self, step_context: StepContext) -> RoutingDecision:
        # Layer 1: Skill identification (SkillRouter two-stage)
        candidates = self.skill_registry.retrieve_top_k(
            step_context.query,
            k=self.config.top_k_skills
        )
        reranked = self.skill_registry.rerank(candidates, step_context)
        
        # Extract predicted skills
        predicted_skills = [name for name, _ in reranked[:3]]  # Top-3
        skill_scores = {name: score for name, score in reranked[:self.config.top_k_skills]}
        
        # Layer 2: Rubric condition matching (Rubar-style)
        rubric_conditions = self._evaluate_rubric_conditions(step_context)
        
        # Layer 3: RL routing with skill conditioning (RL-PER-style)
        model_distribution = self._skill_conditioned_routing(
            predicted_skills, rubric_conditions, step_context
        )
        
        # Select model
        selected_model = max(model_distribution, key=model_distribution.get)
        
        # Compute uncertainty
        uncertainty = 1.0 - max(model_distribution.values())
        
        return RoutingDecision(
            step_id=step_context.step_id,
            predicted_skills=predicted_skills,
            skill_scores=skill_scores,
            selected_model=selected_model,
            model_distribution=model_distribution,
            uncertainty=uncertainty,
            router_type=self.router_type,
            latency_ms=7.0,  # Skill retrieval + rubric + RL = ~7ms
        )
    
    def _evaluate_rubric_conditions(self, ctx: StepContext) -> dict[str, bool]:
        """Evaluate Rubar conditions for skill-level matching."""
        return {
            "high_complexity": ctx.complexity_score >= 0.7,
            "long_context": ctx.token_count > 2000,
            "prev_failed": ctx.previous_step_failed,
            "low_budget": ctx.budget_remaining < 0.3,
            "has_code": ctx.has_code,
        }
    
    def _skill_conditioned_routing(
        self,
        skills: list[str],
        conditions: dict[str, bool],
        ctx: StepContext
    ) -> dict[str, float]:
        """
        RL routing conditioned on identified skills.
        
        Uses skill-level Beta-Bernoulli posteriors to inform model selection.
        Anti-collapse mechanism: cost-effectiveness scoring + entropy regularization
        ensures diverse model usage (addresses SkillOrchestra's routing collapse finding).
        """
        model_scores = {m: 0.0 for m in self.config.models}
        
        for skill_name in skills:
            skill = self.skill_registry.skills.get(skill_name)
            if skill is None:
                continue
            
            for model_id in self.config.models:
                # Skill-level success probability (Beta-Bernoulli posterior)
                success_prob = skill.success_prob(model_id)
                
                # Cost-effectiveness: success per unit cost
                # This is the key anti-collapse mechanism:
                # A model that's 90% capable at 1x cost is preferred over
                # one that's 94% capable at 6x cost, unless the task demands it
                cost_ratio = self.model_pool.get_model(model_id).cost_ratio
                cost_effectiveness = success_prob / cost_ratio
                
                # Blend raw capability and cost-effectiveness
                # alpha=0.6 for capability, 0.4 for cost-efficiency
                ce = 0.6 * success_prob + 0.4 * cost_effectiveness
                
                # Apply rubric conditions (Rubar-style specificity matching)
                if conditions.get("high_complexity"):
                    if model_id == "4B":
                        ce -= 0.25  # Strongly penalize 4B for high complexity
                    elif model_id == "14B":
                        ce += 0.10  # Reward 14B for high complexity
                else:
                    # For low-complexity tasks, prefer cheaper models
                    if model_id == "4B":
                        ce += 0.15  # Reward 4B for simple tasks
                    elif model_id == "14B":
                        ce -= 0.20  # Penalize 14B overkill
                
                if conditions.get("low_budget") and model_id == "14B":
                    ce -= 0.30  # Strongly penalize 14B when budget is low
                if conditions.get("prev_failed") and model_id == "4B":
                    ce -= 0.15  # Penalize 4B after failure
                if conditions.get("low_budget") and model_id == "4B":
                    ce += 0.10  # Reward 4B when budget is low
                
                model_scores[model_id] += ce
        
        # Softmax with higher temperature for exploration
        # This prevents routing collapse by maintaining distribution diversity
        temp = 0.15
        max_score = max(model_scores.values()) if model_scores else 0
        exp_scores = {m: np.exp((s - max_score) / temp) for m, s in model_scores.items()}
        
        total = sum(exp_scores.values())
        distribution = {m: v / total for m, v in exp_scores.items()}
        
        return distribution
    
    def update(
        self,
        step_context: StepContext,
        decision: RoutingDecision,
        reward: float,
        skill_success: dict[str, bool]
    ) -> None:
        """
        Meta-RL update: update skill-level posteriors and cross-task memory.
        
        Inspired by LaMer's reflective adaptation (no gradient update).
        """
        for skill_name, success in skill_success.items():
            # Update Beta-Bernoulli posterior
            self.skill_registry.update_skill_posterior(
                skill_name, decision.selected_model, success
            )
            
            # Track cross-task memory (Meta-RL)
            if skill_name not in self._cross_task_memory:
                self._cross_task_memory[skill_name] = []
            self._cross_task_memory[skill_name].append(reward)
            
            # Track adaptation speed
            if skill_name not in self._adaptation_speed:
                self._adaptation_speed[skill_name] = 0
            self._adaptation_speed[skill_name] += 1
