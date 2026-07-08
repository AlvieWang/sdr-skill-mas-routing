"""
SDR Evaluation Pipeline - Mock Data Generator

Generates realistic mock data for testing the evaluation pipeline.
Simulates SWE-bench and WebArena task trajectories with:
- Ground-truth skill annotations
- Oracle model choices
- Realistic token costs and latencies
- Failure scenarios (from PawBench categories)
"""

from __future__ import annotations

import random
from typing import List

import numpy as np

from core.types import (
    Trajectory, StepContext, RoutingDecision, StepResult,
    Skill, SkillType, EvaluationConfig, FailureSource
)
from core.skill_registry import SkillRegistry
from core.model_pool import ModelPool


def create_skill_registry(config: EvaluationConfig) -> SkillRegistry:
    """Create a skill registry with initial skills."""
    registry = SkillRegistry(config)
    
    # SWE-bench skills (from Rubar 7-dimension matrix)
    swe_skills = [
        ("swe_retrieve", "Search codebase for relevant files and symbols using grep or search tools", "when user needs to find code or files", SkillType.RETRIEVE),
        ("swe_code_gen", "Generate new code or modify existing code to implement features", "when user asks to write or modify code", SkillType.CODE_GEN),
        ("swe_debug", "Identify and fix bugs in code through analysis and testing", "when code fails tests or produces errors", SkillType.DEBUG),
        ("swe_plan", "Plan implementation approach and design architecture", "when task requires architectural decisions", SkillType.PLAN),
        ("swe_verify", "Verify correctness of code changes through checks and validation", "when code needs validation before commit", SkillType.VERIFY),
        ("swe_test", "Write and run unit tests for code changes", "when tests need to be created or executed", SkillType.TEST),
        ("swe_doc", "Understand and update documentation for code changes", "when documentation needs updating", SkillType.DOC),
    ]
    
    # WebArena skills
    web_skills = [
        ("web_navigate", "Navigate web pages by clicking links and following URLs", "when user needs to browse web pages", SkillType.NAVIGATE),
        ("web_form_fill", "Fill out and submit web forms", "when user needs to complete online forms", SkillType.FORM_FILL),
        ("web_data_extract", "Extract structured data from web pages", "when user needs to scrape or extract information", SkillType.DATA_EXTRACT),
    ]
    
    # Register skills with Beta-Bernoulli priors from Rubar matrix
    rubar_matrix = {
        "4B": {"retrieve": 0.88, "code_gen": 0.55, "debug": 0.45, "plan": 0.50, "verify": 0.90, "test": 0.60, "doc": 0.70},
        "7B": {"retrieve": 0.91, "code_gen": 0.87, "debug": 0.82, "plan": 0.80, "verify": 0.92, "test": 0.85, "doc": 0.86},
        "14B": {"retrieve": 0.93, "code_gen": 0.94, "debug": 0.91, "plan": 0.90, "verify": 0.94, "test": 0.92, "doc": 0.91},
    }
    
    for name, desc, when, skill_type in swe_skills + web_skills:
        skill = Skill(
            name=name,
            description=desc,
            when_to_apply=when,
            skill_type=skill_type,
        )
        
        # Set Beta-Bernoulli priors based on Rubar matrix
        skill_type_key = skill_type.value
        for model_id in config.models:
            base_prob = rubar_matrix.get(model_id, {}).get(skill_type_key, 0.5)
            # Convert to Beta parameters: alpha = prob * 10, beta = (1-prob) * 10
            alpha = max(1.0, base_prob * 10)
            beta = max(1.0, (1 - base_prob) * 10)
            skill.capability_profile[model_id] = (alpha, beta)
            
            # Set cost profile
            cost_map = {"4B": {"avg_tokens": 800, "avg_latency_ms": 15},
                       "7B": {"avg_tokens": 1500, "avg_latency_ms": 25},
                       "14B": {"avg_tokens": 3000, "avg_latency_ms": 45}}
            skill.cost_profile[model_id] = cost_map.get(model_id, {"avg_tokens": 1000, "avg_latency_ms": 30})
        
        skill.selection_gate_score = 0.85  # Passed validation
        registry.register_skill(skill)
    
    return registry


def generate_mock_trajectories(
    n_tasks: int = 20,
    benchmark: str = "swe_bench",
    config: EvaluationConfig = None,
    seed: int = 42
) -> List[Trajectory]:
    """Generate realistic mock trajectories for evaluation."""
    if config is None:
        config = EvaluationConfig()
    
    random.seed(seed)
    np.random.seed(seed)
    
    trajectories = []
    
    # SWE-bench task patterns
    swe_task_templates = [
        {"query": "Fix the authentication bypass vulnerability in the login module",
         "skills": ["swe_debug", "swe_code_gen", "swe_verify"],
         "gt_model": "7B", "complexity": 0.75, "has_code": True, "tokens": 1500},
        {"query": "Add input validation for the API endpoint parameters",
         "skills": ["swe_code_gen", "swe_test"],
         "gt_model": "7B", "complexity": 0.60, "has_code": True, "tokens": 1200},
        {"query": "Search for all references to the deprecated function",
         "skills": ["swe_retrieve"],
         "gt_model": "4B", "complexity": 0.30, "has_code": False, "tokens": 500},
        {"query": "Refactor the database connection pooling architecture",
         "skills": ["swe_plan", "swe_code_gen", "swe_test", "swe_verify"],
         "gt_model": "14B", "complexity": 0.90, "has_code": True, "tokens": 2500},
        {"query": "Run the existing test suite and verify all pass",
         "skills": ["swe_test", "swe_verify"],
         "gt_model": "4B", "complexity": 0.25, "has_code": False, "tokens": 800},
        {"query": "Update the API documentation with new endpoint descriptions",
         "skills": ["swe_doc", "swe_retrieve"],
         "gt_model": "7B", "complexity": 0.45, "has_code": False, "tokens": 1000},
        {"query": "Debug the memory leak in the session management module",
         "skills": ["swe_debug", "swe_verify"],
         "gt_model": "7B", "complexity": 0.80, "has_code": True, "tokens": 1800},
        {"query": "Plan the migration from REST to GraphQL API",
         "skills": ["swe_plan", "swe_doc"],
         "gt_model": "14B", "complexity": 0.85, "has_code": False, "tokens": 2000},
    ]
    
    # WebArena task patterns
    web_task_templates = [
        {"query": "Navigate to the products page and find items under $50",
         "skills": ["web_navigate", "web_data_extract"],
         "gt_model": "4B", "complexity": 0.40, "has_code": False, "tokens": 600},
        {"query": "Fill out the registration form with test user data",
         "skills": ["web_navigate", "web_form_fill"],
         "gt_model": "4B", "complexity": 0.35, "has_code": False, "tokens": 500},
        {"query": "Extract all product names and prices from the catalog page",
         "skills": ["web_navigate", "web_data_extract"],
         "gt_model": "7B", "complexity": 0.55, "has_code": False, "tokens": 900},
        {"query": "Complete the checkout process with credit card payment",
         "skills": ["web_navigate", "web_form_fill", "web_data_extract"],
         "gt_model": "7B", "complexity": 0.70, "has_code": False, "tokens": 1500},
    ]
    
    templates = swe_task_templates if benchmark == "swe_bench" else web_task_templates
    
    for task_idx in range(n_tasks):
        template = templates[task_idx % len(templates)]
        
        # Generate 5-12 steps per task
        n_steps = random.randint(5, 12)
        task_id = f"{benchmark}_task_{task_idx:03d}"
        
        steps = []
        decisions = []
        results = []
        
        prev_failed = False
        budget = 1.0
        
        for step_idx in range(n_steps):
            # Create step context
            ctx = StepContext(
                step_id=step_idx,
                task_id=task_id,
                query=template["query"],
                gt_skills=template["skills"],
                gt_model=template["gt_model"],
                token_count=int(template["tokens"] * np.random.uniform(0.8, 1.2)),
                has_code=template["has_code"],
                programming_lang="python" if template["has_code"] else None,
                complexity_score=template["complexity"] * np.random.uniform(0.9, 1.1),
                budget_remaining=budget,
                previous_step_failed=prev_failed,
                benchmark=benchmark,
            )
            
            # Simulate routing decision (SDR-style)
            predicted_skills = template["skills"][:3]  # Top-3 predicted
            skill_scores = {s: np.random.uniform(0.6, 0.95) for s in predicted_skills}
            
            # Model selection (weighted by skill capabilities)
            model_probs = {}
            for model_id in config.models:
                cap = np.mean([
                    0.88 if model_id == "4B" else (0.87 if model_id == "7B" else 0.94)
                ])
                model_probs[model_id] = cap * np.random.uniform(0.8, 1.0)
            
            total_prob = sum(model_probs.values())
            model_dist = {m: p / total_prob for m, p in model_probs.items()}
            selected_model = max(model_dist, key=model_dist.get)
            
            # Add some randomness to avoid perfect routing
            if random.random() < 0.15:
                selected_model = random.choice(config.models)
            
            decision = RoutingDecision(
                step_id=step_idx,
                predicted_skills=predicted_skills,
                skill_scores=skill_scores,
                selected_model=selected_model,
                model_distribution=model_dist,
                uncertainty=np.random.uniform(0.1, 0.5),
                router_type="sdr",
                latency_ms=7.0 + np.random.uniform(-1, 2),
            )
            
            # Simulate execution result
            model_cost = {"4B": 800, "7B": 1500, "14B": 3000}
            model_latency = {"4B": 15, "7B": 25, "14B": 45}
            
            # Success probability based on model-skill match
            base_success = 0.85 if selected_model == template["gt_model"] else 0.60
            if prev_failed:
                base_success -= 0.15
            success = random.random() < base_success
            
            # Failure attribution
            failure_source = None
            failure_detail = ""
            if not success:
                fail_type = random.choice(list(FailureSource))
                failure_source = fail_type
                failure_detail = f"Step failed due to {fail_type.value}"
            
            # Dual feedback scores
            pre_score = np.random.uniform(0.5, 0.9)
            post_score = 0.8 if success else np.random.uniform(0.2, 0.5)
            
            result = StepResult(
                step_id=step_idx,
                success=success,
                quality_score=0.85 if success else 0.35,
                token_cost=int(model_cost.get(selected_model, 1000) * np.random.uniform(0.9, 1.1)),
                latency_ms=model_latency.get(selected_model, 30) * np.random.uniform(0.9, 1.1),
                model_used=selected_model,
                skills_involved=predicted_skills[:2],
                pre_execution_score=pre_score,
                post_execution_score=post_score,
                failure_source=failure_source,
                failure_detail=failure_detail,
            )
            
            steps.append(ctx)
            decisions.append(decision)
            results.append(result)
            
            # Update state
            prev_failed = not success
            budget -= result.token_cost / 50000.0
            budget = max(0.1, budget)
        
        # Task success if > 70% steps succeed
        task_success = np.mean([r.success for r in results]) > 0.7
        
        traj = Trajectory(
            task_id=task_id,
            benchmark=benchmark,
            steps=steps,
            decisions=decisions,
            results=results,
            task_success=task_success,
            total_tokens=sum(r.token_cost for r in results),
            total_latency_ms=sum(r.latency_ms for r in results),
        )
        trajectories.append(traj)
    
    return trajectories
