#!/usr/bin/env python3
"""
SDR Evaluation Pipeline - Main Runner
=====================================

Skill-Driven Dynamic Routing (SDR) Evaluation Pipeline
AAAI Paper - Rubar / RL-PER / SDR Comparison

This pipeline evaluates three routing strategies:
  1. Rubar   - Deterministic rubric-based routing (7-dim condition matrix)
  2. RL-PER  - RL-pretrained external router (4B Qwen3 router)
  3. SDR     - Skill-Driven Dynamic Router (proposed framework)

Across 6 metric categories (A-F):
  A: Skill-Level Routing Accuracy      (SkillRouter-inspired)
  B: Skill Transfer & Adaptation       (SkillOpt + LaMer-inspired)
  C: Utilization & Stability           (SkillOrchestra-inspired)
  D: Skill Evolution & Quality         (SkillOpt + SkillOrchestra-inspired)
  E: Dual Feedback                     (ToolTree-inspired)
  F: Failure Attribution               (PawBench-inspired)

Usage:
  cd sdr_eval_pipeline
  python run_pipeline.py                    # Run all routers, all metrics
  python run_pipeline.py --router sdr       # Run only SDR router
  python run_pipeline.py --benchmark both   # Run SWE-bench + WebArena
  python run_pipeline.py --output results  # Specify output directory
  python run_pipeline.py --verbose           # Detailed per-metric output

Requirements:
  pip install numpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.types import (
    EvaluationConfig,
    MetricResult,
    Trajectory,
    StepContext,
    RoutingDecision,
    StepResult,
    FailureSource,
)
from core.skill_registry import SkillRegistry
from core.model_pool import ModelPool
from core.router import BaseRouter, RubarRouter, RLPerRouter, SDRRouter

from data.mock_data import create_skill_registry, generate_mock_trajectories

from metrics.routing_accuracy import RoutingAccuracyMetrics
from metrics.transfer import TransferMetrics
from metrics.utilization import UtilizationMetrics
from metrics.skill_evolution import SkillEvolutionMetrics
from metrics.dual_feedback import DualFeedbackMetrics
from metrics.failure_attribution import FailureAttributionMetrics


# ============================================================
# Pipeline Configuration
# ============================================================

ROUTER_NAMES = {
    "rubar": "Rubar (Rubric-Based)",
    "rl_per": "RL-PER (RL Router)",
    "sdr": "SDR (Skill-Driven)",
}

METRIC_CATEGORIES = {
    "A": "Routing Accuracy",
    "B": "Transfer & Adaptation",
    "C": "Utilization & Stability",
    "D": "Skill Evolution",
    "E": "Dual Feedback",
    "F": "Failure Attribution",
}


# ============================================================
# Router Factory
# ============================================================

def create_router(
    router_type: str,
    config: EvaluationConfig,
    model_pool: ModelPool,
    skill_registry: Optional[SkillRegistry] = None,
) -> BaseRouter:
    """Create a router instance by type."""
    if router_type == "rubar":
        return RubarRouter(config, model_pool)
    elif router_type == "rl_per":
        return RLPerRouter(config, model_pool)
    elif router_type == "sdr":
        if skill_registry is None:
            raise ValueError("SDR router requires a skill_registry")
        return SDRRouter(config, model_pool, skill_registry)
    else:
        raise ValueError(f"Unknown router type: {router_type}")


# ============================================================
# Execution Engine
# ============================================================

def execute_trajectory(
    router: BaseRouter,
    trajectory: Trajectory,
    config: EvaluationConfig,
    skill_registry: Optional[SkillRegistry] = None,
) -> Trajectory:
    """
    Execute a trajectory using the given router.
    
    For each step:
      1. Router makes a routing decision (skill + model)
      2. Simulate execution (in production: call actual model)
      3. Record step result
      4. Update router/registry if applicable
    """
    new_steps = []
    new_decisions = []
    new_results = []
    
    prev_failed = False
    budget = 1.0
    
    for i, ctx in enumerate(trajectory.steps):
        # Update context state
        ctx.previous_step_failed = prev_failed
        ctx.budget_remaining = budget
        
        # Router decision
        decision = router.route(ctx)
        
        # Simulate execution
        model_cost = {"4B": 800, "7B": 1500, "14B": 3000}
        model_latency = {"4B": 15.0, "7B": 25.0, "14B": 45.0}
        
        # Success probability based on model-skill match
        base_success = 0.85
        if decision.selected_model == ctx.gt_model:
            base_success = 0.90
        else:
            # Check capability mismatch
            if ctx.complexity_score > 0.7 and decision.selected_model == "4B":
                base_success = 0.45
            elif ctx.complexity_score < 0.4 and decision.selected_model == "14B":
                base_success = 0.92  # Overkill but still works
        
        if prev_failed:
            base_success -= 0.15
        
        success = np.random.random() < base_success
        
        # Failure attribution
        failure_source = None
        failure_detail = ""
        if not success:
            fail_roll = np.random.random()
            if fail_roll < 0.25:
                failure_source = FailureSource.MODEL_REASONING
                failure_detail = "Model reasoning insufficient for task complexity"
            elif fail_roll < 0.45:
                failure_source = FailureSource.SKILL_DISCOVERY_WEAK
                failure_detail = "Required skill not identified by router"
            elif fail_roll < 0.60:
                failure_source = FailureSource.TOOL_MISSING
                failure_detail = "Required tool not available"
            elif fail_roll < 0.75:
                failure_source = FailureSource.WORKSPACE_PERCEPTION
                failure_detail = "Workspace context not properly understood"
            elif fail_roll < 0.88:
                failure_source = FailureSource.NETWORK_FRAGILE
                failure_detail = "Network operation failed"
            else:
                failure_source = FailureSource.COMPLETION_CHECK_LOOSE
                failure_detail = "Completion check too permissive"
        
        # Dual feedback scores
        pre_score = decision.model_distribution.get(decision.selected_model, 0.5)
        post_score = 0.85 if success else np.random.uniform(0.15, 0.45)
        
        # Skills involved (from decision or ground truth)
        skills_involved = decision.predicted_skills[:2] if decision.predicted_skills else ctx.gt_skills[:2]
        
        result = StepResult(
            step_id=ctx.step_id,
            success=success,
            quality_score=0.85 if success else 0.35,
            token_cost=int(model_cost.get(decision.selected_model, 1000) * np.random.uniform(0.9, 1.1)),
            latency_ms=model_latency.get(decision.selected_model, 30) * np.random.uniform(0.9, 1.1),
            model_used=decision.selected_model,
            skills_involved=skills_involved,
            pre_execution_score=pre_score,
            post_execution_score=post_score,
            failure_source=failure_source,
            failure_detail=failure_detail,
        )
        
        # Update router (RL-PER and SDR)
        reward = 1.0 if success else 0.0
        if isinstance(router, RLPerRouter):
            router.update(ctx, decision, reward)
        elif isinstance(router, SDRRouter) and skill_registry:
            skill_success = {s: success for s in skills_involved}
            router.update(ctx, decision, reward, skill_success)
            
            # Check skill evolution
            for skill_name in skills_involved:
                skill_registry.update_skill_posterior(
                    skill_name, decision.selected_model, success
                )
        
        new_steps.append(ctx)
        new_decisions.append(decision)
        new_results.append(result)
        
        prev_failed = not success
        budget -= result.token_cost / 50000.0
        budget = max(0.1, budget)
    
    # Task success if > 70% steps succeed
    task_success = np.mean([r.success for r in new_results]) > 0.7
    
    executed = Trajectory(
        task_id=trajectory.task_id,
        benchmark=trajectory.benchmark,
        steps=new_steps,
        decisions=new_decisions,
        results=new_results,
        task_success=task_success,
        total_tokens=sum(r.token_cost for r in new_results),
        total_latency_ms=sum(r.latency_ms for r in new_results),
    )
    
    return executed


# ============================================================
# Metric Evaluation
# ============================================================

def run_all_metrics(
    trajectories: list[Trajectory],
    config: EvaluationConfig,
    skill_registry: Optional[SkillRegistry] = None,
) -> dict[str, list[MetricResult]]:
    """Run all 6 metric categories on the given trajectories."""
    all_results = {}
    
    # A: Routing Accuracy
    metric_a = RoutingAccuracyMetrics(config)
    all_results["A"] = metric_a.evaluate(trajectories)
    
    # B: Transfer & Adaptation
    metric_b = TransferMetrics(config)
    all_results["B"] = metric_b.evaluate(trajectories)
    
    # C: Utilization & Stability
    metric_c = UtilizationMetrics(config)
    all_results["C"] = metric_c.evaluate(trajectories)
    
    # D: Skill Evolution (requires skill_registry)
    if skill_registry:
        metric_d = SkillEvolutionMetrics(config, skill_registry)
        all_results["D"] = metric_d.evaluate(trajectories)
    else:
        all_results["D"] = []
    
    # E: Dual Feedback
    metric_e = DualFeedbackMetrics(config)
    all_results["E"] = metric_e.evaluate(trajectories)
    
    # F: Failure Attribution
    metric_f = FailureAttributionMetrics(config)
    all_results["F"] = metric_f.evaluate(trajectories)
    
    return all_results


# ============================================================
# Output Formatting
# ============================================================

def format_results_table(
    router_name: str,
    all_metrics: dict[str, list[MetricResult]],
    verbose: bool = False,
) -> str:
    """Format metric results as a readable table."""
    lines = []
    lines.append("=" * 90)
    lines.append(f"  Router: {router_name}")
    lines.append("=" * 90)
    
    for category, name in METRIC_CATEGORIES.items():
        results = all_metrics.get(category, [])
        if not results:
            continue
        
        lines.append(f"\n  Category {category}: {name}")
        lines.append("  " + "-" * 86)
        lines.append(f"  {'Metric':<45} {'Value':>10} {'Baseline':>10} {'Source':<20}")
        lines.append("  " + "-" * 86)
        
        for r in results:
            val_str = f"{r.value:.3f}" if r.value is not None else "N/A"
            base_str = f"{r.baseline_value:.3f}" if r.baseline_value is not None else "---"
            source = r.detail.get("source", "") if r.detail else ""
            lines.append(f"  {r.name:<45} {val_str:>10} {base_str:>10} {source:<20}")
            
            if verbose and r.detail:
                for k, v in r.detail.items():
                    if k != "source":
                        lines.append(f"    {'':>43} └─ {k}: {v}")
        
        lines.append("  " + "-" * 86)
    
    lines.append("")
    return "\n".join(lines)


def format_comparison_table(
    router_results: dict[str, dict[str, list[MetricResult]]],
) -> str:
    """Format a comparison table across routers."""
    lines = []
    lines.append("=" * 110)
    lines.append("  COMPARISON: Rubar vs RL-PER vs SDR")
    lines.append("=" * 110)
    
    for category, name in METRIC_CATEGORIES.items():
        lines.append(f"\n  Category {category}: {name}")
        lines.append("  " + "-" * 106)
        
        # Collect all metric names
        all_metric_names = []
        for router_name in router_results:
            for r in router_results[router_name].get(category, []):
                if r.name not in all_metric_names:
                    all_metric_names.append(r.name)
        
        # Header
        header = f"  {'Metric':<45}"
        for router_name in router_results:
            short_name = router_name.split("(")[0].strip()
            header += f" {short_name:>15}"
        header += f" {'Baseline':>15}"
        lines.append(header)
        lines.append("  " + "-" * 106)
        
        # Rows
        for metric_name in all_metric_names:
            row = f"  {metric_name:<45}"
            baseline = "---"
            for router_name in router_results:
                found = False
                for r in router_results[router_name].get(category, []):
                    if r.name == metric_name:
                        val_str = f"{r.value:.3f}"
                        if r.baseline_value is not None:
                            baseline = f"{r.baseline_value:.3f}"
                        row += f" {val_str:>15}"
                        found = True
                        break
                if not found:
                    row += f" {'N/A':>15}"
            row += f" {baseline:>15}"
            lines.append(row)
        
        lines.append("  " + "-" * 106)
    
    lines.append("")
    return "\n".join(lines)


def save_results_json(
    output_dir: str,
    router_results: dict[str, dict[str, list[MetricResult]]],
) -> str:
    """Save all results as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    
    output = {}
    for router_name, metrics_by_cat in router_results.items():
        output[router_name] = {}
        for cat, results in metrics_by_cat.items():
            output[router_name][cat] = [
                {
                    "name": r.name,
                    "value": r.value,
                    "category": r.category,
                    "description": r.description,
                    "baseline_value": r.baseline_value,
                    "detail": r.detail,
                }
                for r in results
            ]
    
    filepath = os.path.join(output_dir, "results.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    return filepath


# ============================================================
# Main Pipeline
# ============================================================

def run_pipeline(
    routers: list[str],
    benchmark: str = "swe_bench",
    n_tasks: int = 20,
    output_dir: str = "output",
    verbose: bool = False,
    seed: int = 42,
):
    """Run the full evaluation pipeline."""
    print("\n" + "=" * 90)
    print("  SDR Evaluation Pipeline - Skill-Driven Dynamic Routing")
    print("  AAAI Paper | Rubar vs RL-PER vs SDR")
    print("=" * 90)
    
    config = EvaluationConfig(output_dir=output_dir, verbose=verbose)
    
    # Determine benchmarks
    benchmarks = []
    if benchmark == "swe_bench":
        benchmarks = [("swe_bench", n_tasks)]
    elif benchmark == "webarena":
        benchmarks = [("webarena", n_tasks)]
    elif benchmark == "both":
        benchmarks = [("swe_bench", n_tasks), ("webarena", n_tasks)]
    else:
        benchmarks = [(benchmark, n_tasks)]
    
    # Create shared model pool
    model_pool = ModelPool(config)
    print(f"\n  Model Pool: {config.models}")
    for model_id in config.models:
        profile = model_pool.get_model(model_id)
        if profile:
            print(f"    {model_id}: Pass@1={profile.pass_at_1:.3f}, Cost={profile.cost_ratio}x, Latency={profile.base_latency_ms}ms")
    
    # Generate trajectories for all benchmarks
    all_trajectories = {}
    for bench, n in benchmarks:
        print(f"\n  Generating {n} mock trajectories for {bench}...")
        trajs = generate_mock_trajectories(n_tasks=n, benchmark=bench, config=config, seed=seed)
        all_trajectories[bench] = trajs
        success_rate = np.mean([1 if t.task_success else 0 for t in trajs])
        print(f"    Task success rate (mock): {success_rate:.1%}")
        print(f"    Total steps: {sum(len(t.steps) for t in trajs)}")
    
    # Run each router
    router_results = {}
    router_executed = {}
    
    for router_type in routers:
        router_name = ROUTER_NAMES.get(router_type, router_type)
        print(f"\n{'─' * 90}")
        print(f"  Running Router: {router_name}")
        print(f"{'─' * 90}")
        
        start_time = time.time()
        
        # Create skill registry (fresh for each router, for SDR)
        skill_registry = None
        if router_type == "sdr":
            skill_registry = create_skill_registry(config)
            print(f"  Skill Registry: {len(skill_registry.skills)} skills registered")
            print(f"  Skill Coverage: {skill_registry.get_skill_coverage():.1%}")
        
        # Create router
        router = create_router(router_type, config, model_pool, skill_registry)
        
        # Execute trajectories
        all_executed = []
        for bench, trajs in all_trajectories.items():
            for traj in trajs:
                # Reset random seed per trajectory for reproducibility
                np.random.seed(seed + hash(traj.task_id) % 10000)
                executed = execute_trajectory(
                    router, traj, config, skill_registry
                )
                all_executed.append(executed)
        
        elapsed = time.time() - start_time
        
        # Compute metrics
        metrics = run_all_metrics(all_executed, config, skill_registry)
        router_results[router_name] = metrics
        router_executed[router_name] = all_executed
        
        # Print results
        print(format_results_table(router_name, metrics, verbose=verbose))
        
        # Summary stats
        task_sr = np.mean([1 if t.task_success else 0 for t in all_executed])
        step_sr = np.mean([r.success for t in all_executed for r in t.results])
        total_tokens = sum(t.total_tokens for t in all_executed)
        total_latency = sum(t.total_latency_ms for t in all_executed)
        
        print(f"  Summary:")
        print(f"    Task Success Rate:    {task_sr:.1%}")
        print(f"    Step Success Rate:    {step_sr:.1%}")
        print(f"    Total Tokens:         {total_tokens:,}")
        print(f"    Total Latency:        {total_latency:.0f}ms")
        print(f"    Execution Time:       {elapsed:.2f}s")
        
        if skill_registry:
            evo = skill_registry.get_evolution_summary()
            print(f"    Skill Evolution:      {evo['splits']} splits, {evo['merges']} merges")
    
    # Print comparison table if multiple routers
    if len(routers) > 1:
        print("\n" + format_comparison_table(router_results))
    
    # Save JSON results
    json_path = save_results_json(output_dir, router_results)
    print(f"\n  Results saved to: {json_path}")
    
    print("\n" + "=" * 90)
    print("  Pipeline Complete")
    print("=" * 90 + "\n")
    
    return router_results


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SDR Evaluation Pipeline - Skill-Driven Dynamic Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                         # All routers, SWE-bench, 20 tasks
  python run_pipeline.py --router sdr             # Only SDR router
  python run_pipeline.py --benchmark both         # SWE-bench + WebArena
  python run_pipeline.py --n-tasks 50             # 50 tasks per benchmark
  python run_pipeline.py --verbose                # Detailed per-metric output
  python run_pipeline.py --output results/run1    # Custom output directory
        """
    )
    parser.add_argument(
        "--router",
        choices=["rubar", "rl_per", "sdr", "all"],
        default="all",
        help="Which router(s) to evaluate (default: all)"
    )
    parser.add_argument(
        "--benchmark",
        choices=["swe_bench", "webarena", "both"],
        default="swe_bench",
        help="Which benchmark to use (default: swe_bench)"
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=20,
        help="Number of tasks per benchmark (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-metric output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Determine routers
    if args.router == "all":
        routers = ["rubar", "rl_per", "sdr"]
    else:
        routers = [args.router]
    
    run_pipeline(
        routers=routers,
        benchmark=args.benchmark,
        n_tasks=args.n_tasks,
        output_dir=args.output,
        verbose=args.verbose,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
