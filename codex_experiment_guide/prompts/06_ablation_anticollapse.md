# Codex Prompt 06: 反崩溃机制消融实验 (Exp 2)

## Context

验证 SDR 的反崩溃机制（cost-effectiveness scoring + entropy regularization）对路由稳定性的贡献。SkillOrchestra 发现 RL 路由器存在严重的"路由崩溃"问题（98% 时间调用同一模型），SDR 通过两种机制避免此问题。

**假设 H3**: 移除反崩溃机制后，SDR 的路由熵下降 > 50%

## Input Files

- `sdr_eval_pipeline/core/router.py` — SDRRouter._skill_conditioned_routing()
- `code/config.py` — 实验配置
- `output/exp1_baseline/results_raw.json` — 基线结果 (对照)

## Task

### Step 1: 创建 SDR 变体路由器

创建 `code/ablation_router_variants.py`：

```python
"""
SDR 反崩溃机制消融变体

4 个变体:
1. SDR-Full: 完整反崩溃 (0.6*cap + 0.4*cost_eff + softmax temp=0.15)
2. SDR-NoCostEff: 移除 cost-effectiveness 项 (纯 capability + softmax)
3. SDR-NoEntropyReg: 使用 greedy (argmax) 替代 softmax
4. SDR-RLPerStyle: 使用 RL-PER 的 cost penalty (success - 0.05*cost)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdr_eval_pipeline"))

from core.router import SDRRouter
from core.types import StepContext, RoutingDecision
from core.model_pool import ModelPool
from core.skill_registry import SkillRegistry


class SDRNoCostEff(SDRRouter):
    """变体2: 移除 cost-effectiveness 项"""
    
    def _skill_conditioned_routing(self, skills, conditions, ctx):
        model_scores = {m: 0.0 for m in self.config.models}
        
        for skill_name in skills:
            skill = self.skill_registry.skills.get(skill_name)
            if skill is None:
                continue
            for model_id in self.config.models:
                success_prob = skill.success_prob(model_id)
                # 仅使用 capability, 不加 cost-effectiveness
                ce = success_prob
                # 仍保留 rubric 条件
                if conditions.get("high_complexity"):
                    if model_id == "4B": ce -= 0.25
                    elif model_id == "14B": ce += 0.10
                else:
                    if model_id == "4B": ce += 0.15
                    elif model_id == "14B": ce -= 0.20
                model_scores[model_id] += ce
        
        # 仍使用 softmax (保留 entropy regularization)
        temp = 0.15
        max_score = max(model_scores.values()) if model_scores else 0
        exp_scores = {m: np.exp((s - max_score) / temp) for m, s in model_scores.items()}
        total = sum(exp_scores.values())
        return {m: v / total for m, v in exp_scores.items()}


class SDRNoEntropyReg(SDRRouter):
    """变体3: 使用 greedy (argmax) 替代 softmax"""
    
    def _skill_conditioned_routing(self, skills, conditions, ctx):
        model_scores = {m: 0.0 for m in self.config.models}
        
        for skill_name in skills:
            skill = self.skill_registry.skills.get(skill_name)
            if skill is None:
                continue
            for model_id in self.config.models:
                success_prob = skill.success_prob(model_id)
                cost_ratio = self.model_pool.get_model(model_id).cost_ratio
                cost_effectiveness = success_prob / cost_ratio
                ce = 0.6 * success_prob + 0.4 * cost_effectiveness
                if conditions.get("high_complexity"):
                    if model_id == "4B": ce -= 0.25
                    elif model_id == "14B": ce += 0.10
                else:
                    if model_id == "4B": ce += 0.15
                    elif model_id == "14B": ce -= 0.20
                model_scores[model_id] += ce
        
        # Greedy 选择 (无 softmax = 无 entropy regularization)
        best_model = max(model_scores, key=model_scores.get)
        return {m: (1.0 if m == best_model else 0.0) for m in self.config.models}


class SDRRLPerStyle(SDRRouter):
    """变体4: 使用 RL-PER 的 cost penalty 机制"""
    
    def _skill_conditioned_routing(self, skills, conditions, ctx):
        model_scores = {m: 0.0 for m in self.config.models}
        
        for skill_name in skills:
            skill = self.skill_registry.skills.get(skill_name)
            if skill is None:
                continue
            for model_id in self.config.models:
                success_prob = skill.success_prob(model_id)
                # RL-PER 风格: success - cost_penalty
                cost = self.model_pool.get_cost(model_id)
                cost_penalty = cost * 0.05
                ce = success_prob - cost_penalty
                if conditions.get("high_complexity") and model_id == "4B":
                    ce -= 0.1
                if conditions.get("low_budget") and model_id == "14B":
                    ce -= 0.15
                if conditions.get("prev_failed") and model_id == "4B":
                    ce -= 0.2
                model_scores[model_id] += ce
        
        # RL-PER 风格 softmax (高温度 = 更确定 = 更易崩溃)
        temp = 0.3
        exp_scores = {m: np.exp(s / temp) for m, s in model_scores.items()}
        total = sum(exp_scores.values())
        return {m: v / total for m, v in exp_scores.items()}


# 变体注册表
ABLATION_VARIANTS = {
    "full": SDRRouter,
    "no_cost_eff": SDRNoCostEff,
    "no_entropy_reg": SDRNoEntropyReg,
    "rl_per_style": SDRLPerStyle,
}
```

### Step 2: 运行消融实验

创建 `code/ablation_runner.py`：

```python
"""
消融实验统一运行器
"""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdr_eval_pipeline"))

from core.types import EvaluationConfig
from core.skill_registry import SkillRegistry
from core.model_pool import ModelPool
from data.mock_data import create_skill_registry, generate_mock_trajectories
from metrics.utilization import UtilizationMetrics

from ablation_router_variants import ABLATION_VARIANTS
from config import CONFIG


def run_exp2_anticollapse(verbose=True):
    """
    Exp 2: 反崩溃机制消融
    
    4 变体 × 2 benchmark × 5 seeds
    重点指标: C3 (Collapse Rate), C4 (Entropy), C5 (Pareto)
    """
    results = {}
    
    for variant_name, RouterClass in ABLATION_VARIANTS.items():
        variant_results = []
        
        for seed in CONFIG.seeds:
            np.random.seed(seed)
            
            for benchmark in CONFIG.benchmarks:
                n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
                config = EvaluationConfig(models=CONFIG.models, benchmark=benchmark, n_tasks=n_tasks)
                
                skill_registry = create_skill_registry(config)
                model_pool = ModelPool(config)
                trajectories = generate_mock_trajectories(config, n_tasks=n_tasks, benchmark=benchmark)
                
                router = RouterClass(config, model_pool, skill_registry)
                
                # 执行
                executed_trajs = []
                for traj in trajectories:
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                # 计算利用率指标
                um = UtilizationMetrics()
                util_metrics = um.compute(executed_trajs, config)
                
                # 计算成本
                total_tokens = sum(
                    getattr(r, 'tokens_used', 0)
                    for traj in executed_trajs
                    for r in traj.step_results
                )
                
                variant_results.append({
                    "seed": seed,
                    "benchmark": benchmark,
                    "routing_collapse_rate": util_metrics.get("routing_collapse_rate", 0),
                    "routing_entropy": util_metrics.get("routing_entropy", 0),
                    "utilization_balance": util_metrics.get("utilization_balance", 0),
                    "gini_coefficient": util_metrics.get("gini_coefficient", 0),
                    "total_tokens": total_tokens,
                    "model_distribution": util_metrics.get("model_distribution", {}),
                })
        
        # 聚合
        results[variant_name] = {
            "runs": variant_results,
            "mean_collapse": float(np.mean([r["routing_collapse_rate"] for r in variant_results])),
            "mean_entropy": float(np.mean([r["routing_entropy"] for r in variant_results])),
            "mean_tokens": float(np.mean([r["total_tokens"] for r in variant_results])),
            "mean_gini": float(np.mean([r["gini_coefficient"] for r in variant_results])),
        }
        
        if verbose:
            r = results[variant_name]
            print(f"\n  Variant: {variant_name}")
            print(f"    Collapse Rate: {r['mean_collapse']:.3f}")
            print(f"    Entropy:       {r['mean_entropy']:.3f} bits")
            print(f"    Gini:          {r['mean_gini']:.3f}")
            print(f"    Tokens:        {r['mean_tokens']:.0f}")
    
    # 计算消融增益
    full_entropy = results["full"]["mean_entropy"]
    for variant in results:
        results[variant]["entropy_drop_pct"] = (1 - results[variant]["mean_entropy"] / max(full_entropy, 1e-8)) * 100
    
    save_results(results, "exp2_anticollapse")
    return results
```

## Output

1. `output/exp2_anticollapse/results.json` — 4 变体消融结果
2. `code/ablation_router_variants.py` — 变体路由器实现
3. `code/ablation_runner.py` — 消融实验运行器
4. 控制台: 4 变体的 collapse/entropy 对比

## Verification

- [ ] `full` 变体的 Collapse Rate = 0.000
- [ ] `no_entropy_reg` (greedy) 的 Collapse Rate > 0.3
- [ ] `rl_per_style` 的 Collapse Rate > 0.5 (与 RL-PER 类似的崩溃行为)
- [ ] `full` 的 Entropy 最高 (~1.4 bits)
- [ ] `no_entropy_reg` 的 Entropy 最低 (~0 bits, 因为是 greedy)
- [ ] `rl_per_style` 的 Entropy < `full` 的 50%

## Expected Results

| 变体 | Collapse Rate | Entropy (bits) | Gini | Tokens |
|------|---------------|----------------|------|--------|
| **full** | **0.000** | **~1.40** | **~0.15** | **~131K** |
| no_cost_eff | ~0.10 | ~1.30 | ~0.20 | ~145K |
| no_entropy_reg | ~0.40 | ~0.00 | ~0.60 | ~120K |
| rl_per_style | ~0.60 | ~0.50 | ~0.70 | ~200K |

**关键发现**: 
- cost-effectiveness 主要影响模型选择多样性 (no_cost_eff 的 collapse 从 0% 升到 10%)
- entropy regularization 是防止崩溃的关键 (no_entropy_reg 的 collapse 从 0% 升到 40%)
- RL-PER 风格的 cost penalty 无法防止崩溃 (60% collapse，与原 RL-PER 一致)
