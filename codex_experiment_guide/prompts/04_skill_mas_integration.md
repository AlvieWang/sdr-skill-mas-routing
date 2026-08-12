# Codex Prompt 04: Skill-MAS 指标融合实验 (Exp 7)

## Context

将 Skill-MAS 的 7 类 22 个评测指标集成到 SDR pipeline 中，验证引入分布统计指标 (uncertainty, difficulty, priority) 后对 skill 演化效率的改善。

**假设 H8**: 引入 Skill-MAS 的 uncertainty 指标后，skill 演化收敛速度提升 > 30%

## Input Files

- `skill_mas_metrics/skill_mas_metrics.py` — Skill-MAS 指标模块
- `sdr_eval_pipeline/metrics/skill_evolution.py` — SDR skill 演化指标
- `output/exp1_sdr_extended/results.json` — 多轨迹采样结果
- `code/config.py` — 实验配置

## Task

### Step 1: 创建融合指标计算器

创建 `code/skill_mas_integration.py`：

```python
"""
Skill-MAS 指标融合模块

将 Skill-MAS 的 7 类指标集成到 SDR 评估流程中:
1. 分布统计 -> 增强 SDR 的 skill 演化决策
2. 选择性反思 -> 优化 SDR 的 skill refinement 预算
3. 消融指标 -> 度量选择性策略的增益
"""
import sys
import os
import json
import numpy as np
from typing import Dict, List, Tuple

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skill_mas_metrics"))


class SkillMASIntegration:
    """Skill-MAS 指标与 SDR pipeline 的融合层"""
    
    def __init__(self, config):
        self.config = config
        self.evolution_history = []  # 记录每轮演化结果
        
    def compute_integrated_metrics(
        self,
        rollout_results: Dict[str, List[float]],
        skill_registry,
        router,
        trajectories
    ) -> Dict:
        """
        计算融合后的指标集
        
        返回包含 SDR 原始指标 + Skill-MAS 补充指标的统一字典
        """
        results = {}
        
        # === Skill-MAS 指标 ===
        
        # 1. 分布统计
        task_stats = self._compute_distributional(rollout_results)
        results["distributional"] = task_stats
        
        # 2. 选择性反思
        selected, elbow = self._compute_selective_reflection(task_stats)
        results["selective_reflection"] = {
            "elbow_index": elbow,
            "selected_count": len(selected),
            "total_count": len(task_stats),
            "selection_ratio": len(selected) / max(len(task_stats), 1)
        }
        
        # 3. 迁移性 (如果有跨域数据)
        results["transferability"] = {
            "cross_task_delta": None,  # 由 Exp 5 填充
            "cross_model_delta": None,
        }
        
        # 4. 成本
        results["cost"] = self._compute_cost(router, trajectories)
        
        # 5. 演化追踪
        results["evolution_tracking"] = self._compute_evolution(skill_registry)
        
        # === SDR 原始指标 (对照) ===
        results["sdr_original"] = {
            "skill_hit_at_1": self._get_metric(router, "skill_hit_at_1"),
            "routing_entropy": self._get_metric(router, "routing_entropy"),
            "routing_collapse": self._get_metric(router, "routing_collapse"),
        }
        
        return results
    
    def _compute_distributional(self, rollout_results):
        """Skill-MAS 分布统计"""
        task_stats = {}
        all_u, all_d = [], []
        
        for task_id, scores in rollout_results.items():
            mean_s = float(np.mean(scores))
            std_s = float(np.std(scores))
            task_stats[task_id] = {
                "mean": mean_s,
                "uncertainty": std_s,
                "difficulty": -mean_s,
            }
            all_u.append(std_s)
            all_d.append(-mean_s)
        
        # 归一化
        u_min, u_max = min(all_u), max(all_u)
        d_min, d_max = min(all_d), max(all_d)
        
        for tid in task_stats:
            task_stats[tid]["norm_u"] = (task_stats[tid]["uncertainty"] - u_min) / max(u_max - u_min, 1e-8)
            task_stats[tid]["norm_d"] = (task_stats[tid]["difficulty"] - d_min) / max(d_max - d_min, 1e-8)
            task_stats[tid]["priority"] = 0.5 * (task_stats[tid]["norm_u"] + task_stats[tid]["norm_d"])
        
        # 汇总统计
        return {
            "per_task": task_stats,
            "summary": {
                "mean_uncertainty": float(np.mean(all_u)),
                "mean_difficulty": float(np.mean(all_d)),
                "max_uncertainty": float(max(all_u)),
                "high_uncertainty_ratio": float(np.mean([u > np.mean(all_u) for u in all_u])),
            }
        }
    
    def _compute_selective_reflection(self, task_stats):
        """Skill-MAS 选择性反思: 肘部检测"""
        sorted_tasks = sorted(
            task_stats["per_task"].items(),
            key=lambda x: x[1]["priority"],
            reverse=True
        )
        priorities = [s["priority"] for _, s in sorted_tasks]
        n = len(priorities)
        
        if n <= 2:
            return [t for t, _ in sorted_tasks], n
        
        first_diffs = [priorities[j] - priorities[j+1] for j in range(n-1)]
        second_diffs = [abs(first_diffs[j] - first_diffs[j+1]) for j in range(n-2)]
        elbow = int(np.argmax(second_diffs)) + 1
        
        return [t for t, _ in sorted_tasks[:elbow]], elbow
    
    def _compute_cost(self, router, trajectories):
        """成本统计"""
        total_tokens = 0
        total_success = 0
        
        for traj in trajectories:
            for result in getattr(traj, 'step_results', []):
                total_tokens += getattr(result, 'tokens_used', 0)
                if getattr(result, 'success', False):
                    total_success += 1
        
        n_steps = sum(len(getattr(t, 'steps', [])) for t in trajectories)
        
        return {
            "total_tokens": total_tokens,
            "inference_cost": total_tokens * 0.000002,  # 假设 $2/M tokens
            "success_rate": total_success / max(n_steps, 1),
        }
    
    def _compute_evolution(self, skill_registry):
        """演化追踪"""
        skills = getattr(skill_registry, 'skills', {})
        total_alpha = sum(s.alpha for s in skills.values())
        total_beta = sum(s.beta for s in skills.values())
        
        return {
            "skill_count": len(skills),
            "total_evidence": total_alpha + total_beta,
            "avg_success_prob": total_alpha / max(total_alpha + total_beta, 1),
            "evolution_rounds": len(self.evolution_history),
        }
    
    def _get_metric(self, router, metric_name):
        """从路由器获取指标值"""
        return getattr(router, metric_name, None)
    
    def run_evolution_with_skillmas(
        self,
        router,
        trajectories,
        skill_registry,
        config,
        n_rounds: int = 10,
        strategy: str = "priority"
    ):
        """
        使用 Skill-MAS 策略驱动 skill 演化
        
        strategy:
            "priority" - Skill-MAS 优先级排序 + 肘部截断
            "full" - 全量反思 (所有 task)
            "random" - 随机 50% task
            "none" - 不演化
        """
        evolution_log = []
        
        for round_idx in range(n_rounds):
            # 多轨迹采样
            rollout_results = {}
            for traj in trajectories:
                scores = []
                for k in range(5):
                    np.random.seed(round_idx * 1000 + k)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    n_success = sum(1 for r in executed.step_results if r.success)
                    scores.append(n_success / max(len(executed.step_results), 1))
                rollout_results[traj.task_id] = scores
            
            # 分布统计
            task_stats = self._compute_distributional(rollout_results)
            
            # 选择 task 进行 skill 演化
            if strategy == "priority":
                selected, elbow = self._compute_selective_reflection(task_stats)
            elif strategy == "full":
                selected = list(rollout_results.keys())
                elbow = len(selected)
            elif strategy == "random":
                all_tasks = list(rollout_results.keys())
                selected = np.random.choice(all_tasks, size=len(all_tasks)//2, replace=False).tolist()
                elbow = len(selected)
            else:  # none
                selected = []
                elbow = 0
            
            # 对选中的 task 更新 skill 后验
            n_updates = 0
            for task_id in selected:
                scores = rollout_results[task_id]
                success = np.mean(scores) > 0.5
                # 更新 skill registry
                for skill_name in trajectories[0].steps[0].gt_skills if trajectories else []:
                    if skill_name in skill_registry.skills:
                        skill = skill_registry.skills[skill_name]
                        if success:
                            skill.alpha += 1
                        else:
                            skill.beta += 1
                        n_updates += 1
            
            # 记录演化日志
            round_log = {
                "round": round_idx,
                "elbow_index": elbow,
                "selected_count": len(selected),
                "n_skill_updates": n_updates,
                "mean_score": float(np.mean([np.mean(s) for s in rollout_results.values()])),
                "mean_uncertainty": task_stats["summary"]["mean_uncertainty"],
                "strategy": strategy,
            }
            evolution_log.append(round_log)
            
            # 检查收敛
            if round_idx > 2:
                recent_scores = [l["mean_score"] for l in evolution_log[-3:]]
                if np.std(recent_scores) < 0.01:
                    round_log["converged"] = True
                    break
        
        return {
            "strategy": strategy,
            "evolution_log": evolution_log,
            "convergence_round": next((l["round"] for l in evolution_log if l.get("converged")), len(evolution_log)),
            "final_score": evolution_log[-1]["mean_score"] if evolution_log else 0,
            "total_updates": sum(l["n_skill_updates"] for l in evolution_log),
        }
```

### Step 2: 运行融合实验

```python
def run_exp7_skillmas(verbose=True):
    """
    Exp 7: Skill-MAS 指标融合实验
    
    4 个变体:
    1. base: SDR 原始指标集 (无 Skill-MAS)
    2. +uncertainty: 加入 step-level uncertainty
    3. +priority: 加入 priority score 排序
    4. +all: 全部 Skill-MAS 分布统计
    """
    import numpy as np
    from core.types import EvaluationConfig
    from core.skill_registry import SkillRegistry
    from core.model_pool import ModelPool
    from core.router import SDRRouter
    from data.mock_data import create_skill_registry, generate_mock_trajectories
    
    integration = SkillMASIntegration(CONFIG)
    
    strategies = ["none", "random", "full", "priority"]
    results = {}
    
    for strategy in strategies:
        strategy_results = []
        
        for seed in CONFIG.seeds:
            np.random.seed(seed)
            
            config = EvaluationConfig(models=CONFIG.models, benchmark="swe_bench", n_tasks=CONFIG.n_tasks_swe)
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(config, n_tasks=CONFIG.n_tasks_swe, benchmark="swe_bench")
            
            router = SDRRouter(config, model_pool, skill_registry)
            
            # 运行演化
            evolution_result = integration.run_evolution_with_skillmas(
                router, trajectories, skill_registry, config,
                n_rounds=CONFIG.evolution_rounds,
                strategy=strategy
            )
            
            strategy_results.append(evolution_result)
        
        results[strategy] = {
            "runs": strategy_results,
            "mean_convergence": float(np.mean([r["convergence_round"] for r in strategy_results])),
            "mean_final_score": float(np.mean([r["final_score"] for r in strategy_results])),
            "mean_total_updates": float(np.mean([r["total_updates"] for r in strategy_results])),
        }
        
        if verbose:
            print(f"\n  Strategy: {strategy}")
            print(f"    Convergence: {results[strategy]['mean_convergence']:.1f} rounds")
            print(f"    Final score: {results[strategy]['mean_final_score']:.4f}")
            print(f"    Total updates: {results[strategy]['mean_total_updates']:.0f}")
    
    # 计算增益
    base_conv = results["none"]["mean_convergence"]
    for strategy in strategies:
        results[strategy]["convergence_speedup"] = base_conv / max(results[strategy]["mean_convergence"], 1)
    
    save_results(results, "exp7_skillmas")
    return results
```

## Output

1. `output/exp7_skillmas/results.json` — 4 策略 × 5 seeds 的演化结果
2. `code/skill_mas_integration.py` — 融合模块代码
3. 控制台: 收敛速度对比

## Verification

- [ ] 4 个策略全部运行完成
- [ ] `priority` 策略的收敛轮次 < `full` 策略
- [ ] `priority` 策略的 final_score ≥ `full` 策略的 final_score
- [ ] `priority` 策略的 total_updates < `full` 策略 (更高效)
- [ ] `none` 策略的 final_score 最低

## Expected Results

| 策略 | 收敛轮次 | Final Score | Total Updates |
|------|---------|-------------|---------------|
| none | 10 (未收敛) | ~0.65 | 0 |
| random | ~8 | ~0.72 | ~250 |
| full | ~6 | ~0.78 | ~500 |
| **priority** | **~4** | **~0.80** | **~150** |

关键发现：`priority` 策略用最少的更新次数 (~150 vs full 的 ~500) 达到了最优或持平的性能，收敛速度提升 ~33%。
