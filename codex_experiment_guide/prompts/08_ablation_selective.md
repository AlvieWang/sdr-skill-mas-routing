# Codex Prompt 08: 选择性反思消融实验 (Exp 4)

## Context

验证 Skill-MAS 的选择性反思机制（优先级排序 + 肘部截断）是否优于全量/随机反思。这是 Skill-MAS 的核心创新之一，我们将其引入 SDR 的 skill 演化流程。

**假设 H5**: 优先级驱动的 skill 演化比全量反思节省 > 50% 计算成本，且性能不降

## Input Files

- `code/skill_mas_integration.py` — SkillMASIntegration (含演化逻辑)
- `code/config.py` — 实验配置
- `output/exp1_sdr_extended/results.json` — 多轨迹采样结果 (提供 priority 数据)

## Task

### Step 1: 实现选择性反思消融

在 `code/ablation_runner.py` 中添加：

```python
def run_exp4_selective(verbose=True):
    """
    Exp 4: 选择性反思消融实验
    
    4 种演化策略:
    1. priority: Skill-MAS 优先级排序 + 肘部截断
    2. full: 全量反思 (所有 task)
    3. random: 随机 50% task
    4. none: 不演化
    
    对比指标:
    - D1: Skill Refinement Rate
    - D3: Convergence rounds
    - 成本: Evolution Cost (tokens)
    - 最终性能: Avg.Perf
    """
    from skill_mas_integration import SkillMASIntegration
    
    strategies = ["priority", "full", "random", "none"]
    results = {}
    
    for strategy in strategies:
        strategy_results = []
        
        for seed in CONFIG.seeds:
            np.random.seed(seed)
            
            config = EvaluationConfig(
                models=CONFIG.models, 
                benchmark="swe_bench", 
                n_tasks=CONFIG.n_tasks_swe
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(
                config, n_tasks=CONFIG.n_tasks_swe, benchmark="swe_bench"
            )
            
            router = SDRRouter(config, model_pool, skill_registry)
            integration = SkillMASIntegration(CONFIG)
            
            # 运行演化
            evolution_result = integration.run_evolution_with_skillmas(
                router, trajectories, skill_registry, config,
                n_rounds=CONFIG.evolution_rounds,
                strategy=strategy
            )
            
            # 计算最终性能 (在测试集上)
            test_trajectories = generate_mock_trajectories(
                config, n_tasks=20, benchmark="swe_bench"
            )
            
            total_success = 0
            total_steps = 0
            total_tokens = 0
            
            for traj in test_trajectories:
                executed = execute_trajectory(router, traj, config, skill_registry)
                for result in executed.step_results:
                    total_steps += 1
                    total_tokens += getattr(result, 'tokens_used', 0)
                    if result.success:
                        total_success += 1
            
            # 计算演化成本
            evolution_cost_tokens = sum(
                len(traj.steps) * CONFIG.rollout_per_task * 1500  # 平均 token/step
                for traj in trajectories
            ) * evolution_result["convergence_round"]
            
            # 如果是 priority/random, 只计算选中 task 的成本
            if strategy == "priority":
                avg_selection_ratio = np.mean([
                    l["selected_count"] / max(l.get("total_count", 50), 1)
                    for l in evolution_result["evolution_log"]
                ])
                evolution_cost_tokens = int(evolution_cost_tokens * avg_selection_ratio)
            elif strategy == "random":
                evolution_cost_tokens = int(evolution_cost_tokens * 0.5)
            elif strategy == "none":
                evolution_cost_tokens = 0
            
            strategy_results.append({
                "seed": seed,
                "convergence_round": evolution_result["convergence_round"],
                "final_score": evolution_result["final_score"],
                "total_updates": evolution_result["total_updates"],
                "test_success_rate": total_success / max(total_steps, 1),
                "test_tokens": total_tokens,
                "evolution_cost_tokens": evolution_cost_tokens,
                "evolution_cost_usd": evolution_cost_tokens * 0.000002,
            })
        
        results[strategy] = {
            "runs": strategy_results,
            "mean_convergence": float(np.mean([r["convergence_round"] for r in strategy_results])),
            "mean_final_score": float(np.mean([r["final_score"] for r in strategy_results])),
            "mean_test_success": float(np.mean([r["test_success_rate"] for r in strategy_results])),
            "mean_evolution_cost": float(np.mean([r["evolution_cost_tokens"] for r in strategy_results])),
            "mean_evolution_cost_usd": float(np.mean([r["evolution_cost_usd"] for r in strategy_results])),
        }
        
        if verbose:
            r = results[strategy]
            print(f"\n  Strategy: {strategy}")
            print(f"    Convergence:     {r['mean_convergence']:.1f} rounds")
            print(f"    Final score:     {r['mean_final_score']:.4f}")
            print(f"    Test success:    {r['mean_test_success']:.4f}")
            print(f"    Evolution cost:  {r['mean_evolution_cost']:.0f} tokens (${r['mean_evolution_cost_usd']:.2f})")
    
    # 计算消融增益
    full_cost = results["full"]["mean_evolution_cost"]
    for strategy in results:
        results[strategy]["cost_saving_pct"] = (1 - results[strategy]["mean_evolution_cost"] / max(full_cost, 1)) * 100
        results[strategy]["perf_vs_full"] = results[strategy]["mean_test_success"] - results["full"]["mean_test_success"]
    
    save_results(results, "exp4_selective")
    return results
```

### Step 2: 生成消融对比表

```python
def generate_ablation_table_exp4(results: dict) -> str:
    """生成 Exp 4 的 LaTeX 对比表"""
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Selective Reflection Ablation}")
    lines.append("\\label{tab:ablation_selective}")
    lines.append("\\begin{tabular}{l|cccc}")
    lines.append("\\toprule")
    lines.append("Strategy & Convergence & Test Success & Evolution Cost & Cost Saving \\\\")
    lines.append(" & (rounds) & (\\%) & (tokens) & (\\%) \\\\")
    lines.append("\\midrule")
    
    for strategy in ["priority", "full", "random", "none"]:
        r = results[strategy]
        conv = f"{r['mean_convergence']:.1f}"
        success = f"{r['mean_test_success']*100:.1f}"
        cost = f"{r['mean_evolution_cost']:.0f}"
        saving = f"{r['cost_saving_pct']:.1f}"
        
        if strategy == "priority":
            lines.append(f"\\textbf{{Priority}} & \\textbf{{{conv}}} & \\textbf{{{success}}} & {cost} & \\textbf{{{saving}}} \\\\")
        else:
            lines.append(f"{strategy.capitalize()} & {conv} & {success} & {cost} & {saving} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)
```

## Output

1. `output/exp4_selective/results.json` — 4 策略消融结果
2. `output/exp4_selective/ablation_table.tex` — LaTeX 对比表
3. 控制台: 策略对比

## Verification

- [ ] `priority` 的收敛轮次 < `full` 的收敛轮次
- [ ] `priority` 的 Test Success ≥ `full` 的 Test Success (性能不降)
- [ ] `priority` 的 Evolution Cost < `full` 的 50% (成本减半)
- [ ] `none` 的 Test Success 最低 (不演化最差)
- [ ] `random` 的表现介于 `none` 和 `full` 之间

## Expected Results

| 策略 | 收敛轮次 | Test Success | Evolution Cost | Cost Saving |
|------|---------|-------------|----------------|-------------|
| **Priority** | **~4** | **~80%** | **~75K** | **~70%** |
| Full | ~6 | ~78% | ~250K | 0% |
| Random | ~8 | ~72% | ~125K | ~50% |
| None | 10 (未收敛) | ~65% | 0 | 100% |

**关键发现**:
- Priority 策略用 30% 的成本达到了与 Full 策略持平甚至更好的性能
- 这验证了 Skill-MAS 的核心假设：不是所有 task 都需要反思，优先反思高 uncertainty + 高 difficulty 的 task 最有效
- Random 策略虽然也节省 50% 成本，但性能下降明显，说明"选哪些 task 反思"比"反思多少 task"更重要
