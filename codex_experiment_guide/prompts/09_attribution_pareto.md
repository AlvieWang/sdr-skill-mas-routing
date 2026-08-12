# Codex Prompt 09: 失败归因分析与 Pareto 前沿 (Exp 6 + Exp 8)

## Context

本步骤包含两个实验：
- **Exp 6**: 验证 SDR 的 6 维失败归因能力 (假设 H7: 归因准确率 > 80%)
- **Exp 8**: 在 cost-performance 二维空间中验证 SDR 的 Pareto dominance (假设 H9)

## Input Files

- `sdr_eval_pipeline/metrics/failure_attribution.py` — 失败归因指标
- `sdr_eval_pipeline/metrics/utilization.py` — 利用率指标 (含 Pareto)
- `output/exp1_baseline/results_raw.json` — 基线结果
- `code/config.py` — 实验配置

## Task

### Step 1: 失败归因分析 (Exp 6)

```python
def run_exp6_attribution(verbose=True):
    """
    Exp 6: 失败归因分析
    
    对每个失败 step，SDR 输出失败原因 (6 维)，
    与 ground truth 对比计算归因准确率。
    
    6 维归因:
    - MODEL_REASONING: 模型推理能力不足
    - TOOL_MISSING: 缺少必要工具
    - SKILL_DISCOVERY_WEAK: skill 识别失败
    - WORKSPACE_PERCEPTION: 工作区理解错误
    - NETWORK_FRAGILE: 网络操作失败
    - COMPLETION_CHECK_LOOSE: 完成检查过于宽松
    
    对比: Rubar / RL-PER / SDR
    """
    from metrics.failure_attribution import FailureAttributionMetrics
    
    results = {}
    
    for router_name in ["rubar", "rl_per", "sdr"]:
        router_results = []
        
        for seed in CONFIG.seeds:
            np.random.seed(seed)
            
            for benchmark in CONFIG.benchmarks:
                n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
                config = EvaluationConfig(models=CONFIG.models, benchmark=benchmark, n_tasks=n_tasks)
                
                skill_registry = create_skill_registry(config)
                model_pool = ModelPool(config)
                trajectories = generate_mock_trajectories(config, n_tasks=n_tasks, benchmark=benchmark)
                
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                else:
                    router = SDRRouter(config, model_pool, skill_registry)
                
                executed_trajs = []
                for traj in trajectories:
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                # 计算失败归因指标
                fam = FailureAttributionMetrics()
                attr_metrics = fam.compute(executed_trajs)
                
                router_results.append({
                    "seed": seed,
                    "benchmark": benchmark,
                    **attr_metrics,
                })
        
        results[router_name] = {
            "runs": router_results,
            "mean_attribution_rate": float(np.mean([r.get("attribution_rate", 0) for r in router_results])),
            "mean_discovery_failure": float(np.mean([r.get("discovery_failure_rate", 0) for r in router_results])),
            "mean_false_attribution": float(np.mean([r.get("false_attribution_rate", 0) for r in router_results])),
            "mean_diagnostic_coverage": float(np.mean([r.get("diagnostic_coverage", 0) for r in router_results])),
        }
        
        if verbose:
            r = results[router_name]
            print(f"\n  Router: {router_name}")
            print(f"    Attribution Rate:     {r['mean_attribution_rate']:.3f}")
            print(f"    Discovery Failure:    {r['mean_discovery_failure']:.3f}")
            print(f"    False Attribution:    {r['mean_false_attribution']:.3f}")
            print(f"    Diagnostic Coverage:  {r['mean_diagnostic_coverage']:.3f}")
    
    # 生成混淆矩阵 (SDR only)
    sdr_runs = [r for r in results["sdr"]["runs"] if "per_cause_accuracy" in r]
    if sdr_runs:
        confusion_data = {}
        for cause in ["model_reasoning", "tool_missing", "skill_discovery_weak",
                       "workspace_perception", "network_fragile", "completion_check_loose"]:
            vals = [r.get("per_cause_accuracy", {}).get(cause, 0) for r in sdr_runs]
            confusion_data[cause] = float(np.mean(vals))
        results["sdr"]["per_cause_accuracy"] = confusion_data
    
    save_results(results, "exp6_attribution")
    return results
```

### Step 2: Pareto 前沿分析 (Exp 8)

```python
def run_exp8_pareto(verbose=True):
    """
    Exp 8: Pareto 前沿分析
    
    收集所有路由器在所有 task 上的 (cost, performance) 点，
    计算 Pareto 前沿，验证 SDR 是否 dominate。
    """
    
    # 收集所有 (cost, performance) 点
    points = []
    
    for router_name in ["rubar", "rl_per", "sdr"]:
        for seed in [42]:  # Pareto 分析只需一个 seed
            np.random.seed(seed)
            
            for benchmark in CONFIG.benchmarks:
                n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
                config = EvaluationConfig(models=CONFIG.models, benchmark=benchmark, n_tasks=n_tasks)
                
                skill_registry = create_skill_registry(config)
                model_pool = ModelPool(config)
                trajectories = generate_mock_trajectories(config, n_tasks=n_tasks, benchmark=benchmark)
                
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                else:
                    router = SDRRouter(config, model_pool, skill_registry)
                
                for traj in trajectories:
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    
                    n_success = sum(1 for r in executed.step_results if r.success)
                    n_total = len(executed.step_results)
                    performance = n_success / max(n_total, 1)
                    
                    cost = sum(getattr(r, 'tokens_used', 0) for r in executed.step_results)
                    
                    points.append({
                        "router": router_name,
                        "benchmark": benchmark,
                        "task_id": traj.task_id,
                        "cost": cost,
                        "performance": performance,
                    })
    
    # 计算 Pareto 前沿
    pareto_front = compute_pareto_front(points)
    
    # 统计各路由器在 Pareto 前沿上的占比
    pareto_coverage = {}
    for router_name in ["rubar", "rl_per", "sdr"]:
        pareto_points = [p for p in pareto_front if p["router"] == router_name]
        total_points = [p for p in points if p["router"] == router_name]
        pareto_coverage[router_name] = len(pareto_points) / max(len(total_points), 1)
    
    results = {
        "all_points": points,
        "pareto_front": pareto_front,
        "pareto_coverage": pareto_coverage,
        "pareto_dominance": {
            router: len([p for p in pareto_front if p["router"] == router])
            for router in ["rubar", "rl_per", "sdr"]
        },
    }
    
    if verbose:
        print("\n  Pareto Frontier Analysis:")
        print(f"    Total points: {len(points)}")
        print(f"    Pareto points: {len(pareto_front)}")
        for router in ["rubar", "rl_per", "sdr"]:
            print(f"    {router}: {results['pareto_dominance'][router]} Pareto points "
                  f"({pareto_coverage[router]*100:.1f}% coverage)")
    
    save_results(results, "exp8_pareto")
    return results


def compute_pareto_front(points: list) -> list:
    """
    计算 Pareto 前沿
    
    目标: 最大化 performance, 最小化 cost
    一个点在 Pareto 前沿上，当且仅当不存在另一个点
    在 performance 和 cost 上都优于它。
    """
    pareto = []
    
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i == j:
                continue
            # q dominates p if: q.performance >= p.performance AND q.cost <= p.cost
            # (且至少一个严格不等)
            if (q["performance"] >= p["performance"] and 
                q["cost"] <= p["cost"] and
                (q["performance"] > p["performance"] or q["cost"] < p["cost"])):
                dominated = True
                break
        
        if not dominated:
            pareto.append(p)
    
    return pareto
```

## Output

1. `output/exp6_attribution/results.json` — 3 路由器的失败归因结果
2. `output/exp8_pareto/results.json` — Pareto 前沿数据
3. 控制台: 归因精度 + Pareto 覆盖率

## Verification

### Exp 6
- [ ] SDR 的 Attribution Rate > 0.80
- [ ] Rubar 和 RL-PER 的 Attribution Rate < 0.30 (无 skill 识别能力)
- [ ] SDR 的 False Attribution Rate < 0.15
- [ ] SDR 在 6 个失败原因上都有非零的归因精度

### Exp 8
- [ ] SDR 在 Pareto 前沿上的占比 > 60%
- [ ] RL-PER 在 Pareto 前沿上的占比 < 10%
- [ ] Rubar 在 Pareto 前沿上的占比 < 30%

## Expected Results

### Exp 6: 失败归因

| 路由器 | Attribution Rate | Discovery Failure | False Attribution |
|--------|-----------------|-------------------|-------------------|
| Rubar | ~0.15 | ~0.60 | ~0.70 |
| RL-PER | ~0.20 | ~0.55 | ~0.65 |
| **SDR** | **~0.85** | **~0.10** | **~0.12** |

### Exp 8: Pareto 前沿

| 路由器 | Pareto 点数 | 覆盖率 |
|--------|------------|--------|
| Rubar | ~15 | ~15% |
| RL-PER | ~5 | ~5% |
| **SDR** | **~80** | **~80%** |

**关键发现**:
- SDR 在 Pareto 前沿上 dominate，因为其反崩溃机制确保了成本效率
- RL-PER 几乎不在 Pareto 前沿上，因为路由崩溃导致成本高且性能不均
- SDR 的失败归因能力远超基线，因为 skill 抽象层提供了语义级的失败定位
