# Codex Prompt 02: 基线对比实验 (Exp 1)

## Context

实验目标：对比三个路由器 (Rubar, RL-PER, SDR) 在 SWE-bench 和 WebArena 上的全面表现，收集全部 6 类 29 个指标，使用 5 个随机种子重复实验。

这是所有后续消融实验的基础，需要确保数据完整且可复现。

## Input Files

- `sdr_eval_pipeline/run_pipeline.py` — 主 pipeline 入口
- `sdr_eval_pipeline/core/router.py` — 3 个路由器实现
- `sdr_eval_pipeline/metrics/` — 6 类指标模块
- `sdr_eval_pipeline/data/mock_data.py` — 数据生成器
- `code/config.py` — 实验配置

## Task

### Step 1: 实现基线对比实验运行器

在 `code/experiment_runner.py` 中实现 `run_experiment("exp1_baseline")` 函数：

```python
def run_exp1_baseline(verbose=True):
    """
    Exp 1: 基线对比实验
    
    3 routers × 2 benchmarks × 5 seeds = 30 runs
    收集全部 29 个 SDR 指标
    """
    import numpy as np
    from core.types import EvaluationConfig
    from core.skill_registry import SkillRegistry
    from core.model_pool import ModelPool
    from core.router import RubarRouter, RLPerRouter, SDRRouter
    from data.mock_data import create_skill_registry, generate_mock_trajectories
    from metrics.routing_accuracy import RoutingAccuracyMetrics
    from metrics.transfer import TransferMetrics
    from metrics.utilization import UtilizationMetrics
    from metrics.skill_evolution import SkillEvolutionMetrics
    from metrics.dual_feedback import DualFeedbackMetrics
    from metrics.failure_attribution import FailureAttributionMetrics
    
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    benchmarks = CONFIG.benchmarks
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                benchmark=benchmark,
                n_tasks=n_tasks
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(config, n_tasks=n_tasks, benchmark=benchmark)
            
            for router_name in routers:
                # 创建路由器
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                elif router_name == "sdr":
                    router = SDRRouter(config, model_pool, skill_registry)
                
                # 执行轨迹
                executed_trajs = []
                for traj in trajectories:
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                # 计算指标
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    # 保存结果
    save_results(results, "exp1_baseline")
    return results


def compute_all_metrics(trajectories, router, skill_registry, config):
    """计算全部 6 类指标"""
    metrics = {}
    
    # A: 路由准确率
    ra = RoutingAccuracyMetrics()
    metrics["A"] = ra.compute(trajectories, router)
    
    # B: 迁移与适应
    tm = TransferMetrics()
    metrics["B"] = tm.compute(trajectories, skill_registry)
    
    # C: 利用率与稳定性
    um = UtilizationMetrics()
    metrics["C"] = um.compute(trajectories, config)
    
    # D: Skill 演化
    se = SkillEvolutionMetrics()
    metrics["D"] = se.compute(skill_registry)
    
    # E: 双反馈
    df = DualFeedbackMetrics()
    metrics["E"] = df.compute(trajectories)
    
    # F: 失败归因
    fa = FailureAttributionMetrics()
    metrics["F"] = fa.compute(trajectories)
    
    return metrics
```

### Step 2: 实现统计聚合

在同一文件中添加结果聚合函数：

```python
def aggregate_results(results: dict) -> dict:
    """
    将 5 个 seed 的结果聚合为 mean ± std
    """
    import numpy as np
    from collections import defaultdict
    
    # 按 (router, benchmark, metric_category, metric_id) 分组
    groups = defaultdict(list)
    for key, metrics in results.items():
        parts = key.split("_")
        router = parts[0]
        benchmark = "_".join(parts[1:-1])
        for cat, cat_metrics in metrics.items():
            for mid, val in cat_metrics.items():
                groups[(router, benchmark, cat, mid)].append(val)
    
    aggregated = {}
    for (router, bench, cat, mid), vals in groups.items():
        vals = [v for v in vals if isinstance(v, (int, float))]
        if vals:
            k = f"{router}_{bench}_{cat}_{mid}"
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": vals
            }
    
    return aggregated
```

### Step 3: 生成对比表格

```python
def generate_comparison_table(aggregated: dict) -> str:
    """生成 LaTeX 格式的对比表格"""
    
    # 核心指标列表
    core_metrics = [
        ("A", "skill_hit_at_1", "Skill Hit@1"),
        ("C", "routing_entropy", "Routing Entropy"),
        ("C", "routing_collapse_rate", "Collapse Rate"),
        ("C", "cost_effectiveness", "Cost-Effectiveness"),
        ("E", "plan_f1", "Plan F1"),
        ("E", "exec_f1", "Exec F1"),
        ("F", "attribution_rate", "Attribution Rate"),
    ]
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Baseline Comparison: Rubar vs RL-PER vs SDR}")
    lines.append("\\label{tab:baseline}")
    lines.append("\\begin{tabular}{l|ccc|ccc}")
    lines.append("\\toprule")
    lines.append(" & \\multicolumn{3}{c|}{SWE-bench} & \\multicolumn{3}{c}{WebArena} \\\\")
    lines.append("Metric & Rubar & RL-PER & SDR & Rubar & RL-PER & SDR \\\\")
    lines.append("\\midrule")
    
    for cat, mid, label in core_metrics:
        row = [label]
        for router in ["rubar", "rl_per", "sdr"]:
            for bench in ["swe_bench", "web_arena"]:
                k = f"{router}_{bench}_{cat}_{mid}"
                if k in aggregated:
                    row.append(f"{aggregated[k]['mean']:.3f}")
                else:
                    row.append("-")
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)
```

## Output

1. `output/exp1_baseline/results_raw.json` — 原始结果 (30 runs)
2. `output/exp1_baseline/results_aggregated.json` — 聚合结果 (mean ± std)
3. `output/exp1_baseline/comparison_table.tex` — LaTeX 对比表
4. 控制台输出: 3 路由器 × 2 benchmark 的核心指标对比

## Verification

- [ ] 30 个 run 全部完成 (3 × 2 × 5)
- [ ] SDR 的 Skill Hit@1 > 0 (Rubar 和 RL-PER 应为 0)
- [ ] RL-PER 的 Routing Collapse Rate > 0.5
- [ ] SDR 的 Routing Collapse Rate < 0.05
- [ ] SDR 的 Total Tokens < Rubar 的 Total Tokens
- [ ] LaTeX 表格可正确编译

## Expected Results (参考值)

| 指标 | Rubar | RL-PER | SDR |
|------|-------|--------|-----|
| Skill Hit@1 | 0.000 | 0.000 | 0.600-0.700 |
| Routing Entropy | ~1.0 | ~0.5 | ~1.4 |
| Routing Collapse | 0.000 | 0.500-0.900 | 0.000-0.050 |
| Cost-Effectiveness | ~0.35 | ~0.27 | ~0.57 |
| Total Tokens | ~187K | ~231K | ~131K |
| Plan F1 | 0.000 | 0.000 | ~0.510 |
