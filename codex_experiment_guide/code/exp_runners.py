"""
实验运行器实现

实现各个实验的具体运行逻辑。
"""
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "sdr_eval_pipeline"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "skill_mas_metrics"))
sys.path.insert(0, SCRIPT_DIR)

from config import CONFIG, EXPERIMENT_PATHS

from core.types import EvaluationConfig
from core.skill_registry import SkillRegistry
from core.model_pool import ModelPool
from core.router import RubarRouter, RLPerRouter, SDRRouter
from data.mock_data import create_skill_registry, generate_mock_trajectories
from run_pipeline import execute_trajectory
from metrics.routing_accuracy import RoutingAccuracyMetrics
from metrics.transfer import TransferMetrics
from metrics.utilization import UtilizationMetrics
from metrics.skill_evolution import SkillEvolutionMetrics
from metrics.dual_feedback import DualFeedbackMetrics
from metrics.failure_attribution import FailureAttributionMetrics


def compute_all_metrics(trajectories, router, skill_registry, config):
    """计算全部 6 类指标"""
    metrics = {}
    
    # A: 路由准确率
    ra = RoutingAccuracyMetrics(config)
    metrics["A"] = ra.evaluate(trajectories)
    
    # B: 迁移与适应
    tm = TransferMetrics(config)
    metrics["B"] = tm.evaluate(trajectories)
    
    # C: 利用率与稳定性
    um = UtilizationMetrics(config)
    metrics["C"] = um.evaluate(trajectories)
    
    # D: Skill 演化
    if skill_registry:
        se = SkillEvolutionMetrics(config, skill_registry)
        metrics["D"] = se.evaluate(trajectories)
    
    # E: 双反馈
    df = DualFeedbackMetrics(config)
    metrics["E"] = df.evaluate(trajectories)
    
    # F: 失败归因
    fa = FailureAttributionMetrics(config)
    metrics["F"] = fa.evaluate(trajectories)
    
    return metrics


def save_results(results: dict, exp_name: str):
    """保存实验结果到 JSON"""
    output_dir = EXPERIMENT_PATHS.get(exp_name, f"output/{exp_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  Results saved to {path}")
    return path


def aggregate_results(results: dict) -> dict:
    """
    将 5 个 seed 的结果聚合为 mean ± std
    """
    # 按 (router, benchmark, metric_category, metric_id) 分组
    groups = defaultdict(list)
    for key, metrics in results.items():
        parts = key.split("_")
        router = parts[0]
        
        # 提取 benchmark (可能在中间部分)
        if "swe" in "_".join(parts).lower():
            benchmark = "swe_bench"
        elif "web" in "_".join(parts).lower():
            benchmark = "web_arena"
        else:
            benchmark = "unknown"
        
        for cat, cat_results in metrics.items():
            if isinstance(cat_results, list):
                for result_obj in cat_results:
                    metric_id = result_obj.name.lower().replace(" ", "_").replace("-", "_")
                    val = result_obj.value
                    if isinstance(val, (int, float)):
                        groups[(router, benchmark, cat, metric_id)].append(val)
    
    aggregated = {}
    for (router, bench, cat, mid), vals in groups.items():
        if vals:
            k = f"{router}_{bench}_{cat}_{mid}"
            aggregated[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": vals,
                "n": len(vals)
            }
    
    return aggregated


def generate_comparison_table(aggregated: dict) -> str:
    """生成 LaTeX 格式的对比表格"""
    
    # 核心指标列表
    core_metrics = [
        ("A", "skill_hit@1", "Skill Hit@1"),
        ("C", "routing_entropy", "Routing Entropy"),
        ("C", "routing_collapse_rate", "Collapse Rate"),
        ("C", "skill-level_cost-effectiveness", "Cost-Effectiveness"),
        ("E", "skill-level_plan_f1", "Plan F1"),
        ("E", "skill-level_exec_f1", "Exec F1"),
        ("F", "skill-level_failure_attribution", "Attribution Rate"),
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
                    mean_val = aggregated[k]['mean']
                    row.append(f"{mean_val:.3f}")
                else:
                    # 尝试匹配模糊的键名
                    found = False
                    for ak in aggregated.keys():
                        if router in ak and bench in ak and cat in ak and mid.split("_")[0] in ak:
                            mean_val = aggregated[ak]['mean']
                            row.append(f"{mean_val:.3f}")
                            found = True
                            break
                    if not found:
                        row.append("-")
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def run_exp1_baseline(verbose=True):
    """
    Exp 1: 基线对比实验
    
    3 routers × 2 benchmarks × 5 seeds = 30 runs
    收集全部 29 个 SDR 指标
    """
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    benchmarks = CONFIG.benchmarks
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp1_baseline"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
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
                    # 为每个轨迹设置单独的随机种子
                    np.random.seed(seed + hash(traj.task_id) % 10000)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                # 计算指标
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    # 保存原始结果
    save_results(results, "exp1_baseline")
    
    # 聚合结果
    aggregated = aggregate_results(results)
    agg_path = os.path.join(EXPERIMENT_PATHS["exp1_baseline"], "results_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"  Aggregated results saved to {agg_path}")
    
    # 生成 LaTeX 表格
    latex_table = generate_comparison_table(aggregated)
    latex_path = os.path.join(EXPERIMENT_PATHS["exp1_baseline"], "comparison_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"  LaTeX table saved to {latex_path}")
    
    # 打印核心指标对比
    print("\n  Core Metrics Summary:")
    for metric in [("A", "skill_hit@1"), ("C", "routing_entropy"), ("C", "routing_collapse_rate")]:
        cat, mid = metric
        print(f"    {cat}.{mid}:")
        for router in ["rubar", "rl_per", "sdr"]:
            for bench in ["swe_bench", "web_arena"]:
                k = f"{router}_{bench}_{cat}_{mid}"
                if k in aggregated:
                    print(f"      {router}_{bench}: {aggregated[k]['mean']:.3f} ± {aggregated[k]['std']:.3f}")
    
    return results


def run_exp1_sdr_extended(verbose=True):
    """
    Exp 1+: SDR 多轨迹采样 + 分布统计
    
    类似 Exp 1，但只运行 SDR，并添加 Skill-MAS 的分布统计指标
    """
    results = {}
    benchmarks = CONFIG.benchmarks
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp1_sdr_extended"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
            router = SDRRouter(config, model_pool, skill_registry)
            
            # 执行轨迹 (K=5 多轨迹采样)
            all_executed = []
            for traj in trajectories:
                executed_set = []
                for k in range(CONFIG.rollout_per_task):
                    np.random.seed(seed + hash(traj.task_id) % 10000 + k)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_set.append(executed)
                all_executed.extend(executed_set)
            
            # 计算指标
            key = f"sdr_{benchmark}_seed{seed}"
            results[key] = compute_all_metrics(
                all_executed, router, skill_registry, config
            )
            
            if verbose:
                print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] SDR / {benchmark} (K={CONFIG.rollout_per_task}) done")
    
    save_results(results, "exp1_sdr_extended")
    return results


def run_exp2_anticollapse(verbose=True):
    """
    Exp 2: 反崩溃机制消融
    
    比较不同路由器的崩溃率和熵分布
    """
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in CONFIG.benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp2_anticollapse"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
            for router_name in routers:
                # 创建路由器
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                elif router_name == "sdr":
                    router = SDRRouter(config, model_pool, skill_registry)
                
                executed_trajs = []
                for traj in trajectories:
                    np.random.seed(seed + hash(traj.task_id) % 10000)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    save_results(results, "exp2_anticollapse")
    return results


def run_exp3_dualfeedback(verbose=True):
    """
    Exp 3: 双反馈机制分析
    
    分析路由器的反馈机制
    """
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in CONFIG.benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp3_dualfeedback"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
            for router_name in routers:
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                elif router_name == "sdr":
                    router = SDRRouter(config, model_pool, skill_registry)
                
                executed_trajs = []
                for traj in trajectories:
                    np.random.seed(seed + hash(traj.task_id) % 10000)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    save_results(results, "exp3_dualfeedback")
    return results


def run_exp4_selective(verbose=True):
    """
    Exp 4: 选择性反思策略分析
    
    比较不同路由器的反思和适应能力
    """
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in CONFIG.benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp4_selective"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
            for router_name in routers:
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                elif router_name == "sdr":
                    router = SDRRouter(config, model_pool, skill_registry)
                
                executed_trajs = []
                for traj in trajectories:
                    np.random.seed(seed + hash(traj.task_id) % 10000)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    save_results(results, "exp4_selective")
    return results


def run_exp5_transfer(verbose=True):
    """
    Exp 5: 跨任务迁移实验
    
    迁移路径: SWE→Web / Web→SWE / SWE→ML
    """
    results = {}
    transfer_paths = [
        ("swe_to_web", "swe_bench", "web_arena"),
        ("web_to_swe", "web_arena", "swe_bench"),
    ]
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for path_name, source_bench, target_bench in transfer_paths:
            # 在源域上训练
            n_tasks_source = CONFIG.n_tasks_swe if source_bench == "swe_bench" else CONFIG.n_tasks_web
            config_source = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp5_transfer"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config_source)
            model_pool = ModelPool(config_source)
            trajectories_source = generate_mock_trajectories(n_tasks=n_tasks_source, benchmark=source_bench, config=config_source, seed=seed)
            
            router = SDRRouter(config_source, model_pool, skill_registry)
            
            for traj in trajectories_source:
                np.random.seed(seed + hash(traj.task_id) % 10000)
                execute_trajectory(router, traj, config_source, skill_registry)
            
            # 在目标域上测试
            n_tasks_target = CONFIG.n_tasks_swe if target_bench == "swe_bench" else CONFIG.n_tasks_web
            config_target = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp5_transfer"],
                verbose=verbose
            )
            
            trajectories_target = generate_mock_trajectories(n_tasks=n_tasks_target, benchmark=target_bench, config=config_target, seed=seed)
            
            executed_trajs = []
            for traj in trajectories_target:
                np.random.seed(seed + hash(traj.task_id) % 10000 + 1000)
                executed = execute_trajectory(router, traj, config_target, skill_registry)
                executed_trajs.append(executed)
            
            key = f"{path_name}_seed{seed}"
            results[key] = compute_all_metrics(
                executed_trajs, router, skill_registry, config_target
            )
            
            if verbose:
                print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {path_name} done")
    
    save_results(results, "exp5_transfer")
    return results


def run_exp6_attribution(verbose=True):
    """
    Exp 6: 失败归因分析
    
    分析各路由器的失败归因准确率
    """
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in CONFIG.benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp6_attribution"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
            for router_name in routers:
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                elif router_name == "sdr":
                    router = SDRRouter(config, model_pool, skill_registry)
                
                executed_trajs = []
                for traj in trajectories:
                    np.random.seed(seed + hash(traj.task_id) % 10000)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    save_results(results, "exp6_attribution")
    return results


def run_exp7_skillmas(verbose=True):
    """
    Exp 7: Skill-MAS 指标分析
    
    比较不同路由器的Skill-MAS指标
    """
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in CONFIG.benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp7_skillmas"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
            for router_name in routers:
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                elif router_name == "sdr":
                    router = SDRRouter(config, model_pool, skill_registry)
                
                executed_trajs = []
                for traj in trajectories:
                    np.random.seed(seed + hash(traj.task_id) % 10000)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    save_results(results, "exp7_skillmas")
    return results


def run_exp8_pareto(verbose=True):
    """
    Exp 8: Pareto 前沿分析
    
    收集所有路由器在 cost-performance 空间中的点
    """
    results = {}
    routers = ["rubar", "rl_per", "sdr"]
    
    for seed_idx, seed in enumerate(CONFIG.seeds):
        np.random.seed(seed)
        
        for benchmark in CONFIG.benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(
                models=CONFIG.models,
                output_dir=EXPERIMENT_PATHS["exp8_pareto"],
                verbose=verbose
            )
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(n_tasks=n_tasks, benchmark=benchmark, config=config, seed=seed)
            
            for router_name in routers:
                if router_name == "rubar":
                    router = RubarRouter(config, model_pool)
                elif router_name == "rl_per":
                    router = RLPerRouter(config, model_pool)
                elif router_name == "sdr":
                    router = SDRRouter(config, model_pool, skill_registry)
                
                executed_trajs = []
                for traj in trajectories:
                    np.random.seed(seed + hash(traj.task_id) % 10000)
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                key = f"{router_name}_{benchmark}_seed{seed}"
                results[key] = compute_all_metrics(
                    executed_trajs, router, skill_registry, config
                )
                
                if verbose:
                    print(f"  [{seed_idx+1}/{len(CONFIG.seeds)}] {router_name} / {benchmark} done")
    
    save_results(results, "exp8_pareto")
    return results