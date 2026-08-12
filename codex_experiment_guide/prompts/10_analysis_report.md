# Codex Prompt 10: 统计检验、可视化与自动报告生成

## Context

所有 8 个实验已完成，数据保存在 `output/` 目录下。本步骤对全部实验结果进行统计检验、生成可视化图表，并自动生成论文级别的实验报告。

## Input Files

- `output/exp1_baseline/results_aggregated.json`
- `output/exp2_anticollapse/results.json`
- `output/exp3_dualfeedback/results.json`
- `output/exp4_selective/results.json`
- `output/exp5_transfer/results.json`
- `output/exp6_attribution/results.json`
- `output/exp7_skillmas/results.json`
- `output/exp8_pareto/results.json`

## Task

### Step 1: 统计检验模块

创建 `code/analysis/statistical_tests.py`：

```python
"""
统计检验模块

对所有实验结果进行:
1. Welch's t-test (两组对比)
2. One-way ANOVA + Tukey HSD (多组对比)
3. Mann-Whitney U (非参数替代)
4. Cohen's d (效应量)
5. Bootstrap 95% CI
"""
import json
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats


def welch_t_test(group1: List[float], group2: List[float]) -> Dict:
    """Welch's t-test (不假设等方差)"""
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    
    # Cohen's d
    pooled_std = np.sqrt(
        ((len(group1) - 1) * np.var(group1, ddof=1) + 
         (len(group2) - 1) * np.var(group2, ddof=1)) /
        (len(group1) + len(group2) - 2)
    )
    cohens_d = (np.mean(group1) - np.mean(group2)) / max(pooled_std, 1e-8)
    
    # Bootstrap CI
    ci = bootstrap_ci(group1, group2)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "cohens_d": float(cohens_d),
        "effect_size": interpret_effect_size(cohens_d),
        "mean_diff": float(np.mean(group1) - np.mean(group2)),
        "ci_95": ci,
    }


def anova_tukey(groups: Dict[str, List[float]]) -> Dict:
    """One-way ANOVA + Tukey HSD"""
    from scipy.stats import f_oneway
    
    group_values = list(groups.values())
    f_stat, p_value = f_oneway(*group_values)
    
    # Tukey HSD (需要 statsmodels)
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        data = []
        labels = []
        for name, vals in groups.items():
            data.extend(vals)
            labels.extend([name] * len(vals))
        tukey = pairwise_tukeyhsd(data, labels, alpha=0.05)
        tukey_summary = tukey.summary().as_html()
    except ImportError:
        tukey_summary = "statsmodels not installed"
    
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "tukey_hsd": tukey_summary,
    }


def bootstrap_ci(group1: List[float], group2: List[float], 
                n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap 置信区间"""
    diffs = []
    for _ in range(n_bootstrap):
        s1 = np.random.choice(group1, size=len(group1), replace=True)
        s2 = np.random.choice(group2, size=len(group2), replace=True)
        diffs.append(np.mean(s1) - np.mean(s2))
    
    lower = float(np.percentile(diffs, (1 - ci) / 2 * 100))
    upper = float(np.percentile(diffs, (1 + ci) / 2 * 100))
    return (lower, upper)


def interpret_effect_size(d: float) -> str:
    """解释 Cohen's d 效应量"""
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"


def run_all_statistical_tests(output_dir: str = "output"):
    """对所有实验结果进行统计检验"""
    results = {}
    
    # Exp 1: SDR vs Rubar vs RL-PER
    exp1_path = os.path.join(output_dir, "exp1_baseline", "results_raw.json")
    if os.path.exists(exp1_path):
        with open(exp1_path) as f:
            exp1 = json.load(f)
        
        # 对每个核心指标做 t-test: SDR vs Rubar, SDR vs RL-PER
        core_metrics = ["skill_hit_at_1", "routing_entropy", "routing_collapse_rate", 
                       "cost_effectiveness", "plan_f1"]
        
        for metric in core_metrics:
            for benchmark in ["swe_bench", "web_arena"]:
                sdr_vals = extract_metric_values(exp1, "sdr", benchmark, metric)
                rubar_vals = extract_metric_values(exp1, "rubar", benchmark, metric)
                rlper_vals = extract_metric_values(exp1, "rl_per", benchmark, metric)
                
                if sdr_vals and rubar_vals:
                    key = f"exp1_{metric}_{benchmark}_sdr_vs_rubar"
                    results[key] = welch_t_test(sdr_vals, rubar_vals)
                
                if sdr_vals and rlper_vals:
                    key = f"exp1_{metric}_{benchmark}_sdr_vs_rlper"
                    results[key] = welch_t_test(sdr_vals, rlper_vals)
    
    # Exp 2: 反崩溃消融 ANOVA
    exp2_path = os.path.join(output_dir, "exp2_anticollapse", "results.json")
    if os.path.exists(exp2_path):
        with open(exp2_path) as f:
            exp2 = json.load(f)
        
        collapse_groups = {}
        for variant, data in exp2.items():
            collapse_groups[variant] = [r["routing_collapse_rate"] for r in data["runs"]]
        
        results["exp2_anticollapse_anova"] = anova_tukey(collapse_groups)
    
    # 保存
    with open(os.path.join(output_dir, "statistical_tests.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def extract_metric_values(results, router, benchmark, metric):
    """从原始结果中提取特定指标的所有值"""
    vals = []
    for key, metrics in results.items():
        if f"{router}_{benchmark}" in key:
            # 遍历指标类别查找
            for cat_metrics in metrics.values():
                if isinstance(cat_metrics, dict) and metric in cat_metrics:
                    val = cat_metrics[metric]
                    if isinstance(val, (int, float)):
                        vals.append(val)
    return vals
```

### Step 2: 可视化模块

创建 `code/analysis/visualization.py`：

```python
"""
可视化模块

生成以下图表:
1. 雷达图: 3 路由器 × 6 类指标
2. 柱状图: 反崩溃消融对比
3. 散点图: 双反馈一致性
4. 热力图: 跨任务迁移
5. 混淆矩阵: 失败归因
6. Pareto 前沿散点图
7. 演化收敛曲线
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


def plot_radar_chart(exp1_results, output_path):
    """雷达图: 3 路由器 × 6 类指标"""
    
    categories = ['Routing\nAccuracy', 'Transfer', 'Utilization', 
                  'Skill\nEvolution', 'Dual\nFeedback', 'Failure\nAttribution']
    
    routers = {'Rubar': 'rubar', 'RL-PER': 'rl_per', 'SDR': 'sdr'}
    colors = {'Rubar': '#3498db', 'RL-PER': '#e74c3c', 'SDR': '#2ecc71'}
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for router_name, router_key in routers.items():
        values = []
        for cat in ['A', 'B', 'C', 'D', 'E', 'F']:
            # 获取该类别的平均指标值
            cat_metrics = exp1_results.get(f"{router_key}_swe_bench", {}).get(cat, {})
            if cat_metrics:
                vals = [v for v in cat_metrics.values() if isinstance(v, (int, float))]
                values.append(np.mean(vals) if vals else 0)
            else:
                values.append(0)
        
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=router_name, color=colors[router_name])
        ax.fill(angles, values, alpha=0.15, color=colors[router_name])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('SDR vs Baselines: 6-Category Metric Comparison', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_collapse_ablation(exp2_results, output_path):
    """柱状图: 反崩溃消融对比"""
    
    variants = list(exp2_results.keys())
    collapse_rates = [exp2_results[v]["mean_collapse"] for v in variants]
    entropies = [exp2_results[v]["mean_entropy"] for v in variants]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    
    # Collapse Rate
    bars1 = ax1.bar(variants, collapse_rates, color=colors)
    ax1.set_ylabel('Routing Collapse Rate', fontsize=12)
    ax1.set_title('Anti-Collapse: Collapse Rate', fontsize=13)
    ax1.axhline(y=0.05, color='red', linestyle='--', label='Threshold (5%)')
    ax1.legend()
    for bar, val in zip(bars1, collapse_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    
    # Entropy
    bars2 = ax2.bar(variants, entropies, color=colors)
    ax2.set_ylabel('Routing Entropy (bits)', fontsize=12)
    ax2.set_title('Anti-Collapse: Entropy', fontsize=13)
    for bar, val in zip(bars2, entropies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_transfer_heatmap(exp5_results, output_path):
    """热力图: 跨任务迁移"""
    
    domains = ['SWE→Web', 'Web→SWE', 'SWE→ML']
    deltas = [exp5_results[d]["mean_success_delta"] * 100 for d in domains]
    
    fig, ax = plt.subplots(figsize=(8, 3))
    
    matrix = np.array([deltas])
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=20)
    
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, fontsize=11)
    ax.set_yticks([0])
    ax.set_yticklabels(['Success Rate Δ (pp)'], fontsize=11)
    
    for i, val in enumerate(deltas):
        ax.text(i, 0, f'{val:+.1f}', ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Δ (pp)')
    ax.set_title('Cross-Task Transfer Heatmap', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pareto_frontier(exp8_results, output_path):
    """Pareto 前沿散点图"""
    
    points = exp8_results["all_points"]
    pareto = exp8_results["pareto_front"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'rubar': '#3498db', 'rl_per': '#e74c3c', 'sdr': '#2ecc71'}
    labels = {'rubar': 'Rubar', 'rl_per': 'RL-PER', 'sdr': 'SDR'}
    
    for router in ['rubar', 'rl_per', 'sdr']:
        r_points = [p for p in points if p["router"] == router]
        x = [p["cost"] / 1000 for p in r_points]  # K tokens
        y = [p["performance"] * 100 for p in r_points]  # %
        ax.scatter(x, y, c=colors[router], label=labels[router], alpha=0.5, s=30)
    
    # Pareto 前沿
    pareto_sdr = [p for p in pareto if p["router"] == "sdr"]
    if pareto_sdr:
        px = [p["cost"] / 1000 for p in sorted(pareto_sdr, key=lambda p: p["cost"])]
        py = [p["performance"] * 100 for p in sorted(pareto_sdr, key=lambda p: p["cost"])]
        ax.plot(px, py, 'g-', linewidth=2, label='SDR Pareto Front')
    
    ax.set_xlabel('Cost (K tokens)', fontsize=12)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('Cost-Performance Pareto Frontier', fontsize=13)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_evolution_convergence(exp7_results, output_path):
    """演化收敛曲线"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    strategies = {'priority': '#2ecc71', 'full': '#3498db', 'random': '#f39c12', 'none': '#e74c3c'}
    
    for strategy, color in strategies.items():
        if strategy in exp7_results:
            runs = exp7_results[strategy]["runs"]
            # 平均每个 round 的 score
            max_rounds = max(len(r["evolution_log"]) for r in runs)
            avg_scores = []
            for r_idx in range(max_rounds):
                scores = [r["evolution_log"][r_idx]["mean_score"] 
                         for r in runs if r_idx < len(r["evolution_log"])]
                avg_scores.append(np.mean(scores))
            
            ax.plot(range(len(avg_scores)), avg_scores, 'o-', 
                   label=strategy.capitalize(), color=color, linewidth=2)
    
    ax.set_xlabel('Evolution Round', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Skill Evolution Convergence', fontsize=13)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_visualizations(output_dir: str = "output"):
    """生成全部可视化图表"""
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Exp 1: 雷达图
    exp1_path = os.path.join(output_dir, "exp1_baseline", "results_aggregated.json")
    if os.path.exists(exp1_path):
        with open(exp1_path) as f:
            exp1 = json.load(f)
        plot_radar_chart(exp1, os.path.join(viz_dir, "exp1_radar.png"))
    
    # Exp 2: 反崩溃柱状图
    exp2_path = os.path.join(output_dir, "exp2_anticollapse", "results.json")
    if os.path.exists(exp2_path):
        with open(exp2_path) as f:
            exp2 = json.load(f)
        plot_collapse_ablation(exp2, os.path.join(viz_dir, "exp2_collapse.png"))
    
    # Exp 5: 迁移热力图
    exp5_path = os.path.join(output_dir, "exp5_transfer", "results.json")
    if os.path.exists(exp5_path):
        with open(exp5_path) as f:
            exp5 = json.load(f)
        plot_transfer_heatmap(exp5, os.path.join(viz_dir, "exp5_transfer.png"))
    
    # Exp 7: 收敛曲线
    exp7_path = os.path.join(output_dir, "exp7_skillmas", "results.json")
    if os.path.exists(exp7_path):
        with open(exp7_path) as f:
            exp7 = json.load(f)
        plot_evolution_convergence(exp7, os.path.join(viz_dir, "exp7_convergence.png"))
    
    # Exp 8: Pareto 前沿
    exp8_path = os.path.join(output_dir, "exp8_pareto", "results.json")
    if os.path.exists(exp8_path):
        with open(exp8_path) as f:
            exp8 = json.load(f)
        plot_pareto_frontier(exp8, os.path.join(viz_dir, "exp8_pareto.png"))
    
    print(f"Visualizations saved to {viz_dir}/")
```

### Step 3: 自动报告生成

创建 `code/analysis/report_generator.py`：

```python
"""
自动报告生成器

读取所有实验结果，生成结构化的 Markdown 实验报告，
包含表格、统计检验结果和图表引用。
"""
import json
import os
from datetime import datetime


def generate_experiment_report(output_dir: str = "output") -> str:
    """生成完整的实验报告"""
    
    report = []
    report.append("# SDR x Skill-MAS 实验报告\n")
    report.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    # === Exp 1: 基线对比 ===
    report.append("## 1. 基线对比实验 (Exp 1)\n")
    exp1 = load_json(os.path.join(output_dir, "exp1_baseline", "results_aggregated.json"))
    if exp1:
        report.append("### 核心指标对比\n")
        report.append("| 指标 | Rubar | RL-PER | SDR | 显著性 |")
        report.append("|------|-------|--------|-----|--------|")
        # ... 填充数据
        report.append("")
    
    # === Exp 2: 反崩溃消融 ===
    report.append("## 2. 反崩溃机制消融 (Exp 2)\n")
    exp2 = load_json(os.path.join(output_dir, "exp2_anticollapse", "results.json"))
    if exp2:
        report.append("| 变体 | Collapse Rate | Entropy | Gini |")
        report.append("|------|--------------|---------|------|")
        for variant in ["full", "no_cost_eff", "no_entropy_reg", "rl_per_style"]:
            if variant in exp2:
                r = exp2[variant]
                report.append(f"| {variant} | {r['mean_collapse']:.3f} | {r['mean_entropy']:.3f} | {r['mean_gini']:.3f} |")
        report.append("")
        report.append("![反崩溃消融](visualizations/exp2_collapse.png)\n")
    
    # === Exp 3: 双反馈消融 ===
    report.append("## 3. 双反馈消融 (Exp 3)\n")
    exp3 = load_json(os.path.join(output_dir, "exp3_dualfeedback", "results.json"))
    if exp3:
        report.append("| 变体 | Plan F1 | Exec F1 | Pre-Post Match |")
        report.append("|------|---------|---------|----------------|")
        for variant in ["full", "pre_only", "post_only", "none"]:
            if variant in exp3:
                r = exp3[variant]
                report.append(f"| {variant} | {r['mean_plan_f1']:.3f} | {r['mean_exec_f1']:.3f} | {r['mean_pre_post_match']:.3f} |")
        report.append("")
    
    # === Exp 4: 选择性反思 ===
    report.append("## 4. 选择性反思消融 (Exp 4)\n")
    exp4 = load_json(os.path.join(output_dir, "exp4_selective", "results.json"))
    if exp4:
        report.append("| 策略 | 收敛轮次 | Test Success | 成本节省 |")
        report.append("|------|---------|-------------|---------|")
        for strategy in ["priority", "full", "random", "none"]:
            if strategy in exp4:
                r = exp4[strategy]
                report.append(f"| {strategy} | {r['mean_convergence']:.1f} | {r['mean_test_success']*100:.1f}% | {r['cost_saving_pct']:.1f}% |")
        report.append("")
    
    # === Exp 5: 迁移实验 ===
    report.append("## 5. 跨任务迁移 (Exp 5)\n")
    exp5 = load_json(os.path.join(output_dir, "exp5_transfer", "results.json"))
    if exp5:
        report.append("| 迁移路径 | Success Δ (pp) | Skill Hit Δ | Token Eff. |")
        report.append("|----------|----------------|-------------|------------|")
        for path in ["SWE→Web", "Web→SWE", "SWE→ML"]:
            if path in exp5:
                r = exp5[path]
                report.append(f"| {path} | {r['mean_success_delta']*100:+.1f} | {r['mean_skill_hit_delta']:+.3f} | {r['mean_token_efficiency']:.2f}x |")
        report.append("")
        report.append("![迁移热力图](visualizations/exp5_transfer.png)\n")
    
    # === Exp 6: 失败归因 ===
    report.append("## 6. 失败归因 (Exp 6)\n")
    exp6 = load_json(os.path.join(output_dir, "exp6_attribution", "results.json"))
    if exp6:
        report.append("| 路由器 | Attribution Rate | False Attribution |")
        report.append("|--------|-----------------|-------------------|")
        for router in ["rubar", "rl_per", "sdr"]:
            if router in exp6:
                r = exp6[router]
                report.append(f"| {router} | {r['mean_attribution_rate']:.3f} | {r['mean_false_attribution']:.3f} |")
        report.append("")
    
    # === Exp 7: Skill-MAS 融合 ===
    report.append("## 7. Skill-MAS 指标融合 (Exp 7)\n")
    exp7 = load_json(os.path.join(output_dir, "exp7_skillmas", "results.json"))
    if exp7:
        report.append("| 策略 | 收敛轮次 | Final Score | Total Updates |")
        report.append("|------|---------|-------------|---------------|")
        for strategy in ["priority", "full", "random", "none"]:
            if strategy in exp7:
                r = exp7[strategy]
                report.append(f"| {strategy} | {r['mean_convergence']:.1f} | {r['mean_final_score']:.4f} | {r['mean_total_updates']:.0f} |")
        report.append("")
        report.append("![收敛曲线](visualizations/exp7_convergence.png)\n")
    
    # === Exp 8: Pareto 前沿 ===
    report.append("## 8. Pareto 前沿 (Exp 8)\n")
    exp8 = load_json(os.path.join(output_dir, "exp8_pareto", "results.json"))
    if exp8:
        report.append("| 路由器 | Pareto 点数 | 覆盖率 |")
        report.append("|--------|------------|--------|")
        for router in ["rubar", "rl_per", "sdr"]:
            count = exp8["pareto_dominance"].get(router, 0)
            coverage = exp8["pareto_coverage"].get(router, 0)
            report.append(f"| {router} | {count} | {coverage*100:.1f}% |")
        report.append("")
        report.append("![Pareto 前沿](visualizations/exp8_pareto.png)\n")
    
    # === 统计检验 ===
    report.append("## 9. 统计检验\n")
    stats_path = os.path.join(output_dir, "statistical_tests.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats_results = json.load(f)
        
        report.append("| 对比 | p-value | Cohen's d | 效应量 | 显著 |")
        report.append("|------|---------|-----------|--------|------|")
        for key, val in stats_results.items():
            if isinstance(val, dict) and "p_value" in val:
                report.append(f"| {key} | {val['p_value']:.4f} | {val['cohens_d']:.3f} | {val['effect_size']} | {'✓' if val['significant'] else '✗'} |")
        report.append("")
    
    # === 结论 ===
    report.append("## 10. 结论\n")
    report.append("### 假设验证汇总\n")
    report.append("| 假设 | 内容 | 验证结果 |")
    report.append("|------|------|---------|")
    report.append("| H1 | SDR Skill Hit@1 显著高于基线 | 待填 |")
    report.append("| H2 | SDR 路由崩溃率 < 5% | 待填 |")
    report.append("| H3 | 移除反崩溃后熵下降 > 50% | 待填 |")
    report.append("| H4 | 双反馈使 Plan F1 提升 > 15pp | 待填 |")
    report.append("| H5 | 优先级演化节省 > 50% 成本 | 待填 |")
    report.append("| H6 | 跨任务迁移增益 > 10pp | 待填 |")
    report.append("| H7 | 失败归因准确率 > 80% | 待填 |")
    report.append("| H8 | Skill-MAS 指标提升收敛 > 30% | 待填 |")
    report.append("| H9 | SDR 在 Pareto 前沿 dominate | 待填 |")
    
    report_text = "\n".join(report)
    
    # 保存
    report_path = os.path.join(output_dir, "EXPERIMENT_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    
    return report_text


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
```

### Step 4: 运行全部分析和报告生成

```python
def run_analysis_and_report():
    """运行统计检验、可视化和报告生成"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from analysis.statistical_tests import run_all_statistical_tests
    from analysis.visualization import generate_all_visualizations
    from analysis.report_generator import generate_experiment_report
    
    print("=" * 60)
    print("Running Statistical Tests...")
    print("=" * 60)
    stats = run_all_statistical_tests()
    print(f"  {len(stats)} tests completed")
    
    print("\n" + "=" * 60)
    print("Generating Visualizations...")
    print("=" * 60)
    generate_all_visualizations()
    
    print("\n" + "=" * 60)
    print("Generating Experiment Report...")
    print("=" * 60)
    report = generate_experiment_report()
    print(f"  Report saved to output/EXPERIMENT_REPORT.md")
    print(f"  Report length: {len(report)} characters")
    
    print("\n" + "=" * 60)
    print("All analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_analysis_and_report()
```

## Output

1. `output/statistical_tests.json` — 全部统计检验结果
2. `output/visualizations/exp1_radar.png` — 雷达图
3. `output/visualizations/exp2_collapse.png` — 反崩溃柱状图
4. `output/visualizations/exp5_transfer.png` — 迁移热力图
5. `output/visualizations/exp7_convergence.png` — 收敛曲线
6. `output/visualizations/exp8_pareto.png` — Pareto 前沿图
7. `output/EXPERIMENT_REPORT.md` — 完整实验报告

## Verification

- [ ] 统计检验覆盖全部核心指标对比
- [ ] 所有 p-value < 0.05 的对比标记为显著
- [ ] 5 张可视化图表全部生成
- [ ] 实验报告包含全部 8 个实验的结果表格
- [ ] 假设验证汇总表已填充
- [ ] 报告中的图片路径正确
