# Skill-MAS Evaluation Metrics Extraction Module

从 Skill-MAS 官方 GitHub 仓库 ([linhh29/Skill-MAS](https://github.com/linhh29/Skill-MAS)) 提取的全部评测指标计算代码，整理为可独立运行的 Python 模块。

## 源仓库

- **GitHub**: https://github.com/linhh29/Skill-MAS
- **项目页**: https://linhh29.github.io/blog/Skill-MAS/index.html
- **Demo**: https://skill-mas-demo.hehailin.life/
- **论文**: arXiv:2606.18837 (蚂蚁集团 × 港科大)

## 指标体系总览

共 **7 类、22 个评测维度**，每个指标都标注了源代码文件位置：

| 类别 | 指标数 | 源文件 | 核心公式 |
|------|--------|--------|---------|
| A. 主性能 | 3 | `evolution/assemble_select.py` | `round_score = mean(per_task_mean)` |
| B. 分布统计 | 5 | `evolution/elbow_selection.py` | `priority = (norm(std) + norm(-mean)) / 2` |
| C. 选择性反思 | 2 | `evolution/elbow_selection.py` | 二阶差分肘部检测 |
| D. 迁移性 | 3 | `evolution/contrastive_reflect.py` | `source_gap = high_score - low_score` |
| E. 成本 | 4 | `utils/llm_cost.py` | `cost = (prompt/1M)*in_rate + (output/1M)*out_rate` |
| F. 演化追踪 | 3 | `evolution/bank_optimizer.py` | knee_index + score_trajectory |
| G. Benchmark | 2+ | `dataset/*/score.py` | 各benchmark特定评分 |

## 指标详细定义

### A. 主性能指标 (Main Performance)

来源: `evolution/assemble_select.py` → `compute_round_score()`, `finalize_best_round()`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Avg.Perf** | `(1/|T|) * Σ_t [(1/K) * Σ_k score(t,k)]` | 每个任务取K条轨迹均值，再跨任务取均值 |
| **Per-Benchmark Score** | `mean(scores_t)` | 单个任务的平均分 |
| **Best Round** | `argmax(round_score)` + tie-break | 最优轮次选择 (分数→复杂度→稳定性) |

### B. 分布统计指标 (Distributional Stats)

来源: `evolution/elbow_selection.py` → `_priority_vectors()`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Uncertainty** | `σ_i = std(trajectory_scores_i)` | 每个任务K条轨迹分数的总体标准差 (ddof=0) |
| **Difficulty** | `d_i = -mean(trajectory_scores_i)` | 负均值，分数越低→难度越大 |
| **U_norm** | `minmax(uncertainties)` | 跨任务 min-max 归一化 |
| **D_norm** | `minmax(difficulties)` | 跨任务 min-max 归一化 |
| **Priority** | `p_i = (U_norm_i + D_norm_i) / 2` | 混合优先级分数 [0,1] |

### C. 选择性反思指标 (Selective Reflection)

来源: `evolution/elbow_selection.py` → `adaptive_elbow_count()`, `second_diff_elbow_detail()`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Elbow Index** | `argmax(|second_diffs|) + 1` | 降序排列 priority 曲线的最大拐点 |
| **Selected Count** | `min(elbow_idx * sensitivity, max_cases, n)` | 实际选择反思的任务数 |

二阶差分计算:
```
diffs[i] = sorted[i] - sorted[i+1]           # 一阶差分
second_diffs[i] = diffs[i] - diffs[i+1]       # 二阶差分
elbow_idx = argmax(|second_diffs|) + 1
```

### D. 迁移性指标 (Transferability)

来源: `evolution/contrastive_reflect.py` → `DomainPatch.source_gap`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Within-task Contrast (source_gap)** | `max(scores) - min(scores)` | 同一任务高/低分轨迹的差距 |
| **Cross-LLM Transfer Δ** | `mean(target_scores) - mean(source_scores)` | 跨模型迁移增益 |
| **Cross-task Transfer** | per-task delta matrix | 跨任务迁移矩阵 (heatmap 数据) |

### E. 成本指标 (Cost Metrics)

来源: `utils/llm_cost.py` → `vita_rollout_cost_report()`, `build_round_cost_document()`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Inference Cost** | `(prompt/1M)*in_rate + (output/1M)*out_rate` | rollout阶段每次LLM调用的USD成本 |
| **Evolution Cost** | `Σ(optimizer_calls cost)` | 反思+bank优化阶段的LLM调用成本 |
| **Round Total Cost** | `inference_cost + evolution_cost` | 单轮总成本 |
| **Cumulative Cost** | `Σ(round_total)` | 累积跨轮成本 |

### F. 演化追踪指标 (Evolution Tracking)

来源: `evolution/bank_optimizer.py` → `_write_knee_artifacts()`, `assemble_select.py` → `finalize_best_round()`

| 指标 | 公式 | 说明 |
|------|------|------|
| **Knee Index** | elbow on priority curve | 优先级曲线的肘部位置 |
| **Score Trajectory** | `[(round_idx, round_score), ...]` | 跨轮分数轨迹 |
| **Convergence Round** | 首次连续3轮提升 < 0.5pp | 分数收敛轮次 |
| **Improvement** | `best_score - baseline_score` | 相对基线的提升 |

### G. 各 Benchmark 评分 (Per-Benchmark Scoring)

| Benchmark | 源文件 | 评分方式 | 输出范围 |
|-----------|--------|---------|---------|
| **HLE-Math** | `dataset/hlemath/score.py` | `\boxed{}` 提取 + sympy 符号等价 | {0, 1} |
| **BrowseComp-Plus** | `dataset/BrowseComp-Plus/score.py` | 归一化精确匹配 / LLM Judge | {0, 1} |
| **DeepResearch Bench** | `dataset/deep_research_bench/utils/score_calculator.py` | 加权多维度 LLM Judge | [0, 10] |
| **VitaBench** | `dataset/vitabench/src/vita/metrics/agent_metrics.py` | NL Assertion Ratio + pass^k | [0, 1] |

#### VitaBench 特有指标:
- **NL Assertion Ratio**: `satisfied_rubrics / total_rubrics`
- **pass^k** = `C(c,k) / C(n,k)` (随机选k个全成功的概率)
- **pass@k** = `1 - C(n-c,k) / C(n,k)` (至少1个成功的概率)
- **average@k** = `mean(rewards)`

## 使用方法

### 基本用法

```python
from skill_mas_metrics import (
    SkillMASMetricsReport,
    TrajectoryRecord,
    PhaseSnapshot,
)

# 准备轨迹数据
by_task = {
    "task_001": [
        TrajectoryRecord(
            schema="skill_mas_trajectory_record_v1",
            bench_backend="vitabench",
            round_idx=3,
            task_id="task_001",
            trajectory_idx=0,
            trajectory_tag="task_001_traj_00",
            score=0.75,
            score_source="vitabench_nl_assertion_ratio",
            log_path="...",
            raw_result_path="...",
        ),
        # ... more trajectories
    ],
}

# 生成完整报告
report = SkillMASMetricsReport.generate_full_report(
    by_task=by_task,
    round_idx=3,
    optimizer_usage=[
        {"phase": "contrastive_reflection_phase1", "model": "gpt-4o",
         "usage": {"prompt_tokens": 8000, "output_tokens": 2000}},
    ],
    rollout_usage=[
        {"prompt_tokens": 12000, "output_tokens": 3000, "total_tokens": 15000},
    ],
    model="gpt-4o",
)

print(report["A_main_performance"]["round_score"])
```

### 单独使用各类指标

```python
from skill_mas_metrics import (
    MainPerformanceMetrics,
    DistributionalMetrics,
    SelectiveReflectionMetrics,
    TransferabilityMetrics,
    CostMetrics,
    EvolutionTrackingMetrics,
    HLEMATHScorer,
    BrowseCompScorer,
    VitaBenchMetrics,
    DRBScorer,
)

# A. 主性能
score = MainPerformanceMetrics.compute_round_score(by_task)

# B. 分布统计
priorities = DistributionalMetrics.compute_priority_scores(samples_scores)

# C. 选择性反思
selected_ids, report = SelectiveReflectionMetrics.compute_reflection_task_selection(
    task_rows, max_reflection_cases=5
)

# D. 迁移性
gap, high, low = TransferabilityMetrics.compute_source_gap(trajectories)

# E. 成本
cost = CostMetrics.estimate_cost(prompt_tokens=1000, output_tokens=500, model="gpt-4o")

# F. 演化追踪
evo_report = EvolutionTrackingMetrics.compute_round_priority_report(by_task, round_idx=3)

# G. Benchmark 评分
score, pred = HLEMATHScorer.calculate_score("\\boxed{42}", "The answer is \\boxed{42}")
score, pred = BrowseCompScorer.calculate_score("Paris", "paris")
ratio = VitaBenchMetrics.nl_assertion_ratio(rubrics)
```

### 运行自测

```bash
cd skill_mas_metrics
python3 skill_mas_metrics.py
```

## 模块结构

```
skill_mas_metrics/
├── skill_mas_metrics.py    # 主模块 (~900行, 含7类22个指标)
└── README.md               # 本文档
```

## 源代码文件映射

| 本模块类 | 源仓库文件 | 源函数 |
|---------|-----------|--------|
| `MainPerformanceMetrics` | `evolution/assemble_select.py` | `compute_round_score()`, `finalize_best_round()` |
| `DistributionalMetrics` | `evolution/elbow_selection.py` | `_priority_vectors()`, `_population_std()`, `_normalize_minmax_1d()` |
| `SelectiveReflectionMetrics` | `evolution/elbow_selection.py` | `adaptive_elbow_count()`, `second_diff_elbow_detail()`, `compute_reflection_task_selection()` |
| `TransferabilityMetrics` | `evolution/contrastive_reflect.py` | `DomainPatch.source_gap` 字段计算 |
| `CostMetrics` | `utils/llm_cost.py` | `vita_rollout_cost_report()`, `build_round_cost_document()`, `merge_cumulative_summary()` |
| `EvolutionTrackingMetrics` | `evolution/bank_optimizer.py` | `_write_knee_artifacts()` |
| `HLEMATHScorer` | `dataset/hlemath/score.py` | `HLEMATHScorer.calculate_score()` |
| `BrowseCompScorer` | `dataset/BrowseComp-Plus/score.py` | `BrowseCompScorer.calculate_score()` |
| `VitaBenchMetrics` | `dataset/vitabench/src/vita/metrics/agent_metrics.py` | `pass_hat_k()`, `pass_at_k()`, `average_at_k()`, `compute_metrics()` |
| `DRBScorer` | `dataset/deep_research_bench/utils/score_calculator.py` | `calculate_weighted_scores()` |
| `TrajectoryRecord` | `evolution/schemas.py` | `TrajectoryRecord` dataclass |
| `DomainPatch` | `evolution/schemas.py` | `DomainPatch` dataclass |

## 依赖

- Python 3.9+ (无第三方依赖，仅用标准库)
- 如需完整 sympy 符号比较，需安装: `pip install sympy regex`
