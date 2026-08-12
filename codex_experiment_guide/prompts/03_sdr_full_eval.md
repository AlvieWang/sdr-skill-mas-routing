# Codex Prompt 03: SDR 完整评估与多轨迹采样 (Exp 1 扩展)

## Context

在 Exp 1 基线对比的基础上，对 SDR 路由器进行更深入的评估。引入 Skill-MAS 的多轨迹采样机制 (K=5 rollout per task)，计算 step-level uncertainty 和 difficulty 指标。

## Input Files

- `sdr_eval_pipeline/core/router.py` — SDRRouter 实现
- `sdr_eval_pipeline/core/skill_registry.py` — Skill 注册表
- `output/exp1_baseline/results_raw.json` — Exp 1 基线结果
- `skill_mas_metrics/skill_mas_metrics.py` — Skill-MAS 指标模块

## Task

### Step 1: 实现多轨迹采样

在 `code/experiment_runner.py` 中添加多轨迹采样函数：

```python
def multi_trajectory_rollout(
    router, 
    trajectories, 
    config, 
    skill_registry,
    K: int = 5,
    seed: int = 42
):
    """
    Skill-MAS 风格的多轨迹采样
    
    对每个 task 执行 K 次独立 rollout，
    记录每次的得分用于计算 uncertainty 和 difficulty。
    
    参数:
        K: 每个 task 的 rollout 次数 (Skill-MAS 推荐值为 5)
    
    返回:
        rollout_results: {task_id: [score_1, score_2, ..., score_K]}
    """
    import numpy as np
    np.random.seed(seed)
    
    rollout_results = {}
    
    for traj in trajectories:
        task_id = traj.task_id
        scores = []
        
        for k in range(K):
            # 每次 rollout 使用不同的随机种子
            np.random.seed(seed * 1000 + k)
            
            # 执行轨迹
            executed = execute_trajectory(router, traj, config, skill_registry)
            
            # 计算该 task 的得分
            # 得分 = 成功 step 数 / 总 step 数
            n_success = sum(1 for r in executed.step_results if r.success)
            n_total = len(executed.step_results)
            score = n_success / max(n_total, 1)
            scores.append(score)
        
        rollout_results[task_id] = scores
    
    return rollout_results
```

### Step 2: 计算分布统计指标

```python
def compute_distributional_metrics(rollout_results: dict):
    """
    计算 Skill-MAS 的分布统计指标
    
    对每个 task:
    - mean_score: K 次 rollout 的平均得分
    - uncertainty: K 次 rollout 的标准差
    - difficulty: -mean_score (越低越难)
    
    然后计算:
    - normalized_uncertainty: min-max 归一化后的 uncertainty
    - normalized_difficulty: min-max 归一化后的 difficulty
    - priority_score: 0.5 * (norm_u + norm_d)
    """
    import numpy as np
    
    task_stats = {}
    
    for task_id, scores in rollout_results.items():
        mean_score = float(np.mean(scores))
        uncertainty = float(np.std(scores))
        difficulty = -mean_score
        
        task_stats[task_id] = {
            "mean": mean_score,
            "uncertainty": uncertainty,
            "difficulty": difficulty,
            "scores": scores
        }
    
    # Min-max 归一化
    all_u = [s["uncertainty"] for s in task_stats.values()]
    all_d = [s["difficulty"] for s in task_stats.values()]
    
    u_min, u_max = min(all_u), max(all_u)
    d_min, d_max = min(all_d), max(all_d)
    
    for task_id, stats in task_stats.items():
        norm_u = (stats["uncertainty"] - u_min) / max(u_max - u_min, 1e-8)
        norm_d = (stats["difficulty"] - d_min) / max(d_max - d_min, 1e-8)
        priority = 0.5 * (norm_u + norm_d)
        
        stats["normalized_uncertainty"] = norm_u
        stats["normalized_difficulty"] = norm_d
        stats["priority_score"] = priority
    
    return task_stats
```

### Step 3: 实现肘部检测

```python
def elbow_detection(task_stats: dict):
    """
    Skill-MAS 的二阶差分肘部检测
    
    将 task 按 priority_score 降序排列，
    找到二阶差分最大的位置作为截断点。
    
    返回:
        selected_tasks: 被选中的 task_id 列表
        elbow_index: 截断位置
    """
    import numpy as np
    
    # 按 priority 降序排列
    sorted_tasks = sorted(
        task_stats.items(),
        key=lambda x: x[1]["priority_score"],
        reverse=True
    )
    
    priorities = [s["priority_score"] for _, s in sorted_tasks]
    n = len(priorities)
    
    if n <= 2:
        return [t for t, _ in sorted_tasks], n
    
    # 计算一阶差分
    first_diffs = [priorities[j] - priorities[j+1] for j in range(n-1)]
    
    # 计算二阶差分
    second_diffs = [abs(first_diffs[j] - first_diffs[j+1]) for j in range(n-2)]
    
    # 找到二阶差分最大的位置
    elbow_idx = int(np.argmax(second_diffs)) + 1
    
    selected = [t for t, _ in sorted_tasks[:elbow_idx]]
    
    return selected, elbow_idx
```

### Step 4: 运行 SDR 完整评估

```python
def run_exp1_sdr_extended(verbose=True):
    """
    Exp 1 扩展: SDR 完整评估 + 多轨迹采样
    """
    import numpy as np
    from core.types import EvaluationConfig
    from core.skill_registry import SkillRegistry
    from core.model_pool import ModelPool
    from core.router import SDRRouter
    from data.mock_data import create_skill_registry, generate_mock_trajectories
    
    all_results = {}
    
    for seed in CONFIG.seeds:
        np.random.seed(seed)
        
        for benchmark in CONFIG.benchmarks:
            n_tasks = CONFIG.n_tasks_swe if benchmark == "swe_bench" else CONFIG.n_tasks_web
            config = EvaluationConfig(models=CONFIG.models, benchmark=benchmark, n_tasks=n_tasks)
            
            skill_registry = create_skill_registry(config)
            model_pool = ModelPool(config)
            trajectories = generate_mock_trajectories(config, n_tasks=n_tasks, benchmark=benchmark)
            
            # 创建 SDR 路由器
            router = SDRRouter(config, model_pool, skill_registry)
            
            # 多轨迹采样 (K=5)
            rollout_results = multi_trajectory_rollout(
                router, trajectories, config, skill_registry,
                K=CONFIG.rollout_per_task, seed=seed
            )
            
            # 分布统计
            task_stats = compute_distributional_metrics(rollout_results)
            
            # 肘部检测
            selected_tasks, elbow_idx = elbow_detection(task_stats)
            
            key = f"sdr_{benchmark}_seed{seed}"
            all_results[key] = {
                "rollout_results": rollout_results,
                "task_stats": task_stats,
                "elbow_index": elbow_idx,
                "selected_tasks": selected_tasks,
            }
            
            if verbose:
                print(f"  seed={seed} benchmark={benchmark}")
                print(f"    Elbow index: {elbow_idx}/{n_tasks}")
                print(f"    Mean uncertainty: {np.mean([s['uncertainty'] for s in task_stats.values()]):.4f}")
                print(f"    Mean difficulty: {np.mean([s['difficulty'] for s in task_stats.values()]):.4f}")
    
    save_results(all_results, "exp1_sdr_extended")
    return all_results
```

## Output

1. `output/exp1_sdr_extended/results.json` — 多轨迹采样 + 分布统计结果
2. 控制台输出: 每个 seed × benchmark 的肘部位置和分布统计

## Verification

- [ ] K=5 rollout 全部完成
- [ ] 每个 task 有 5 个 score 值
- [ ] uncertainty 和 difficulty 已正确计算
- [ ] priority_score = 0.5 * (norm_u + norm_d) 在 [0, 1] 范围内
- [ ] 肘部检测返回的 index 在 [1, n_tasks] 范围内
- [ ] 高 uncertainty 的 task 被优先选中

## Key Insight

这一步引入了 Skill-MAS 的核心创新——通过多轨迹采样获取 task 的分布信息，从而驱动后续的选择性反思。这些指标是 Exp 4 (选择性反思消融) 和 Exp 7 (Skill-MAS 指标融合) 的基础。
