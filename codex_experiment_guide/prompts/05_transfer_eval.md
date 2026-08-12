# Codex Prompt 05: 跨任务迁移实验 (Exp 5)

## Context

验证 SDR 的 skill 画像能否跨任务域迁移。使用 Meta-RL 的 episode 记忆机制，将在源域 (如 SWE-bench) 学到的 skill 后验分布迁移到目标域 (如 WebArena)，观察迁移增益。

**假设 H6**: SWE-bench 上学到的 skill 迁移到 WebArena 后仍获得 > 10pp 增益

## Input Files

- `sdr_eval_pipeline/core/skill_registry.py` — Skill 注册表 (含 Beta-Bernoulli 后验)
- `sdr_eval_pipeline/core/router.py` — SDRRouter (含 Meta-RL 记忆)
- `sdr_eval_pipeline/data/mock_data.py` — SWE-bench 和 WebArena 数据生成器
- `code/config.py` — 实验配置

## Task

### Step 1: 实现迁移实验运行器

创建 `code/transfer_runner.py`：

```python
"""
跨任务 Skill 迁移实验

迁移路径:
1. SWE-bench → WebArena (代码→网页)
2. WebArena → SWE-bench (网页→代码)  
3. SWE-bench → MLAgentBench (代码→ML)

对比:
- with_transfer: 使用源域训练的 skill registry
- no_transfer: 使用空白 skill registry
"""
import sys
import os
import json
import numpy as np
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sdr_eval_pipeline"))

from core.types import EvaluationConfig
from core.skill_registry import SkillRegistry
from core.model_pool import ModelPool
from core.router import SDRRouter
from data.mock_data import create_skill_registry, generate_mock_trajectories


def train_on_source_domain(
    source_benchmark: str,
    n_tasks: int = 50,
    n_rounds: int = 5,
    seed: int = 42
) -> SkillRegistry:
    """
    在源域上训练 skill registry
    
    过程:
    1. 创建初始 skill registry
    2. 在源域任务上执行 K 轮
    3. 每轮更新 skill 的 Beta-Bernoulli 后验
    4. 返回训练好的 skill registry
    """
    np.random.seed(seed)
    
    config = EvaluationConfig(models=["4B", "7B", "14B"], benchmark=source_benchmark, n_tasks=n_tasks)
    skill_registry = create_skill_registry(config)
    model_pool = ModelPool(config)
    trajectories = generate_mock_trajectories(config, n_tasks=n_tasks, benchmark=source_benchmark)
    
    router = SDRRouter(config, model_pool, skill_registry)
    
    # 训练多轮
    for round_idx in range(n_rounds):
        for traj in trajectories:
            executed = execute_trajectory(router, traj, config, skill_registry)
            
            # 更新 skill 后验
            for i, result in enumerate(executed.step_results):
                decision = executed.routing_decisions[i] if i < len(executed.routing_decisions) else None
                if decision and decision.predicted_skills:
                    for skill_name in decision.predicted_skills[:2]:
                        if skill_name in skill_registry.skills:
                            skill = skill_registry.skills[skill_name]
                            if result.success:
                                skill.alpha += 1
                            else:
                                skill.beta += 1
    
    return skill_registry


def evaluate_on_target_domain(
    skill_registry: SkillRegistry,
    target_benchmark: str,
    n_tasks: int = 50,
    seed: int = 42
) -> Dict:
    """
    在目标域上评估
    
    使用给定的 skill_registry (可能是迁移的或空白的)
    """
    np.random.seed(seed + 1000)
    
    config = EvaluationConfig(models=["4B", "7B", "14B"], benchmark=target_benchmark, n_tasks=n_tasks)
    model_pool = ModelPool(config)
    trajectories = generate_mock_trajectories(config, n_tasks=n_tasks, benchmark=target_benchmark)
    
    router = SDRRouter(config, model_pool, skill_registry)
    
    # 执行评估
    total_success = 0
    total_steps = 0
    total_tokens = 0
    skill_hits = 0
    total_skills = 0
    
    for traj in trajectories:
        executed = execute_trajectory(router, traj, config, skill_registry)
        
        for i, result in enumerate(executed.step_results):
            total_steps += 1
            total_tokens += getattr(result, 'tokens_used', 0)
            if result.success:
                total_success += 1
            
            # Skill hit rate
            decision = executed.routing_decisions[i] if i < len(executed.routing_decisions) else None
            if decision and decision.predicted_skills:
                total_skills += 1
                if any(s in traj.steps[i].gt_skills for s in decision.predicted_skills):
                    skill_hits += 1
    
    return {
        "success_rate": total_success / max(total_steps, 1),
        "avg_tokens": total_tokens / max(len(trajectories), 1),
        "skill_hit_rate": skill_hits / max(total_skills, 1),
        "total_tasks": len(trajectories),
    }


def compute_transfer_metrics(with_transfer: Dict, without_transfer: Dict) -> Dict:
    """计算迁移增益"""
    return {
        "success_rate_delta": with_transfer["success_rate"] - without_transfer["success_rate"],
        "skill_hit_delta": with_transfer["skill_hit_rate"] - without_transfer["skill_hit_rate"],
        "token_efficiency": without_transfer["avg_tokens"] / max(with_transfer["avg_tokens"], 1),
        "with_transfer": with_transfer,
        "without_transfer": without_transfer,
    }


def run_exp5_transfer(verbose=True):
    """
    Exp 5: 跨任务迁移实验
    
    3 条迁移路径 × 5 seeds × {with_transfer, without_transfer}
    """
    transfer_paths = [
        ("swe_bench", "web_arena", "SWE→Web"),
        ("web_arena", "swe_bench", "Web→SWE"),
        ("swe_bench", "mlagent_bench", "SWE→ML"),
    ]
    
    all_results = {}
    
    for source, target, label in transfer_paths:
        path_results = []
        
        for seed in CONFIG.seeds:
            # 训练源域
            trained_registry = train_on_source_domain(source, seed=seed)
            
            # 评估目标域 (with transfer)
            with_transfer = evaluate_on_target_domain(trained_registry, target, seed=seed)
            
            # 评估目标域 (without transfer - 空白 registry)
            config = EvaluationConfig(models=["4B", "7B", "14B"], benchmark=source, n_tasks=50)
            blank_registry = create_skill_registry(config)
            without_transfer = evaluate_on_target_domain(blank_registry, target, seed=seed)
            
            # 计算迁移增益
            transfer_metrics = compute_transfer_metrics(with_transfer, without_transfer)
            transfer_metrics["seed"] = seed
            path_results.append(transfer_metrics)
        
        # 聚合
        all_results[label] = {
            "source": source,
            "target": target,
            "runs": path_results,
            "mean_success_delta": float(np.mean([r["success_rate_delta"] for r in path_results])),
            "mean_skill_hit_delta": float(np.mean([r["skill_hit_delta"] for r in path_results])),
            "mean_token_efficiency": float(np.mean([r["token_efficiency"] for r in path_results])),
        }
        
        if verbose:
            r = all_results[label]
            print(f"\n  {label}:")
            print(f"    Success rate delta: {r['mean_success_delta']:+.4f} ({r['mean_success_delta']*100:+.1f}pp)")
            print(f"    Skill hit delta:    {r['mean_skill_hit_delta']:+.4f}")
            print(f"    Token efficiency:   {r['mean_token_efficiency']:.2f}x")
    
    save_results(all_results, "exp5_transfer")
    return all_results
```

### Step 2: 生成迁移热力图数据

```python
def generate_transfer_heatmap_data(results: dict) -> dict:
    """
    生成迁移热力图数据 (类似 Skill-MAS Figure 3)
    
    行: Source domain
    列: Target domain
    值: Success rate delta (pp)
    """
    domains = ["swe_bench", "web_arena", "mlagent_bench"]
    domain_labels = ["SWE-bench", "WebArena", "MLAgentBench"]
    
    heatmap = {}
    for source_label in domain_labels:
        heatmap[source_label] = {}
        for target_label in domain_labels:
            if source_label == target_label:
                heatmap[source_label][target_label] = 0.0  # 对角线
            else:
                # 查找对应的迁移结果
                for key, val in results.items():
                    if val["source"] in source and val["target"] in target:
                        heatmap[source_label][target_label] = val["mean_success_delta"] * 100
                        break
    
    return heatmap
```

## Output

1. `output/exp5_transfer/results.json` — 3 路径 × 5 seeds 的迁移结果
2. `output/exp5_transfer/heatmap_data.json` — 迁移热力图数据
3. `code/transfer_runner.py` — 迁移实验代码
4. 控制台: 迁移增益对比

## Verification

- [ ] 3 条迁移路径全部完成
- [ ] SWE→Web 的 success_rate_delta > 0.10 (10pp)
- [ ] 同域迁移 (对角线) 的增益最大
- [ ] 跨域迁移仍为正值 (skill 可迁移)
- [ ] skill_hit_rate 的迁移增益 > 0 (迁移后 skill 识别更准)

## Expected Results

| 迁移路径 | Success Δ (pp) | Skill Hit Δ | Token Eff. |
|----------|----------------|-------------|------------|
| SWE→Web | +12-18 | +0.15-0.25 | 1.2-1.4x |
| Web→SWE | +8-14 | +0.10-0.20 | 1.1-1.3x |
| SWE→ML | +5-10 | +0.05-0.15 | 1.0-1.2x |

**关键发现**: SWE-bench → WebArena 的迁移增益最大，因为代码域的 skill (如 code_gen, debug, verify) 在 WebArena 中也有部分适用 (如 form_fill 需要代码理解)。
