# Codex Prompt 07: 双反馈消融实验 (Exp 3)

## Context

验证 SDR 的双反馈机制（预执行评估 + 后执行评估）对路由决策质量的贡献。预执行评估是"前瞻"——在路由前预测模型适配度；后执行评估是"后顾"——在执行后评估实际表现。

**假设 H4**: 双反馈机制使 Plan F1 提升 > 15pp

## Input Files

- `sdr_eval_pipeline/metrics/dual_feedback.py` — 双反馈指标模块
- `sdr_eval_pipeline/core/router.py` — SDRRouter (含双反馈逻辑)
- `code/config.py` — 实验配置

## Task

### Step 1: 创建双反馈消融变体

在 `code/ablation_router_variants.py` 中添加双反馈变体：

```python
class SDRPreOnly(SDRRouter):
    """变体: 仅预执行反馈 (移除后执行评估)"""
    
    def _post_execution_eval(self, step_context, decision, result):
        """禁用后执行评估"""
        return None
    
    def _update_with_feedback(self, step_context, decision, result):
        """仅使用预执行信号更新"""
        pre_score = decision.model_distribution.get(decision.selected_model, 0.5)
        if pre_score > 0.7:
            # 高预执行分数 → 更新 skill
            for skill_name in decision.predicted_skills[:1]:
                if skill_name in self.skill_registry.skills:
                    self.skill_registry.skills[skill_name].alpha += 0.5  # 弱更新


class SDRPostOnly(SDRRouter):
    """变体: 仅后执行反馈 (移除预执行评估)"""
    
    def _pre_execution_eval(self, step_context, skills, conditions):
        """禁用预执行评估"""
        return 0.5  # 中性分数
    
    def _update_with_feedback(self, step_context, decision, result):
        """仅使用后执行信号更新"""
        if result.success:
            for skill_name in decision.predicted_skills[:1]:
                if skill_name in self.skill_registry.skills:
                    self.skill_registry.skills[skill_name].alpha += 1
        else:
            for skill_name in decision.predicted_skills[:1]:
                if skill_name in self.skill_registry.skills:
                    self.skill_registry.skills[skill_name].beta += 1


class SDRNoFeedback(SDRRouter):
    """变体: 无反馈 (静态路由, 不更新 skill)"""
    
    def _pre_execution_eval(self, step_context, skills, conditions):
        return 0.5
    
    def _post_execution_eval(self, step_context, decision, result):
        return None
    
    def _update_with_feedback(self, step_context, decision, result):
        pass  # 不更新


# 添加到变体注册表
DUAL_FEEDBACK_VARIANTS = {
    "full": SDRRouter,        # 预执行 + 后执行
    "pre_only": SDRPreOnly,   # 仅预执行
    "post_only": SDRPostOnly,  # 仅后执行
    "none": SDRNoFeedback,    # 无反馈
}
```

### Step 2: 运行双反馈消融

在 `code/ablation_runner.py` 中添加：

```python
def run_exp3_dualfeedback(verbose=True):
    """
    Exp 3: 双反馈消融实验
    
    4 变体 × 2 benchmark × 5 seeds
    重点指标: E1 (Pre-Post Match), E2 (Feedback Gap), E3 (Plan F1), E4 (Exec F1)
    """
    from ablation_router_variants import DUAL_FEEDBACK_VARIANTS
    from metrics.dual_feedback import DualFeedbackMetrics
    
    results = {}
    
    for variant_name, RouterClass in DUAL_FEEDBACK_VARIANTS.items():
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
                
                executed_trajs = []
                for traj in trajectories:
                    executed = execute_trajectory(router, traj, config, skill_registry)
                    executed_trajs.append(executed)
                
                # 计算双反馈指标
                dfm = DualFeedbackMetrics()
                feedback_metrics = dfm.compute(executed_trajs)
                
                # 计算路由准确率 (A 类指标作为对照)
                from metrics.routing_accuracy import RoutingAccuracyMetrics
                ra = RoutingAccuracyMetrics()
                accuracy_metrics = ra.compute(executed_trajs, router)
                
                variant_results.append({
                    "seed": seed,
                    "benchmark": benchmark,
                    "pre_post_match": feedback_metrics.get("pre_post_match", 0),
                    "feedback_gap": feedback_metrics.get("feedback_gap", 0),
                    "plan_f1": feedback_metrics.get("plan_f1", 0),
                    "exec_f1": feedback_metrics.get("exec_f1", 0),
                    "skill_hit_at_1": accuracy_metrics.get("skill_hit_at_1", 0),
                    "model_match_rate": accuracy_metrics.get("model_match_rate", 0),
                })
        
        results[variant_name] = {
            "runs": variant_results,
            "mean_pre_post_match": float(np.mean([r["pre_post_match"] for r in variant_results])),
            "mean_feedback_gap": float(np.mean([r["feedback_gap"] for r in variant_results])),
            "mean_plan_f1": float(np.mean([r["plan_f1"] for r in variant_results])),
            "mean_exec_f1": float(np.mean([r["exec_f1"] for r in variant_results])),
            "mean_skill_hit": float(np.mean([r["skill_hit_at_1"] for r in variant_results])),
            "mean_model_match": float(np.mean([r["model_match_rate"] for r in variant_results])),
        }
        
        if verbose:
            r = results[variant_name]
            print(f"\n  Variant: {variant_name}")
            print(f"    Pre-Post Match: {r['mean_pre_post_match']:.3f}")
            print(f"    Feedback Gap:   {r['mean_feedback_gap']:.3f}")
            print(f"    Plan F1:        {r['mean_plan_f1']:.3f}")
            print(f"    Exec F1:        {r['mean_exec_f1']:.3f}")
            print(f"    Skill Hit@1:    {r['mean_skill_hit']:.3f}")
    
    # 计算消融增益
    full_plan_f1 = results["full"]["mean_plan_f1"]
    for variant in results:
        results[variant]["plan_f1_drop_pp"] = (full_plan_f1 - results[variant]["mean_plan_f1"]) * 100
    
    save_results(results, "exp3_dualfeedback")
    return results
```

## Output

1. `output/exp3_dualfeedback/results.json` — 4 变体消融结果
2. 控制台: 双反馈指标对比

## Verification

- [ ] `full` 的 Plan F1 最高
- [ ] `none` 的 Plan F1 最低
- [ ] `full` 的 Pre-Post Match 最高 (预执行和后执行评估最一致)
- [ ] `pre_only` 的 Feedback Gap 最大 (没有后执行来校正)
- [ ] `full` vs `none` 的 Plan F1 差距 > 15pp

## Expected Results

| 变体 | Pre-Post Match | Feedback Gap | Plan F1 | Exec F1 | Skill Hit@1 |
|------|---------------|-------------|---------|---------|-------------|
| **full** | **~0.75** | **~0.15** | **~0.51** | **~0.68** | **~0.63** |
| pre_only | ~0.60 | ~0.35 | ~0.40 | ~0.55 | ~0.55 |
| post_only | ~0.55 | ~0.25 | ~0.35 | ~0.60 | ~0.50 |
| none | ~0.45 | ~0.45 | ~0.20 | ~0.40 | ~0.35 |

**关键发现**:
- 预执行反馈主要改善 Plan F1 (前瞻性路由决策)
- 后执行反馈主要改善 Exec F1 (经验积累)
- 两者结合效果 > 任何单一反馈 (非加性效应)
- 无反馈时 Skill Hit@1 仍 > 0 (因为 skill registry 本身有先验知识)
