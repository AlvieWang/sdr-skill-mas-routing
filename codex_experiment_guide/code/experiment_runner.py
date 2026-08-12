"""
SDR x Skill-MAS 统一实验运行器

支持按实验编号运行单个或全部实验。

用法:
    python experiment_runner.py --exp all         # 运行全部实验
    python experiment_runner.py --exp exp1         # 仅运行 Exp 1
    python experiment_runner.py --exp exp1,exp2    # 运行 Exp 1 和 2
    python experiment_runner.py --exp all --verbose # 详细输出

实验列表:
    exp1_baseline       - 基线对比 (Rubar vs RL-PER vs SDR)
    exp1_sdr_extended   - SDR 多轨迹采样 + 分布统计
    exp2_anticollapse   - 反崩溃机制消融
    exp3_dualfeedback   - 双反馈消融
    exp4_selective      - 选择性反思消融
    exp5_transfer       - 跨任务迁移
    exp6_attribution    - 失败归因分析
    exp7_skillmas       - Skill-MAS 指标融合
    exp8_pareto         - Pareto 前沿分析
    analysis            - 统计检验 + 可视化 + 报告
"""
import argparse
import json
import os
import sys
import time

# 添加路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "sdr_eval_pipeline"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "skill_mas_metrics"))
sys.path.insert(0, SCRIPT_DIR)

from config import CONFIG, EXPERIMENT_PATHS


def save_results(results: dict, exp_name: str):
    """保存实验结果到 JSON"""
    output_dir = EXPERIMENT_PATHS.get(exp_name, f"output/{exp_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  Results saved to {path}")


def execute_trajectory(router, trajectory, config, skill_registry=None):
    """
    执行单条轨迹 (委托给 sdr_eval_pipeline)
    
    这是一个代理函数，实际实现在 sdr_eval_pipeline/run_pipeline.py 中。
    如果该模块已导入，则使用其实现；否则使用此处的简化版本。
    """
    try:
        from run_pipeline import execute_trajectory as _exec
        return _exec(router, trajectory, config, skill_registry)
    except ImportError:
        # 简化版: 直接调用 router.route()
        from core.types import StepResult, FailureSource
        import numpy as np
        
        new_steps = list(trajectory.steps)
        new_decisions = []
        new_results = []
        
        prev_failed = False
        budget = 1.0
        
        for ctx in trajectory.steps:
            ctx.previous_step_failed = prev_failed
            ctx.budget_remaining = budget
            
            decision = router.route(ctx)
            new_decisions.append(decision)
            
            # 模拟执行
            model_cost = {"4B": 800, "7B": 1500, "14B": 3000}
            tokens = model_cost.get(decision.selected_model, 1000)
            
            base_success = 0.85
            if decision.selected_model == ctx.gt_model:
                base_success = 0.90
            elif ctx.complexity_score > 0.7 and decision.selected_model == "4B":
                base_success = 0.45
            
            if prev_failed:
                base_success -= 0.15
            
            success = np.random.random() < base_success
            
            result = StepResult(
                step_id=ctx.step_id,
                success=success,
                tokens_used=tokens,
                latency_ms=20.0,
                failure_source=None if success else FailureSource.MODEL_REASONING,
                failure_detail="" if success else "Simulated failure",
            )
            new_results.append(result)
            prev_failed = not success
        
        trajectory.steps = new_steps
        trajectory.routing_decisions = new_decisions
        trajectory.step_results = new_results
        return trajectory


# ============================================================
# 实验注册表
# ============================================================

EXPERIMENTS = {
    "exp1_baseline": "Exp 1: 基线对比 (Rubar vs RL-PER vs SDR)",
    "exp1_sdr_extended": "Exp 1+: SDR 多轨迹采样 + 分布统计",
    "exp2_anticollapse": "Exp 2: 反崩溃机制消融",
    "exp3_dualfeedback": "Exp 3: 双反馈消融",
    "exp4_selective": "Exp 4: 选择性反思消融",
    "exp5_transfer": "Exp 5: 跨任务迁移",
    "exp6_attribution": "Exp 6: 失败归因分析",
    "exp7_skillmas": "Exp 7: Skill-MAS 指标融合",
    "exp8_pareto": "Exp 8: Pareto 前沿分析",
    "analysis": "统计检验 + 可视化 + 报告生成",
}


def run_experiment(exp_id: str, verbose: bool = True):
    """运行指定编号的实验"""
    print(f"\n{'='*60}")
    print(f"  Running: {EXPERIMENTS.get(exp_id, exp_id)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if exp_id == "exp1_baseline":
            # 参考 prompts/02_baseline_eval.md 中的实现
            from exp_runners import run_exp1_baseline
            results = run_exp1_baseline(verbose=verbose)
        
        elif exp_id == "exp1_sdr_extended":
            from exp_runners import run_exp1_sdr_extended
            results = run_exp1_sdr_extended(verbose=verbose)
        
        elif exp_id == "exp2_anticollapse":
            from exp_runners import run_exp2_anticollapse
            results = run_exp2_anticollapse(verbose=verbose)
        
        elif exp_id == "exp3_dualfeedback":
            from exp_runners import run_exp3_dualfeedback
            results = run_exp3_dualfeedback(verbose=verbose)
        
        elif exp_id == "exp4_selective":
            from exp_runners import run_exp4_selective
            results = run_exp4_selective(verbose=verbose)
        
        elif exp_id == "exp5_transfer":
            from exp_runners import run_exp5_transfer
            results = run_exp5_transfer(verbose=verbose)
        
        elif exp_id == "exp6_attribution":
            from exp_runners import run_exp6_attribution
            results = run_exp6_attribution(verbose=verbose)
        
        elif exp_id == "exp7_skillmas":
            from exp_runners import run_exp7_skillmas
            results = run_exp7_skillmas(verbose=verbose)
        
        elif exp_id == "exp8_pareto":
            from exp_runners import run_exp8_pareto
            results = run_exp8_pareto(verbose=verbose)
        
        elif exp_id == "analysis":
            from analysis.report_generator import run_analysis_and_report
            run_analysis_and_report()
            results = None
        
        else:
            print(f"  Unknown experiment: {exp_id}")
            return
        
        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s")
        
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="SDR x Skill-MAS Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--exp", type=str, default="all",
        help="实验编号 (逗号分隔) 或 'all'"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用实验"
    )
    args = parser.parse_args()
    
    if args.list:
        print("Available experiments:")
        for eid, desc in EXPERIMENTS.items():
            print(f"  {eid:25s} - {desc}")
        return
    
    if args.exp == "all":
        for exp_id in EXPERIMENT_PATHS.keys():
            run_experiment(exp_id, args.verbose)
        run_experiment("analysis", args.verbose)
    else:
        for exp_id in args.exp.split(","):
            exp_id = exp_id.strip()
            run_experiment(exp_id, args.verbose)


if __name__ == "__main__":
    main()
