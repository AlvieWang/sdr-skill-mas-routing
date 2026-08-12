"""
SDR x Skill-MAS 实验统一配置

所有实验共享的配置参数，包括模型池、benchmark、重复次数、
路由器参数、反崩溃阈值、双反馈开关等。
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    """实验全局配置"""
    
    # === 模型池 ===
    models: List[str] = field(default_factory=lambda: ["4B", "7B", "14B"])
    
    # === Benchmark ===
    benchmarks: List[str] = field(default_factory=lambda: ["swe_bench", "web_arena"])
    
    # === 实验重复 ===
    n_runs: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    
    # === 任务数 ===
    n_tasks_swe: int = 50
    n_tasks_web: int = 50
    
    # === Skill 演化 (Skill-MAS 参数) ===
    evolution_rounds: int = 10
    rollout_per_task: int = 5  # K=5 (Skill-MAS 推荐值)
    
    # === SDR 路由器参数 ===
    sdr_capability_weight: float = 0.6
    sdr_cost_eff_weight: float = 0.4
    sdr_softmax_temp: float = 0.15
    
    # === 反崩溃 ===
    collapse_threshold: float = 0.80
    
    # === 双反馈 ===
    enable_pre_execution: bool = True
    enable_post_execution: bool = True
    
    # === 输出 ===
    output_dir: str = "output"
    
    # === 统计 ===
    alpha: float = 0.05
    bootstrap_samples: int = 10000


# 全局配置实例
CONFIG = ExperimentConfig()

# 实验名称到输出路径的映射
EXPERIMENT_PATHS = {
    "exp1_baseline": "output/exp1_baseline",
    "exp1_sdr_extended": "output/exp1_sdr_extended",
    "exp2_anticollapse": "output/exp2_anticollapse",
    "exp3_dualfeedback": "output/exp3_dualfeedback",
    "exp4_selective": "output/exp4_selective",
    "exp5_transfer": "output/exp5_transfer",
    "exp6_attribution": "output/exp6_attribution",
    "exp7_skillmas": "output/exp7_skillmas",
    "exp8_pareto": "output/exp8_pareto",
}

# 核心指标列表 (用于快速引用)
CORE_METRICS = {
    "A": {
        "skill_hit_at_1": "Skill Hit@1",
        "skill_mrr_at_10": "Skill MRR@10",
        "model_match_rate": "Model Match Rate",
    },
    "C": {
        "routing_entropy": "Routing Entropy",
        "routing_collapse_rate": "Collapse Rate",
        "cost_effectiveness": "Cost-Effectiveness",
        "utilization_balance": "Utilization Balance",
    },
    "E": {
        "plan_f1": "Plan F1",
        "exec_f1": "Exec F1",
        "pre_post_match": "Pre-Post Match",
    },
    "F": {
        "attribution_rate": "Attribution Rate",
        "discovery_failure_rate": "Discovery Failure",
    },
}

# 假设列表 (用于报告生成)
HYPOTHESES = {
    "H1": "SDR Skill Hit@1 显著高于 Rubar 和 RL-PER (p<0.05)",
    "H2": "SDR 路由崩溃率 < 5%，RL-PER > 50%",
    "H3": "移除反崩溃机制后 SDR 路由熵下降 > 50%",
    "H4": "双反馈机制使 Plan F1 提升 > 15pp",
    "H5": "优先级演化比全量反思节省 > 50% 成本且性能不降",
    "H6": "SWE-bench skill 迁移到 WebArena 仍获 > 10pp 增益",
    "H7": "SDR 失败归因准确率 > 80%",
    "H8": "引入 Skill-MAS uncertainty 后收敛速度提升 > 30%",
    "H9": "SDR 在 Pareto 前沿上 dominate 两个基线",
}
