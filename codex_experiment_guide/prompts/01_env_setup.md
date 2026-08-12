# Codex Prompt 01: 环境搭建与验证

## Context

你正在为 AAAI 2027 论文搭建 SDR (Skill-Driven Dynamic Routing) 评估实验环境。已有两个代码包：
1. `sdr_eval_pipeline/` — SDR 评估 pipeline (3 路由器 + 6 类 29 个指标)
2. `skill_mas_metrics/` — Skill-MAS 指标提取模块 (7 类 22 个指标)

需要搭建统一的实验环境，确保两个模块可以联合运行。

## Input Files

- `sdr_eval_pipeline/` 目录 (含 core/, metrics/, data/, run_pipeline.py)
- `skill_mas_metrics/skill_mas_metrics.py`

## Task

### Step 1: 创建实验配置文件

创建 `code/config.py`，包含以下配置：

```python
"""
SDR x Skill-MAS 实验统一配置
"""
from dataclasses import dataclass, field
from typing import List

@dataclass
class ExperimentConfig:
    # 模型池
    models: List[str] = field(default_factory=lambda: ["4B", "7B", "14B"])
    
    # Benchmark
    benchmarks: List[str] = field(default_factory=lambda: ["swe_bench", "web_arena"])
    
    # 实验重复次数 (不同随机种子)
    n_runs: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    
    # 任务数
    n_tasks_swe: int = 50
    n_tasks_web: int = 50
    
    # Skill 演化
    evolution_rounds: int = 10
    rollout_per_task: int = 5  # K=5 (Skill-MAS 推荐值)
    
    # SDR 路由器参数
    sdr_capability_weight: float = 0.6
    sdr_cost_eff_weight: float = 0.4
    sdr_softmax_temp: float = 0.15
    
    # 反崩溃阈值
    collapse_threshold: float = 0.80  # 单模型占用率超过此值判定为崩溃
    
    # 双反馈
    enable_pre_execution: bool = True
    enable_post_execution: bool = True
    
    # 输出目录
    output_dir: str = "output"
    
    # 统计检验
    alpha: float = 0.05
    bootstrap_samples: int = 10000

# 全局配置实例
CONFIG = ExperimentConfig()

# 实验名称到输出路径的映射
EXPERIMENT_PATHS = {
    "exp1_baseline": "output/exp1_baseline",
    "exp2_anticollapse": "output/exp2_anticollapse",
    "exp3_dualfeedback": "output/exp3_dualfeedback",
    "exp4_selective": "output/exp4_selective",
    "exp5_transfer": "output/exp5_transfer",
    "exp6_attribution": "output/exp6_attribution",
    "exp7_skillmas": "output/exp7_skillmas",
    "exp8_pareto": "output/exp8_pareto",
}
```

### Step 2: 验证 SDR Pipeline 可运行

```bash
cd sdr_eval_pipeline
python3 run_pipeline.py --n-tasks 10 --verbose
```

确认输出包含 3 个路由器的对比结果，且 SDR 的 Routing Collapse Rate = 0.0。

### Step 3: 验证 Skill-MAS 指标模块可运行

```bash
cd skill_mas_metrics
python3 skill_mas_metrics.py
```

确认输出包含 7 类指标的自测结果。

### Step 4: 创建统一入口

创建 `code/experiment_runner.py`，提供统一的实验运行接口：

```python
"""
SDR x Skill-MAS 统一实验运行器
支持按实验编号运行单个或全部实验
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

from config import CONFIG, EXPERIMENT_PATHS

def run_experiment(exp_id: str, verbose: bool = True):
    """运行指定编号的实验"""
    pass  # 由后续 prompt 实现

def main():
    parser = argparse.ArgumentParser(description="SDR x Skill-MAS Experiment Runner")
    parser.add_argument("--exp", type=str, default="all",
                       help="实验编号: exp1, exp2, ..., exp8, all")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    if args.exp == "all":
        for exp_id in sorted(EXPERIMENT_PATHS.keys()):
            run_experiment(exp_id, args.verbose)
    else:
        run_experiment(args.exp, args.verbose)

if __name__ == "__main__":
    main()
```

## Output

1. `code/config.py` — 实验配置文件
2. `code/experiment_runner.py` — 统一实验入口 (骨架)
3. 验证日志 (两个模块均能成功运行)

## Verification

- [ ] `python3 sdr_eval_pipeline/run_pipeline.py --n-tasks 5` 成功运行
- [ ] `python3 skill_mas_metrics/skill_mas_metrics.py` 成功运行
- [ ] `code/config.py` 可被 import
- [ ] 输出目录结构创建完成
