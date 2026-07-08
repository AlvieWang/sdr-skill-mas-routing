# SDR + Skill-MAS Routing Evaluation

> **AAAI Paper**: Skill-Driven Dynamic Routing (SDR) for Multi-Model Agent Systems

This repository contains two complementary evaluation modules for multi-model routing research:

1. **`sdr_eval_pipeline/`** — SDR (Skill-Driven Dynamic Routing) evaluation pipeline with 3 router implementations and 6 categories of metrics (29 metrics total)
2. **`skill_mas_metrics/`** — Skill-MAS evaluation metrics extraction module, ported from the official [Skill-MAS](https://github.com/linhh29/Skill-MAS) repository

## Repository Structure

```
.
├── sdr_eval_pipeline/          # SDR evaluation pipeline
│   ├── core/                    # Core framework (types, skill_registry, model_pool, router)
│   ├── metrics/                 # 6 metric categories (A-F, 29 metrics)
│   ├── data/                    # Mock data generator (SWE-bench + WebArena)
│   ├── run_pipeline.py          # Main entry point
│   ├── requirements.txt
│   └── README.md
├── skill_mas_metrics/           # Skill-MAS metrics extraction
│   ├── skill_mas_metrics.py     # 7 categories, 22 metrics (1640 lines)
│   └── README.md
└── docs/                        # Analysis documents
    ├── skill_driven_routing_analysis.md   # SDR framework analysis (8 systems, 8 gaps, 6 metric categories)
    ├── skill_mas_metrics_extraction.md     # Skill-MAS metrics extraction from paper
    ├── rubar_formalization.md             # Rubar router formalization
    └── rl_per_formalization.md            # RL-PER router formalization
```

## SDR Pipeline

### Three Routers

| Router | Description |
|--------|-------------|
| **RubarRouter** | Deterministic 7-dimensional rubric-based routing |
| **RLPerRouter** | RL-based policy router (with routing collapse tendency) |
| **SDRRouter** | 5-layer fusion: skill retrieval + rubric + skill-conditioned RL + dual feedback + Meta-RL |

### 6 Metric Categories (29 Metrics)

| Category | Metrics | Source |
|----------|---------|--------|
| **A. Routing Accuracy** | Skill Hit@1, MRR@10, Recall@K, FC@10 | SkillRouter |
| **B. Transfer** | Cross-task/model Transfer Score, Fallback Rate | SkillOpt |
| **C. Utilization** | Entropy, Collapse Rate, Cost-Effectiveness, Pareto Coverage | SkillOrchestra |
| **D. Skill Evolution** | Refinement Rate, Coverage, Split/Merge, Convergence | SkillOrchestra + SkillOpt |
| **E. Dual Feedback** | Pre/Post Exec Match, Plan/Exec F1, Feedback Gap | ToolTree |
| **F. Failure Attribution** | Attribution Rate, Discovery Failure, 6-dim breakdown | PawBench |

### Quick Start

```bash
cd sdr_eval_pipeline
python3 run_pipeline.py --verbose          # Run all 3 routers
python3 run_pipeline.py --router sdr       # Run SDR only
python3 run_pipeline.py --benchmark both   # SWE-bench + WebArena
```

### Sample Results (Mock, 10 tasks)

| Metric | Rubar | RL-PER | SDR |
|--------|-------|--------|-----|
| Skill Hit@1 | 0.000 | 0.000 | **0.625** |
| Routing Entropy | 0.993 | 0.459 | **1.385** |
| Routing Collapse | 0.000 | 0.000 | **0.000** |
| Cost-Effectiveness | 0.345 | 0.267 | **0.574** |
| Total Tokens | 187K | 231K | **131K** |

## Skill-MAS Metrics

### 7 Metric Categories (22 Metrics)

| Category | Key Metrics |
|----------|-------------|
| **A. Main Performance** | Avg.Perf, Avg.Cost, Per-Benchmark Score |
| **B. Distributional** | Per-task Uncertainty (std), Difficulty (-mean), Priority Score |
| **C. Selective Reflection** | Elbow Index (2nd-order diff), Selected Task Set |
| **D. Transferability** | Cross-LLM/Task/Full Transfer Delta |
| **E. Cost** | Inference Cost, Evolution Cost (separated) |
| **F. Evolution Tracking** | Best Round, Module-level Modification, Convergence |
| **G. Benchmark Scoring** | HLE-Math, BrowseComp, DRB, VitaBench |

### Quick Start

```bash
cd skill_mas_metrics
python3 skill_mas_metrics.py    # Self-test with mock data
```

### Source Code Mapping

| Module | Source File | Key Function |
|--------|------------|--------------|
| MainPerformanceMetrics | `evolution/assemble_select.py` | `compute_round_score()` |
| DistributionalMetrics | `evolution/elbow_selection.py` | `_priority_vectors()` |
| SelectiveReflectionMetrics | `evolution/elbow_selection.py` | `adaptive_elbow_count()` |
| TransferabilityMetrics | `evolution/contrastive_reflect.py` | `DomainPatch.source_gap` |
| CostMetrics | `utils/llm_cost.py` | `vita_rollout_cost_report()` |
| EvolutionTrackingMetrics | `evolution/bank_optimizer.py` | `_write_knee_artifacts()` |
| HLEMATHScorer | `dataset/hlemath/score.py` | `calculate_score()` |
| BrowseCompScorer | `dataset/BrowseComp-Plus/score.py` | `calculate_score()` |
| VitaBenchMetrics | `dataset/vitabench/.../agent_metrics.py` | `pass_hat_k()`, `pass_at_k()` |
| DRBScorer | `dataset/deep_research_bench/.../score_calculator.py` | `calculate_weighted_scores()` |

## Model Pool

| Model | Role | Cost Ratio |
|-------|------|-----------|
| 4B | Lightweight executor | 1x |
| 7B | Balanced executor | 3x |
| 14B | Heavyweight executor | 6x |

## License

MIT

## References

- [Skill-MAS](https://github.com/linhh29/Skill-MAS) — Lin et al., arXiv:2606.18837
- [SkillOpt](https://microsoft.github.io/SkillOpt/) — Microsoft Research
- [SkillRouter](https://arxiv.org/abs/2603.22455) — Skill routing at scale
- [SkillOrchestra](https://arxiv.org/abs/2602.19672) — Skill-level capability modeling
- [ToolTree](https://arxiv.org/abs/2603.12740) — Dual-feedback tool planning
- [LaMer](https://arxiv.org/abs/2512.16848) — Meta-RL for LLM agents
- [PawBench](https://github.com/agentscope-ai/PawBench) — Model x Harness evaluation
