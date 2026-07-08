# SDR Evaluation Pipeline

**Skill-Driven Dynamic Routing (SDR) Evaluation Pipeline** for the AAAI paper.

Evaluates three routing strategies — **Rubar**, **RL-PER**, and **SDR** — across **6 metric categories (A–F)** covering routing accuracy, skill transfer, utilization stability, skill evolution, dual feedback, and failure attribution.

---

## Quick Start

```bash
cd sdr_eval_pipeline
pip install -r requirements.txt

# Run all three routers on SWE-bench mock data (20 tasks)
python run_pipeline.py

# Only SDR router, verbose output
python run_pipeline.py --router sdr --verbose

# SWE-bench + WebArena, 50 tasks each
python run_pipeline.py --benchmark both --n-tasks 50

# Custom output directory
python run_pipeline.py --output results/run1
```

---

## Architecture

```
sdr_eval_pipeline/
├── run_pipeline.py          # Main entry point & CLI
├── requirements.txt          # numpy
├── core/                     # Core framework
│   ├── types.py              # Data structures (Skill, StepContext, RoutingDecision, ...)
│   ├── skill_registry.py     # Skill storage, retrieval (TF-IDF), reranking, evolution
│   ├── model_pool.py         # 4B/7B/14B model pool with Rubar 7-dim capability matrix
│   └── router.py             # 3 router implementations (Rubar, RL-PER, SDR)
├── metrics/                  # 6 metric categories (A-F)
│   ├── routing_accuracy.py   # A: Skill Hit@1, MRR@10, Recall@K, FC@K, Conditioned Routing Acc
│   ├── transfer.py           # B: Cross-task/model/framework transfer, Adaptation speed, Exploration
│   ├── utilization.py        # C: Utilization balance, Routing entropy, Collapse rate, Pareto
│   ├── skill_evolution.py    # D: Refinement rate, Stability, Coverage, Quality gate, Velocity
│   ├── dual_feedback.py      # E: Pre/post execution match, Feedback gap, Plan F1, Exec F1
│   └── failure_attribution.py # F: Attribution rate, Discovery failure, Mismatch, Harness gap
├── data/                     # Mock data generators
│   └── mock_data.py          # SWE-bench & WebArena trajectory generators
└── output/                   # Results (auto-generated)
    └── results.json          # All metrics as JSON
```

### 5-Layer SDR Framework

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Skill Abstraction                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ SkillRegistry (TF-IDF retrieve + rerank + evolution)   │ │
│  │ Beta-Bernoulli capability profiles per (skill, model)  │ │
│  └───────────────────────┬─────────────────────────────────┘ │
│                          │                                    │
│  Layer 2: Rubar Side      │  Layer 3: RL-PER Side              │
│  ┌──────────────────┐     │     ┌──────────────────────┐       │
│  │ Rubric Condition  │     │     │ Skill-Conditioned    │       │
│  │ Matching          │────┼────▶│ RL Routing            │       │
│  │ (7-dim matrix)    │     │     │ (Beta-Bernoulli +     │       │
│  │                   │     │     │  cost-effectiveness)  │       │
│  └──────────────────┘     │     └───────────┬──────────┘       │
│                           │                 │                  │
│  Layer 4: Dual Feedback   │                 │                  │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Pre-execution (r_pre) → Execute → Post-execution (r_post)│ │
│  │ Feedback Gap = |r_pre - r_post|                          │ │
│  └──────────────────────────────┬───────────────────────────┘ │
│                                 │                              │
│  Layer 5: Meta-RL Transfer      │                              │
│  ┌──────────────────────────────┐                              │
│  │ Cross-task skill memory      │                              │
│  │ Reflective adaptation        │                              │
│  └──────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Routers

### 1. Rubar (Rubric-Based)

Deterministic routing based on a 7-dimension condition matching matrix.

- Evaluates conditions: `high_complexity`, `long_context`, `prev_failed`, `low_budget`, `has_code`
- Priority matching: most specific condition wins
- Latency: <2ms (simple rule matching)
- No skill abstraction (skills = `[]`)

### 2. RL-PER (RL Router)

RL-pretrained external router (simulated 4B Qwen3 router).

- Learned Q-values per (context, model) pair
- Softmax policy with temperature
- **Prone to routing collapse**: policy bias accumulates toward high-capability model
- Exploration rate decays over time
- No skill abstraction (skills = `[]`)

### 3. SDR (Skill-Driven)

The proposed framework combining skill identification + rubric + RL + dual feedback + Meta-RL.

- **Layer 1**: Two-stage skill retrieval (TF-IDF retrieve → context-aware rerank)
- **Layer 2**: Rubric condition matching (Rubar-style specificity)
- **Layer 3**: Skill-conditioned RL routing with cost-effectiveness scoring
  - Anti-collapse mechanism: blends raw capability (0.6) + cost-efficiency (0.4)
  - Context-dependent adjustments (complexity, budget, failure history)
  - Softmax with lower temperature (0.15) for sharper but diverse distribution
- **Layer 4**: Dual feedback (pre/post execution scores)
- **Layer 5**: Meta-RL update (Beta-Bernoulli posterior updates + cross-task memory)
- Latency: ~7ms (retrieval + rubric + RL)

---

## Metric Categories (A-F)

| Category | Name | Inspiration | # Metrics |
|----------|------|-------------|-----------|
| **A** | Routing Accuracy | SkillRouter | 5 |
| **B** | Transfer & Adaptation | SkillOpt + LaMer | 5 |
| **C** | Utilization & Stability | SkillOrchestra | 5 |
| **D** | Skill Evolution | SkillOpt + SkillOrchestra | 5 |
| **E** | Dual Feedback | ToolTree | 5 |
| **F** | Failure Attribution | PawBench | 4 |

### A: Skill-Level Routing Accuracy

| Metric | Description | Baseline |
|--------|-------------|----------|
| Skill Hit@1 | Top-1 predicted skill matches ground truth | 0.70 (Rubar coarse) |
| Skill MRR@10 | Reciprocal rank of first correct skill in top-10 | — |
| Skill Recall@10 | Fraction of ground-truth skills in top-10 | — |
| Skill FC@10 | Top-10 contains ALL ground-truth skills | — |
| Skill-Conditioned Routing Accuracy | Given correct skill, model selection is correct | 0.80 (Rubar matrix) |

### B: Skill Transfer & Adaptation

| Metric | Description | Baseline |
|--------|-------------|----------|
| Cross-task Skill Transfer | Pass@1 gain from skills learned on source task | — (SkillOpt: +15.2pp) |
| Cross-model Skill Transfer | Skill retention when model pool changes | — (SkillOpt: +15.2pp) |
| Cross-framework Skill Transfer | Skill transfer across agent frameworks | — (SkillOpt: +31.8pp) |
| Skill Adaptation Speed | Steps to reach stable skill hit rate | — (LaMer) |
| Skill Exploration Quality | Active exploration score on unseen tasks | — (LaMer: +11-19%) |

### C: Utilization & Stability

| Metric | Description | Baseline |
|--------|-------------|----------|
| Skill-level Utilization Balance | Evenness of model usage (1.0=perfect) | 0.02 (RL-PER collapse) |
| Routing Entropy | Shannon entropy of model selection (bits) | 0.10 (RL-PER collapse) |
| Routing Collapse Rate | Fraction where single model >95% | 0.98 (SkillOrchestra: RL 98%) |
| Skill-level Cost-Effectiveness | Pass rate per 1K tokens per skill | — |
| Pareto Frontier Coverage | Fraction of Pareto-optimal points | — |

### D: Skill Evolution

| Metric | Description | Baseline |
|--------|-------------|----------|
| Skill Refinement Rate | Splits + merges per 100 steps | — |
| Skill Stability | 1 - avg variance of capability profiles | — |
| Skill Coverage | Fraction of skill types with registered skills | 0.70 (Rubar 7/10) |
| Skill Quality Gate Pass Rate | Fraction passing Selection Gate | — (SkillOpt) |
| Skill Velocity | Average steps for new skill to converge | — (SkillOpt: 2-4 epoch) |

### E: Dual Feedback

| Metric | Description | Baseline |
|--------|-------------|----------|
| Pre-execution Skill Match | Correlation of pre-predictions with outcomes | — (Rubar: none) |
| Post-execution Skill Contribution | Average skill-level contribution after execution | — |
| Feedback Gap | |pre - post| execution score difference | 1.0 (no pre-execution) |
| Skill-level Plan F1 | F1 of planned skill sequence vs required | — (ToolTree) |
| Skill-level Exec F1 | F1 of executed outcomes vs expected | — (ToolTree) |

### F: Failure Attribution

| Metric | Description | Baseline |
|--------|-------------|----------|
| Skill-level Failure Attribution | Fraction of failures attributable to skills | 0.0 (Rubar/RL-PER: none) |
| Skill Discovery Failure Rate | Failures from weak skill discovery | — (PawBench: Skill_Use=47.2) |
| Skill-Model Mismatch Rate | Failures from skill-model capability mismatch | — |
| Harness-Skill Interaction Score | Model x harness performance variation | 11.5 (PawBench max gap) |

---

## Configuration

All parameters are in `core/types.py` → `EvaluationConfig`:

```python
config = EvaluationConfig(
    models=["4B", "7B", "14B"],       # Model pool
    model_costs={"4B": 1.0, "7B": 2.5, "14B": 6.0},
    model_latencies={"4B": 15.0, "7B": 25.0, "14B": 45.0},
    top_k_skills=10,                   # Skill retrieval depth
    collapse_threshold=0.95,           # Collapse detection threshold
    feedback_gap_threshold=0.15,       # Dual feedback gap threshold
    pareto_points=50,                  # Pareto frontier resolution
    split_variance_threshold=0.15,     # Skill split trigger
    merge_indistinguishable_threshold=0.05,  # Skill merge trigger
    output_dir="output",
    verbose=False,
)
```

---

## Extending the Pipeline

### Adding a New Router

```python
from core.router import BaseRouter

class MyRouter(BaseRouter):
    def __init__(self, config, model_pool):
        super().__init__(config, model_pool)
        self.router_type = "my_router"
    
    def route(self, step_context):
        # Your routing logic here
        return RoutingDecision(
            step_id=step_context.step_id,
            predicted_skills=[...],
            selected_model="7B",
            router_type=self.router_type,
        )
```

### Adding a New Metric

```python
from core.types import MetricResult

class MyMetrics:
    CATEGORY = "G"
    
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, trajectories):
        return [
            MetricResult(
                name="My Metric",
                value=0.85,
                category=self.CATEGORY,
                description="Description",
                baseline_value=0.50,
            )
        ]
```

### Connecting Real Model Backends

Replace the simulated execution in `run_pipeline.py → execute_trajectory()` with actual model calls:

```python
# Instead of simulated success probability:
response = call_model(decision.selected_model, ctx.query)
success = evaluate_response(response, ctx.gt_skills)
```

### Connecting Real Benchmarks

Replace `data/mock_data.py → generate_mock_trajectories()` with real benchmark data:

```python
from datasets import load_dataset
swe_bench = load_dataset("princeton-nlp/SWE-bench_Verified")

def generate_real_trajectories():
    for instance in swe_bench["test"]:
        # Build StepContext from real instance
        ...
```

---

## Output Format

Results are saved to `output/results.json`:

```json
{
  "SDR (Skill-Driven)": {
    "A": [
      {
        "name": "Skill Hit@1",
        "value": 0.625,
        "category": "A",
        "description": "Top-1 predicted skill matches ground truth",
        "baseline_value": 0.7,
        "detail": {"source": "SkillRouter", "n_samples": 80}
      },
      ...
    ],
    "B": [...],
    ...
  }
}
```

---

## Key Findings (Mock Data)

| Metric | Rubar | RL-PER | SDR | Improvement |
|--------|-------|--------|-----|-------------|
| Skill Hit@1 | 0.000 | 0.000 | **0.625** | +62.5pp (new signal) |
| Routing Entropy | 0.993 | 0.459 | **1.385** | +39.2% vs RL-PER |
| Routing Collapse | 0.000 | 0.000 | **0.000** | All fixed |
| Cost-Effectiveness | 0.345 | 0.267 | **0.574** | +66.4% vs Rubar |
| Skill Plan F1 | 0.000 | 0.000 | **0.512** | New signal |
| Exploration Quality | 9.4 | 4.3 | **13.1** | +39.4% vs Rubar |
| Total Tokens | 187K | 231K | **131K** | -30% vs Rubar |

---

## References

| System | Key Contribution | Metric Inspiration |
|--------|-----------------|-------------------|
| SkillRouter | Two-stage retrieve-and-rerank at scale (80K skills) | Hit@1, MRR@10, Recall@K, FC@K |
| SkillOpt | Text-space skill optimization + Selection Gate | Cross-model/framework transfer, Quality gate |
| SkillOrchestra | Beta-Bernoulli capability modeling + Pareto | Utilization balance, Collapse rate, Stability |
| ToolTree | MCTS tool selection + dual feedback | Pre/post exec scores, Plan/Exec F1 |
| LaMer | Meta-RL for LLM agent exploration | Exploration quality, Adaptation speed |
| PawBench | Model x Harness decomposition | Failure attribution 6-dimensional |
| Arbor | Markdown playbooks + on-demand skill loading | Skill registry design |
| Ares | Per-step dynamic reasoning effort | Router architecture |
