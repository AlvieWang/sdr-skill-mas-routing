# SDR x Skill-MAS Codex 实验指南

> **项目**: Skill-Driven Dynamic Routing (SDR) for Tool-Agent Scenarios
> **目标会议**: AAAI 2027
> **文档版本**: v1.0

本指南包含完整的实验设计文档和 10 个 Codex 提示词，用于指导 Codex (或任何编码 agent) 逐步完成 SDR x Skill-MAS 的全部实验。

---

## 快速开始

### 1. 环境准备

确保以下两个代码包已就位：

```
workspace/
├── sdr_eval_pipeline/        # SDR 评估 pipeline
│   ├── core/
│   ├── metrics/
│   ├── data/
│   ├── run_pipeline.py
│   └── README.md
├── skill_mas_metrics/        # Skill-MAS 指标提取
│   ├── skill_mas_metrics.py
│   └── README.md
└── codex_experiment_guide/   # 本指南
    ├── EXPERIMENT_DESIGN.md
    ├── prompts/              # 10 个 Codex 提示词
    └── code/                 # 代码模板
```

### 2. 执行顺序

按以下顺序将 `prompts/` 下的 `.md` 文件内容复制给 Codex：

| 步骤 | 提示词文件 | 实验内容 | 预计产出 |
|------|-----------|---------|---------|
| 1 | `01_env_setup.md` | 环境搭建与验证 | config.py, experiment_runner.py |
| 2 | `02_baseline_eval.md` | 基线对比 (Exp 1) | 30 runs × 29 metrics |
| 3 | `03_sdr_full_eval.md` | SDR 多轨迹采样 | 分布统计 + 肘部检测 |
| 4 | `04_skill_mas_integration.md` | Skill-MAS 指标融合 (Exp 7) | 4 策略对比 |
| 5 | `05_transfer_eval.md` | 跨任务迁移 (Exp 5) | 3 路径 × 迁移增益 |
| 6 | `06_ablation_anticollapse.md` | 反崩溃消融 (Exp 2) | 4 变体对比 |
| 7 | `07_ablation_dualfeedback.md` | 双反馈消融 (Exp 3) | 4 变体对比 |
| 8 | `08_ablation_selective.md` | 选择性反思消融 (Exp 4) | 4 策略对比 |
| 9 | `09_attribution_pareto.md` | 失败归因 + Pareto (Exp 6+8) | 归因精度 + 前沿分析 |
| 10 | `10_analysis_report.md` | 统计检验 + 可视化 + 报告 | 图表 + 报告 |

### 3. 依赖关系

```
01_env_setup
    └──> 02_baseline_eval
              ├──> 03_sdr_full_eval
              │         └──> 04_skill_mas_integration
              ├──> 06_ablation_anticollapse
              ├──> 07_ablation_dualfeedback
              ├──> 08_ablation_selective
              ├──> 05_transfer_eval
              └──> 09_attribution_pareto
                        └──> 10_analysis_report
```

步骤 2-9 可以并行执行（它们都依赖步骤 1 但相互独立），步骤 10 需要等所有实验完成。

---

## 实验总览

| Exp | 名称 | 自变量 | 核心假设 | 提示词 |
|-----|------|--------|---------|--------|
| 1 | 基线对比 | Router (3) | H1, H2 | 02 |
| 1+ | SDR 多轨迹 | K=5 rollout | — | 03 |
| 2 | 反崩溃消融 | SDR 变体 (4) | H3 | 06 |
| 3 | 双反馈消融 | 反馈模式 (4) | H4 | 07 |
| 4 | 选择性反思 | 策略 (4) | H5 | 08 |
| 5 | 跨任务迁移 | 路径 (3) | H6 | 05 |
| 6 | 失败归因 | Router (3) | H7 | 09 |
| 7 | Skill-MAS 融合 | 指标集 (4) | H8 | 04 |
| 8 | Pareto 前沿 | Router (3) | H9 | 09 |

---

## 指标体系

### SDR Pipeline (6 类 29 个)

| 类别 | 来源 | 代表指标 |
|------|------|---------|
| A: 路由准确率 | SkillRouter | Skill Hit@1, MRR@10 |
| B: 迁移适应 | SkillOpt, LaMer | Cross-task Transfer, Exploration Quality |
| C: 利用率稳定 | SkillOrchestra | Routing Entropy, Collapse Rate, Pareto |
| D: Skill 演化 | SkillOpt | Refinement Rate, Convergence |
| E: 双反馈 | ToolTree | Plan F1, Exec F1, Pre-Post Match |
| F: 失败归因 | PawBench | Attribution Rate, Discovery Failure |

### Skill-MAS 补充 (7 类 22 个)

| 类别 | 代表指标 | 使用场景 |
|------|---------|---------|
| 分布统计 | Uncertainty (std), Difficulty (-mean) | 驱动选择性反思 |
| 选择性反思 | Priority Score, Elbow Index | 优化演化预算 |
| 迁移性 | Cross-LLM/Task Δ | 迁移实验 |
| 消融 | Selective Reflection Gain | 消融对比 |
| 成本 | Evolution Cost | 成本分析 |
| 演化追踪 | Best Round, Module Modification | 演化质量 |

---

## 使用提示词的注意事项

1. **每个提示词是自包含的**：包含完整的上下文、任务描述、代码模板和验证标准
2. **代码模板可直接使用**：提示词中的 Python 代码块是完整的实现，可以直接复制到文件中
3. **预期结果仅供参考**：实际数值可能因随机种子和数据而异，但趋势应该一致
4. **验证步骤是必须的**：每个提示词末尾的 Verification 清单用于确认实验正确性
5. **可以迭代修改**：如果验证不通过，可以让 Codex 根据错误信息调整代码

---

## 输出文件结构

实验完成后，`output/` 目录将包含：

```
output/
├── exp1_baseline/
│   ├── results_raw.json          # 30 runs 原始数据
│   ├── results_aggregated.json   # mean ± std 聚合
│   └── comparison_table.tex      # LaTeX 表格
├── exp1_sdr_extended/
│   └── results.json              # 多轨迹采样 + 分布统计
├── exp2_anticollapse/
│   └── results.json              # 4 变体消融
├── exp3_dualfeedback/
│   └── results.json              # 4 变体消融
├── exp4_selective/
│   ├── results.json
│   └── ablation_table.tex
├── exp5_transfer/
│   ├── results.json
│   └── heatmap_data.json
├── exp6_attribution/
│   └── results.json
├── exp7_skillmas/
│   └── results.json
├── exp8_pareto/
│   └── results.json
├── visualizations/
│   ├── exp1_radar.png
│   ├── exp2_collapse.png
│   ├── exp5_transfer.png
│   ├── exp7_convergence.png
│   └── exp8_pareto.png
├── statistical_tests.json        # 统计检验结果
└── EXPERIMENT_REPORT.md          # 完整实验报告
```

---

## 论文表格映射

| 论文表格 | 数据来源 | 提示词 |
|---------|---------|--------|
| Table 1: 主结果 | exp1_baseline | 02 |
| Table 2: 反崩溃消融 | exp2_anticollapse | 06 |
| Table 3: 双反馈消融 | exp3_dualfeedback | 07 |
| Table 4: 选择性反思 | exp4_selective | 08 |
| Table 5: 迁移实验 | exp5_transfer | 05 |
| Table 6: 失败归因 | exp6_attribution | 09 |
| Table 7: Skill-MAS 融合 | exp7_skillmas | 04 |
| Figure 1: 雷达图 | exp1_baseline | 10 |
| Figure 2: 反崩溃柱状图 | exp2_anticollapse | 10 |
| Figure 3: 迁移热力图 | exp5_transfer | 10 |
| Figure 4: 收敛曲线 | exp7_skillmas | 10 |
| Figure 5: Pareto 前沿 | exp8_pareto | 10 |
