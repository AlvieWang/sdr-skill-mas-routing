# SDR x Skill-MAS 实验设计文档

> **项目**: Skill-Driven Dynamic Routing (SDR) for Tool-Agent Scenarios
> **目标会议**: AAAI 2027
> **模型池**: Qwen3-4B / Qwen2.5-7B / Qwen2.5-14B
> **Benchmark**: SWE-bench (主) / WebArena (副) / MLAgentBench (验证)
> **文档版本**: v1.0

---

## 1. 研究问题与假设

### 1.1 核心研究问题

**RQ1**: 在 tool-agent 场景下，基于 skill 的动态路由是否能比确定性 rubric 路由 (Rubar) 和 RL 路由 (RL-PER) 取得更高的路由准确率和成本效益？

**RQ2**: Skill 级别的双反馈（预执行 + 后执行）是否能改善路由决策质量？

**RQ3**: 基于 skill 的反崩溃机制是否能避免 RL 路由器常见的路由崩溃问题？

**RQ4**: 通过 Meta-RL 积累的 skill 画像能否跨任务迁移？

**RQ5**: 引入 Skill-MAS 的分布统计指标（uncertainty, difficulty, priority）是否能进一步提升 skill 演化效率？

### 1.2 假设

| 假设 | 内容 | 对应实验 |
|------|------|---------|
| H1 | SDR 的 Skill Hit@1 显著高于 Rubar 和 RL-PER (p<0.05) | Exp 1 |
| H2 | SDR 的路由崩溃率 < 5%，而 RL-PER > 50% | Exp 1, 2 |
| H3 | 移除反崩溃机制后，SDR 的路由熵下降 > 50% | Exp 2 |
| H4 | 双反馈机制使 Plan F1 提升 > 15pp | Exp 3 |
| H5 | 优先级驱动的 skill 演化比全量反思节省 > 50% 计算成本，且性能不降 | Exp 4 |
| H6 | SWE-bench 上学到的 skill 迁移到 WebArena 后仍获得 > 10pp 增益 | Exp 5 |
| H7 | SDR 的失败归因准确率 > 80% | Exp 6 |
| H8 | 引入 Skill-MAS 的 uncertainty 指标后，skill 演化收敛速度提升 > 30% | Exp 7 |
| H9 | SDR 在 cost-performance Pareto 前沿上 dominate 两个基线 | Exp 8 |

---

## 2. 实验总览

| Exp | 名称 | 自变量 | 因变量 | 假设 | 预计耗时 |
|-----|------|--------|--------|------|---------|
| 1 | 基线对比 | Router: Rubar / RL-PER / SDR | A1-A5, C1-C5, 成本 | H1, H2 | 30 min |
| 2 | 反崩溃消融 | SDR 变体: full / no-cost-eff / no-entropy-reg | C3, C4, C5 | H3 | 20 min |
| 3 | 双反馈消融 | SDR 变体: full / pre-only / post-only / none | E1-E4, Plan F1 | H4 | 20 min |
| 4 | 选择性反思消融 | 演化策略: priority / full / random / none | D1-D5, 成本 | H5 | 30 min |
| 5 | 跨任务迁移 | 源域→目标域: SWE→Web / Web→SWE / SWE→ML | B1-B3 | H6 | 40 min |
| 6 | 失败归因分析 | Router: Rubar / RL-PER / SDR | F1-F5 | H7 | 20 min |
| 7 | Skill-MAS 指标融合 | 指标集: base / +uncertainty / +priority / +all | D1-D5 收敛速度 | H8 | 30 min |
| 8 | Pareto 前沿分析 | Router: all | Pareto coverage, cost-perf | H9 | 15 min |

---

## 3. 实验详细设计

### Exp 1: 基线对比实验

**目的**: 对比三个路由器在所有指标上的表现

**设置**:
- 数据集: SWE-bench (50 tasks) + WebArena (50 tasks)
- 路由器: RubarRouter, RLPerRouter, SDRRouter
- 重复次数: 5 runs (不同随机种子)
- 评估指标: 全部 6 类 (A-F, 29 个 metric)

**预期结果**:

| 指标 | Rubar | RL-PER | SDR | 显著性 |
|------|-------|--------|-----|--------|
| Skill Hit@1 | ~0% | ~0% | 60-70% | p<0.01 |
| Routing Entropy | ~1.0 | ~0.5 | ~1.4 | p<0.01 |
| Routing Collapse | 0% | 50-90% | <5% | p<0.01 |
| Cost-Effectiveness | ~0.35 | ~0.27 | ~0.57 | p<0.01 |
| Total Tokens | ~187K | ~231K | ~131K | p<0.05 |
| Plan F1 | ~0% | ~0% | ~0.51 | p<0.01 |

### Exp 2: 反崩溃机制消融

**目的**: 验证 cost-effectiveness scoring 和 entropy regularization 各自的贡献

**SDR 变体**:
1. `full`: 完整 SDR (0.6*cap + 0.4*cost_eff + softmax temp=0.15)
2. `no-cost-eff`: 移除 cost-effectiveness 项 (纯 capability + softmax)
3. `no-entropy-reg`: 使用 greedy 选择 (argmax) 替代 softmax
4. `rl-per-style`: 使用 RL-PER 的 cost penalty 机制 (success - 0.05*cost)

**关键指标**: C3 (Collapse Rate), C4 (Entropy), C5 (Pareto)

### Exp 3: 双反馈消融

**目的**: 分离预执行反馈和后执行反馈的贡献

**SDR 变体**:
1. `full`: 预执行 + 后执行
2. `pre-only`: 仅预执行（移除后执行评估）
3. `post-only`: 仅后执行（移除预执行评估）
4. `none`: 无反馈（静态路由）

**关键指标**: E1 (Pre-Post Match), E2 (Feedback Gap), E3 (Plan F1), E4 (Exec F1)

### Exp 4: 选择性反思消融

**目的**: 验证 Skill-MAS 的优先级排序 + 肘部截断是否优于全量/随机反思

**演化策略**:
1. `priority`: p_i = 0.5*(norm_uncertainty + norm_difficulty), 肘部截断
2. `full`: 对所有 task/skill 进行反思
3. `random`: 随机选择 50% task/skill 进行反思
4. `none`: 不进行 skill 演化

**关键指标**: D1 (Refinement Rate), D2 (Coverage), D3 (Convergence), 成本

### Exp 5: 跨任务迁移实验

**目的**: 验证 skill 画像的跨域迁移能力

**迁移路径**:
1. SWE-bench → WebArena (代码→网页)
2. WebArena → SWE-bench (网页→代码)
3. SWE-bench → MLAgentBench (代码→ML)

**度量**: 
- Transfer Score = Target域性能 - 从零开始性能
- Cross-domain Skill Overlap = 共享 skill 占比

### Exp 6: 失败归因分析

**目的**: 验证 SDR 的 6 维失败归因能力

**设置**: 对每个失败 step，SDR 输出失败原因，与 ground truth 对比

**6 维归因**: MODEL_REASONING / TOOL_MISSING / SKILL_DISCOVERY_WEAK / WORKSPACE_PERCEPTION / NETWORK_FRAGILE / COMPLETION_CHECK_LOOSE

### Exp 7: Skill-MAS 指标融合

**目的**: 验证引入 Skill-MAS 的分布统计指标后的改善

**变体**:
1. `base`: SDR 原始指标集
2. `+uncertainty`: 加入 step-level uncertainty (K=5 采样)
3. `+priority`: 加入 priority score 排序
4. `+all`: 加入全部 Skill-MAS 分布统计

**关键指标**: skill 演化收敛轮次, 最终 Avg.Perf

### Exp 8: Pareto 前沿分析

**目的**: 在 cost-performance 二维空间中验证 SDR 的 Pareto dominance

**方法**: 收集所有路由器在所有 task 上的 (cost, performance) 点，计算 Pareto 前沿

---

## 4. 指标体系速查

### 4.1 SDR Pipeline 指标 (6 类 29 个)

| 类别 | ID | 指标名 | 公式/定义 | 来源 |
|------|-----|--------|----------|------|
| A | A1 | Skill Hit@1 | top-1 skill 检索准确率 | SkillRouter |
| A | A2 | Skill MRR@10 | skill 检索 MRR | SkillRouter |
| A | A3 | Skill Recall@K | skill 检索召回率 | SkillRouter |
| A | A4 | Model Match Rate | 模型选择匹配率 | — |
| A | A5 | Skill Coverage | 覆盖的 skill 比例 | — |
| B | B1 | Cross-task Transfer | 跨任务迁移增益 | SkillOpt |
| B | B2 | Cross-model Transfer | 跨模型迁移增益 | SkillOpt |
| B | B3 | Skill Reuse Rate | skill 复用率 | LaMer |
| B | B4 | Exploration Quality | 探索质量 | LaMer |
| C | C1 | Utilization Balance | 模型利用率平衡 | SkillOrchestra |
| C | C2 | Routing Entropy | 路由分布熵 | SkillOrchestra |
| C | C3 | Routing Collapse Rate | 单模型占用率>80%的比例 | SkillOrchestra |
| C | C4 | Gini Coefficient | 模型分配基尼系数 | — |
| C | C5 | Pareto Frontier Coverage | Pareto 前沿覆盖 | SkillOrchestra |
| D | D1 | Skill Refinement Rate | skill 优化率 | SkillOpt |
| D | D2 | Skill Coverage Growth | skill 覆盖增长率 | — |
| D | D3 | Skill Convergence | 收敛轮次 | — |
| D | D4 | Skill Split Count | skill 分裂次数 | SkillOrchestra |
| D | D5 | Skill Merge Count | skill 合并次数 | SkillOrchestra |
| E | E1 | Pre-Post Match | 预/后执行评估一致率 | ToolTree |
| E | E2 | Feedback Gap | 预/后评估差距 | ToolTree |
| E | E3 | Plan F1 | 规划质量 F1 | ToolTree |
| E | E4 | Exec F1 | 执行质量 F1 | ToolTree |
| F | F1 | Attribution Rate | 失败归因准确率 | PawBench |
| F | F2 | Discovery Failure Rate | 未发现所需 skill 的比例 | PawBench |
| F | F3 | Per-Cause Accuracy | 各失败原因归因精度 | PawBench |
| F | F4 | False Attribution Rate | 误归因率 | — |
| F | F5 | Diagnostic Coverage | 诊断覆盖率 | — |

### 4.2 Skill-MAS 补充指标 (7 类 22 个)

| 类别 | 关键指标 | 何时使用 |
|------|---------|---------|
| 主性能 | Avg.Perf, Avg.Cost | Exp 1, 8 |
| 分布统计 | u_i (std), d_i (-mean) | Exp 4, 7 |
| 选择性反思 | p_i, Elbow Index j* | Exp 4, 7 |
| 迁移性 | Cross-LLM/Task Δ | Exp 5 |
| 消融 | Selective Reflection Gain | Exp 4 |
| 成本 | Evolution Cost | Exp 4 |
| 演化追踪 | Best Round, Module Modification | Exp 7 |

---

## 5. 实验执行顺序

```
Phase 1: 环境搭建 (Exp 0)
  └─ 安装依赖, 验证 pipeline 可运行

Phase 2: 基线实验 (Exp 1)
  └─ 运行 3 个路由器 × 2 个 benchmark × 5 runs
  └─ 收集全部 29 个 SDR 指标

Phase 3: 消融实验 (Exp 2, 3, 4 并行)
  ├─ Exp 2: 反崩溃消融 (4 变体)
  ├─ Exp 3: 双反馈消融 (4 变体)
  └─ Exp 4: 选择性反思消融 (4 策略)

Phase 4: 迁移与归因 (Exp 5, 6)
  ├─ Exp 5: 跨任务迁移 (3 路径)
  └─ Exp 6: 失败归因 (3 路由器 × 6 维)

Phase 5: 融合与前沿 (Exp 7, 8)
  ├─ Exp 7: Skill-MAS 指标融合 (4 变体)
  └─ Exp 8: Pareto 前沿分析

Phase 6: 分析与报告
  └─ 统计检验, 可视化, 自动报告生成
```

---

## 6. 统计检验方法

| 场景 | 检验方法 | 说明 |
|------|---------|------|
| 两组对比 | Welch's t-test | 不假设等方差 |
| 多组对比 | One-way ANOVA + Tukey HSD | 事后两两比较 |
| 非参数替代 | Mann-Whitney U / Kruskal-Wallis | 数据不满足正态假设时 |
| 效应量 | Cohen's d | 量化差异大小 |
| 置信区间 | Bootstrap (10000次) | 95% CI |

显著性水平: α = 0.05, 所有检验均使用 Bonferroni 校正。

---

## 7. 预期产出

### 7.1 数据文件
- `output/exp1_baseline/results.json` — 基线对比原始数据
- `output/exp2_anticollapse/results.json` — 反崩溃消融数据
- `output/exp3_dualfeedback/results.json` — 双反馈消融数据
- `output/exp4_selective/results.json` — 选择性反思消融数据
- `output/exp5_transfer/results.json` — 迁移实验数据
- `output/exp6_attribution/results.json` — 失败归因数据
- `output/exp7_skillmas/results.json` — Skill-MAS 融合数据
- `output/exp8_pareto/results.json` — Pareto 前沿数据

### 7.2 可视化
- 路由器对比雷达图 (6 类指标)
- 路由崩溃率柱状图 (消融对比)
- 双反馈一致性散点图
- 跨任务迁移热力图
- 失败归因混淆矩阵
- Pareto 前沿散点图
- Skill 演化收敛曲线

### 7.3 论文表格
- Table 1: 主结果 (3 router × 6 metric category)
- Table 2: 消融实验汇总
- Table 3: 迁移实验结果
- Table 4: 失败归因精度
- Figure 1: SDR 框架架构图
- Figure 2: Pareto 前沿可视化
- Figure 3: Skill 演化追踪

---

## 8. 代码结构

```
codex_experiment_guide/
├── EXPERIMENT_DESIGN.md          # 本文档
├── README.md                      # 快速入门
├── prompts/                       # Codex 提示词 (9 个)
│   ├── 01_env_setup.md           # 环境搭建
│   ├── 02_baseline_eval.md       # 基线对比实验
│   ├── 03_sdr_full_eval.md       # SDR 完整评估
│   ├── 04_skill_mas_integration.md # Skill-MAS 指标融合
│   ├── 05_transfer_eval.md       # 跨任务迁移实验
│   ├── 06_ablation_anticollapse.md # 反崩溃消融
│   ├── 07_ablation_dualfeedback.md # 双反馈消融
│   ├── 08_ablation_selective.md  # 选择性反思消融
│   └── 09_analysis_report.md     # 分析与报告
├── code/
│   ├── config.py                  # 实验配置
│   ├── experiment_runner.py       # 实验编排器
│   ├── ablation_runner.py         # 消融实验运行器
│   ├── transfer_runner.py         # 迁移实验运行器
│   └── analysis/
│       ├── statistical_tests.py   # 统计检验
│       ├── visualization.py       # 可视化
│       └── report_generator.py    # 自动报告
```

---

## 9. Codex 使用说明

### 9.1 使用方式

每个 `prompts/` 下的 `.md` 文件是一个独立的 Codex 提示词，可以直接复制粘贴给 Codex 执行。

推荐执行顺序:
1. 先执行 `01_env_setup.md` 搭建环境
2. 按 `02` → `03` → `04` → `05` → `06` → `07` → `08` → `09` 顺序执行
3. 每个步骤完成后，检查输出目录中的结果文件

### 9.2 提示词格式

每个提示词包含:
- **Context**: 实验背景和目标
- **Input**: 需要读取的文件
- **Task**: 具体任务描述
- **Output**: 预期产出
- **Verification**: 验证步骤
- **Code Template**: 代码模板 (可直接使用或修改)

### 9.3 依赖关系

```
01_env_setup
    └──> 02_baseline_eval
              ├──> 06_ablation_anticollapse
              ├──> 07_ablation_dualfeedback
              ├──> 08_ablation_selective
              ├──> 05_transfer_eval
              └──> 03_sdr_full_eval
                        └──> 04_skill_mas_integration
                                  └──> 09_analysis_report
```
