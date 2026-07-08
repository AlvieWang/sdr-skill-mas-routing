# Skill-MAS 评测指标提取文档

> **论文**: Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems
> **作者**: Hehai Lin, Qi Yang, Chengwei Qin (蚂蚁集团 × 香港科技大学广州)
> **来源**: arXiv:2606.18837 | Github | Project Page
> **提取日期**: 2026-07-06

---

## 目录

1. [指标总览](#1-指标总览)
2. [主性能指标（Table 1）](#2-主性能指标table-1)
3. [分布统计指标（Multi-Trajectory Rollout）](#3-分布统计指标multi-trajectory-rollout)
4. [选择性反思指标（Selective Reflection）](#4-选择性反思指标selective-reflection)
5. [迁移性指标（Table 2 / Figure 3）](#5-迁移性指标table-2--figure-3)
6. [消融实验指标（Table 3）](#6-消融实验指标table-3)
7. [成本指标（Table 6）](#7-成本指标table-6)
8. [Benchmark 专项指标](#8-benchmark-专项指标)
9. [Meta-Skill 演化追踪指标（Figure 4）](#9-meta-skill-演化追踪指标figure-4)
10. [超参数配置（Table 5）](#10-超参数配置table-5)
11. [与 SDR Pipeline 指标对照](#11-与-sdr-pipeline-指标对照)

---

## 1. 指标总览

Skill-MAS 共使用 **7 类、22 个评测指标**，涵盖性能、成本、分布统计、迁移性、消融、演化和基准专项维度。

| 类别 | 指标数 | 核心关注点 | 论文位置 |
|------|--------|-----------|---------|
| 主性能 | 3 | 整体准确率与成本 | Table 1 |
| 分布统计 | 4 | 每任务方差/难度/优先级 | §3.2, Figure 2 |
| 选择性反思 | 2 | 肘部截断/反思命中率 | §3.3.1, Table 3 |
| 迁移性 | 4 | 跨LLM/跨任务迁移增益 | Table 2, Figure 3 |
| 消融 | 3 | 选择性反思/多任务/采样数 | Table 3, Figure 3 |
| 成本 | 2 | 推理成本/进化成本 | Table 1, Table 6 |
| Benchmark专项 | 4 | 各基准特有评分维度 | §5.1 |

---

## 2. 主性能指标（Table 1）

### 2.1 指标定义

| 指标 | 符号 | 定义 | 方向 |
|------|------|------|------|
| **Avg.Perf** | — | 四个 Benchmark 的平均性能（归一化到 [0,100%]） | ↑ 越高越好 |
| **Avg.Cost** | — | 测试集上的平均推理成本（美元 $） | ↓ 越低越好 |
| **Per-Benchmark Score** | DRB/HLE/BCP/VITA | 各 Benchmark 的单独得分 | ↑ 越高越好 |

### 2.2 主要结果（4 个 Meta-agent × 4 个 Benchmark）

| Meta-agent | 指标 | EvoAgent | AOrchestra | AFlow | MAS2 | MAS-Orchestra | Skill-MAS-init | **Skill-MAS-opt** |
|------------|------|----------|------------|-------|------|---------------|----------------|-------------------|
| **Gemini-3.1-Flash** | Avg.Perf | 19.12 | 18.95 | 21.29 | 15.02 | 19.02 | 21.68 | **29.49** |
| | Avg.Cost | $8.20 | $2.31 | $0.44 | $0.91 | $2.48 | $2.63 | **$2.82** |
| **GPT-5.4-Nano** | Avg.Perf | 24.83 | 20.25 | 19.11 | 16.68 | 18.99 | 19.64 | **27.55** |
| | Avg.Cost | $6.26 | $2.45 | $0.69 | $1.27 | $1.84 | $4.24 | **$5.22** |
| **Qwen3.5-Plus** | Avg.Perf | 31.10 | 29.02 | 32.23 | 22.89 | 26.92 | 32.61 | **38.41** |
| | Avg.Cost | $7.91 | $1.91 | $1.56 | $4.04 | $1.35 | $1.83 | **$2.43** |
| **DeepSeek-V4-Flash** | Avg.Perf | 34.30 | 34.36 | 35.70 | 20.06 | 31.05 | 33.72 | **41.05** |
| | Avg.Cost | $3.33 | $2.69 | $1.45 | $0.63 | $1.98 | $1.14 | **$2.12** |

### 2.3 关键发现

- **进化有效**：同一 backbone 下，optimized 行的 Avg.Perf 几乎总是高于 init
- **optimized 几乎全线登顶**：4 个 Meta-agent 中 Skill-MAS-optimized 的 Avg.Perf 均高于所有基线
- **成本-性能位置合理**：落在"高分 + 中等成本"区间，测试阶段一次生成 MAS

---

## 3. 分布统计指标（Multi-Trajectory Rollout）

这是 Skill-MAS 最核心的创新指标体系，用于驱动选择性反思。

### 3.1 每任务采样统计

对每个验证集任务 $t_i$，在当前 Meta-Skill $\mathcal{S}^{(r)}$ 下独立采样 $K$ 条 rollout，记录：

| 指标 | 公式 | 含义 | 用途 |
|------|------|------|------|
| **Per-task Score** | $s_{i,k}$ | 第 $k$ 次 rollout 的归一化得分 | 基础数据 |
| **Per-task Mean** | $\bar{s}_i = \text{Mean}(s_{i,1}, \ldots, s_{i,K})$ | 平均得分 | 度量任务难度 |
| **Per-task Uncertainty** | $u_i = \text{Std}(s_{i,1}, \ldots, s_{i,K})$ | K 次得分标准差 | 度量 Skill 编排稳定性 |
| **Per-task Difficulty** | $d_i = -\bar{s}_i$ | 负均分（均分越低=越难） | 度量任务系统性难度 |

### 3.2 指标设计直觉

> 单次 0.8 分可能是运气；若 5 次得分是 [0.2, 0.8, 0.3, 0.7, 0.2]，说明 Skill 对「并行分支怎么合并」写得含糊——这才是该改的规则。

- **高 uncertainty + 低 mean** = Skill 对该任务的编排策略不稳定且效果差 → 优先反思
- **低 uncertainty + 高 mean** = Skill 已掌握该任务 → 无需优化
- **高 uncertainty + 高 mean** = 偶然成功但不稳定 → 次优先

### 3.3 轨迹记录格式

每条 rollout 记录为元组：
$$\tau_{i,k} = (id_{i,k}, s_{i,k}, \mathcal{S}^{(r)}, \phi_{i,k})$$

其中 $\phi_{i,k}$ 是架构快照和中间结果。

全部 rollout 聚合为 rollout 语料：
$$\mathcal{D}^{(r)} = \{(id_{i,k}, s_{i,k}, \mathcal{S}^{(r)}, \phi_{i,k})\}_{i,k=1}^{N,K}$$

---

## 4. 选择性反思指标（Selective Reflection）

### 4.1 优先级评分

| 指标 | 公式 | 说明 |
|------|------|------|
| **Normalized Uncertainty** | $\tilde{u}_i = \frac{u_i - \min_j u_j}{\max_j u_j - \min_j u_j}$ | 对 uncertainty 做 min-max 归一化 |
| **Normalized Difficulty** | $\tilde{d}_i = \frac{d_i - \min_j d_j}{\max_j d_j - \min_j d_j}$ | 对 difficulty 做 min-max 归一化 |
| **Priority Score** | $p_i = \frac{1}{2}(\tilde{u}_i + \tilde{d}_i)$ | 融合优先级，联合偏好"波动大 + 系统性难"的任务 |

### 4.2 肘部检测截断

| 指标 | 公式 | 说明 |
|------|------|------|
| **First-order Difference** | $\delta_j = p_{(j)} - p_{(j+1)}$ | 优先级序列的一阶差分 |
| **Elbow Index** | $j^* = \arg\max_{j \in \{1,\ldots,N-2\}} |\delta_j - \delta_{j+1}| + 1$ | 最大二阶差分位置，曲线拐点 |
| **Selected Task Set** | $\mathcal{T}_{sel} = \{t_{(1)}, \ldots, t_{(j^*)}\}$ | 拐点前的 top 任务集 |

### 4.3 层次化反思

| 阶段 | 指标 | 说明 |
|------|------|------|
| **Within-task Contrast** | High-score set $H_i$ vs. Low-score set $L_i$ | 以中位数切分 K 条轨迹为高/低分组 |
| **Cross-task Synthesis** | Evidence $\mathcal{E}$ | 跨任务综合，候选补丁排序成证据包 |
| **Patch Quality** | Generalizable principle check | 每条改动必须抽象成通用编排原则 |

---

## 5. 迁移性指标（Table 2 / Figure 3）

### 5.1 迁移实验设计

| 迁移类型 | Skill Source | Test Setting | 说明 |
|----------|-------------|-------------|------|
| **同 LLM 同任务** (No Transfer) | (LLM_A, Task_A) | (LLM_A, Task_A) | 对角线，增益最大 |
| **跨 LLM 同任务** (Cross-LLM) | (LLM_A, Task_A) | (LLM_B, Task_A) | 自然语言不绑 hidden state |
| **同 LLM 跨任务** (Cross-Task) | (LLM_A, Task_A) | (LLM_A, Task_B) | 学到的是任务无关策略 |
| **跨 LLM + 跨任务** (Full Transfer) | (LLM_A, Task_A) | (LLM_B, Task_B) | 最难，多数仍为正 |

### 5.2 迁移增益指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **Absolute Score** | Test Setting 上的原始得分 | 表格中的数值 |
| **Delta (Δ)** | $\Delta = \text{Score}_{\text{with transferred skill}} - \text{Score}_{\text{Skill-MAS-init}}$ | 相对于 init 的增益 |

### 5.3 迁移热力图（Figure 3 Left）

- 使用列向归一化（每个 Test Setting 的 Δ 从 0 到 1）
- 颜色越深 = 迁移增益越大

### 5.4 关键迁移数据

| 迁移场景 | Δ (pp) | 说明 |
|----------|--------|------|
| GPT → GPT (BCP) | +7.74 | 同 LLM 同任务 |
| DeepSeek → DeepSeek (BCP) | +7.14 | 同 LLM 同任务 |
| GPT → GPT (VITA) | +9.53 | 同 LLM 同任务 |
| DeepSeek → DeepSeek (VITA) | +9.53 | 同 LLM 同任务 |
| GPT → DeepSeek (BCP) | +2.97 | 跨 LLM 同任务 |
| DeepSeek → GPT (BCP) | +4.76 | 跨 LLM 同任务 |
| GPT (BCP → VITA) | +7.15 | 同 LLM 跨任务 |
| DeepSeek (BCP → VITA) | +5.95 | 同 LLM 跨任务 |
| GPT (BCP) → DeepSeek (VITA) | +2.38 | 跨 LLM + 跨任务 |
| DeepSeek (VITA) → GPT (BCP) | +2.98 | 跨 LLM + 跨任务 |

---

## 6. 消融实验指标（Table 3）

### 6.1 选择性反思消融

| 变体 | GPT-5.4-Nano BCP | GPT-5.4-Nano VITA | DeepSeek BCP | DeepSeek VITA | 说明 |
|------|-----------------|-------------------|-------------|--------------|------|
| **Ours (Selective)** | 27.38 | 15.48 | 22.62 | 63.10 | 默认配置 |
| **Full-Validation** | 22.02 | 13.10 | 19.64 | 59.52 | 全量反思（无优先级筛选） |
| **Half-Validation** | 21.43 | 9.52 | 17.26 | 58.33 | 随机抽 50% 样本 |

**消融指标**：
- **Selective Reflection Gain** = Ours - Full-Validation = +5.36 pp (BCP, GPT)
- **Full vs. Half Gap** = Full-Validation - Half-Validation = +0.59 pp (BCP, GPT)
- **Label-free Degradation** = Ours - Full-Validation（量化 label 依赖程度）

### 6.2 多任务学习消融

| 变体 | GPT-5.4-Nano BCP | GPT-5.4-Nano VITA | DeepSeek BCP | DeepSeek VITA |
|------|-----------------|-------------------|-------------|--------------|
| **Ours (Single-domain)** | 27.38 | 15.48 | 22.62 | 63.10 |
| **Multi-task Learning** | 20.83 | 16.67 | 22.02 | 64.29 |

**关键发现**：多任务学习在 VitaBench 略涨 (+0.79/+1.19)，但 BrowseComp-Plus 明显下降 (-6.55/-0.60)，说明跨域噪声需要专门机制。

### 6.3 采样数消融（Figure 3 Right）

| Rollout数 K | 性能趋势 | 边际收益 |
|------------|---------|---------|
| K=3 | 基线 | — |
| K=5 | 显著提升 | 3→5 边际收益大 |
| K=7 | 继续提升 | 5→7 边际收益递减 |

**结论**：K=5 是性价比折中点。

---

## 7. 成本指标（Table 6）

### 7.1 推理成本（测试集）

| 指标 | 定义 | 说明 |
|------|------|------|
| **Avg.Cost** | 测试集上的平均推理开销 (USD $) | Table 1 右列，仅含推理不含进化 |

### 7.2 进化成本（验证集）

| Meta-agent | Gemini-3.1-Flash | GPT-5.4-Nano | Qwen3.5-Plus | DeepSeek-V4-Flash |
|------------|----------------|--------------|-------------|-------------------|
| **Avg. Evolution Cost** | $9.35 | $31.36 | $59.06 | $24.54 |

### 7.3 成本分离原则

- **推理成本**：测试集上一次生成 MAS 的开销（瓶颈所在）
- **进化成本**：验证集上 10 轮 × N 任务 × K=5 轨迹的采样开销
- 两者分开统计，Table 1 只报推理成本

### 7.4 成本-性能权衡分析

| 范式 | Avg.Cost | Avg.Perf | 特点 |
|------|---------|---------|------|
| Training-time MAS | 最低 | 最低 | 经验在权重里，难迁移 |
| Inference-time MAS | 最高 | 中高 | per-query 反复搜索 |
| **Skill-MAS** | 中等 | 最高 | 一次生成 + 进化积累 |

---

## 8. Benchmark 专项指标

### 8.1 DeepResearchBench (DRB)

| 指标 | 维度 | 说明 |
|------|------|------|
| **Comprehensiveness** | 全面性 | 报告覆盖的研究范围 |
| **Insight** | 洞察性 | 研究分析的深度 |
| **Instruction-Following** | 指令遵循 | 是否满足用户要求 |
| **Readability** | 可读性 | 报告的结构和表达质量 |

- 100 个 PhD 级研究任务，跨 22 个领域
- 验证集 16 / 测试集 84

### 8.2 HLE-Math (HLE)

| 指标 | 说明 |
|------|------|
| **Accuracy** | 专家级数学题的准确率 |

- Humanity's Last Exam 的 MATH 子集
- 验证集 32 / 测试集 168（从 2500 题中采样 200 题）

### 8.3 BrowseComp-Plus (BCP)

| 指标 | 说明 |
|------|------|
| **Accuracy** | 多跳动态问答准确率 |

- 固定语料库 + 人工验证支持文档 + 对抗性负样本
- 验证集 32 / 测试集 168（从 200 题中采样）

### 8.4 VitaBench (VITA)

| 指标 | 说明 |
|------|------|
| **Rubric-based Success Rate** | 基于评分量表的成功率 |

- 66 个工具，100 个跨场景任务 + 300 个单场景任务
- 使用 rubric-based sliding window evaluator
- 验证集 16 / 测试集 84

### 8.5 统一归一化

> 所有指标统一归一化到 [0, 100%]

---

## 9. Meta-Skill 演化追踪指标（Figure 4）

### 9.1 逐轮演化追踪

| 轮次 | 模块 | 变化 | 来源 Benchmark |
|------|------|------|---------------|
| Round 1 | Task Decomposition | Evidence Weighting | BCP |
| Round 1 | Task Decomposition | Parallel Fan-out for multi-constraint task | BCP |
| Round 2 | Agent Engineering | Weighted-satisfaction protocol | BCP |
| Round 3 | Workflow Topology | Backtracking and dynamic replanning | BCP |
| Round 4 | Workflow Topology | Add a Link-verification task | BCP |
| Round 5 (best) | Workflow Topology | Merge-Node Re-execution Authority | BCP |

### 9.2 模块级修改可追踪性

| 指标 | 说明 |
|------|------|
| **Module-level Modification Count** | 每轮每个模块的修改次数 |
| **Modification Type** | 新增规则 / 修改规则 / 删除规则 |
| **Generalization Check** | 每条改动是否抽象为通用编排原则 |
| **Structural Validity Check** | 改完后的结构检查 |

### 9.3 三模块切分

| 模块 | 职责 | 示例修改 |
|------|------|---------|
| **Task Decomposition** (What) | 拆子任务、定义成功标准 | 约束分组、有界并行、跨实体桥接 |
| **Agent Engineering** (Who) | 实例化子Agent、分配角色工具 | 数据完整性约束、两级验证、重试协议 |
| **Workflow Topology** (How) | 选拓扑、定义数据流 | 菱形fan-in、回退重规划、全局/局部记忆 |

### 9.4 演化质量指标

| 指标 | 说明 |
|------|------|
| **Best Round Selection** | $S^* = \arg\max_r \text{val\_score}(\mathcal{S}^{(r)})$ | 在验证集上选最优轮次的 Skill |
| **Skill Convergence** | 连续轮次间修改量递减 | 趋于稳定的信号 |
| **Skill Complexity Growth** | Skill 文档长度/规则数增长 | 从通用框架→操作性规范 |

---

## 10. 超参数配置（Table 5）

| 超参数 | 符号 | 值 | 说明 |
|--------|------|-----|------|
| **Evolution Rounds** | R | 10 | Skill 进化总轮数 |
| **Rollout per Task** | K | 5 | 每任务每轮采样轨迹数 |
| **LLM Temperature** | — | 1.0 | 采样温度 |
| **Max Tokens** | — | 32768 | 最大输出 token |

### 10.1 数据集划分（Table 4）

| Benchmark | Validation | Test |
|-----------|-----------|------|
| DeepResearchBench | 16 | 84 |
| HLE-Math | 32 | 168 |
| BrowseComp-Plus | 32 | 168 |
| VitaBench | 16 | 84 |
| Multi-task Learning | 48 (每数据集取一半) | — |

### 10.2 Meta-agent 配置

| LLM | 配置 |
|-----|------|
| GPT-5.4-Nano | "low" reasoning effort |
| Gemini-3.1-Flash | "low" reasoning effort |
| Qwen3.5-Plus | 标准版（无额外 reasoning effort） |
| DeepSeek-V4-Flash | 标准版 |
| **LLM-Judge** | Gemini-3.1-Flash |

---

## 11. 与 SDR Pipeline 指标对照

### 11.1 指标维度映射

| 维度 | Skill-MAS 指标 | SDR Pipeline 对应指标 | 关系 |
|------|---------------|---------------------|------|
| **性能** | Avg.Perf (4基准均值) | Task Success Rate | 均为端到端性能 |
| **成本** | Avg.Cost (USD) | Total Tokens / Cost-Effectiveness | Skill-MAS 用美元, SDR 用 token |
| **分布统计** | u_i (std), d_i (-mean) | — (SDR 无此维度) | **Skill-MAS 独有** |
| **优先级** | p_i = 1/2(ũ+d̃) | — (SDR 无此维度) | **Skill-MAS 独有** |
| **选择性** | Elbow Index j*, T_sel | — (SDR 无此维度) | **Skill-MAS 独有** |
| **迁移性** | Cross-LLM/Task Δ | Cross-task/model Transfer Score | 概念对齐, 度量方式不同 |
| **利用率** | — (无) | Utilization Balance, Routing Entropy | **SDR 独有** |
| **崩溃** | — (无) | Routing Collapse Rate | **SDR 独有** |
| **演化** | Round-by-round changes (Fig.4) | Skill Refinement Rate, Coverage | 概念对齐, 粒度不同 |
| **反馈** | Within-task Contrast (H_i vs L_i) | Pre/Post Exec Match, Plan F1 | 不同反馈机制 |
| **归因** | — (无显式归因) | Attribution Rate, Discovery Failure | **SDR 独有** |
| **探索** | K rollout sampling | Skill Exploration Quality | 概念对齐 |
| **Pareto** | — (隐式, Figure 1d) | Pareto Frontier Coverage | SDR 显式度量 |
| **消融** | Selective vs. Full/Half | — (SDR 无消融实验) | **Skill-MAS 独有** |

### 11.2 Skill-MAS 独有指标（SDR 可借鉴）

| 指标 | 来源 | 公式/定义 | SDR 可借鉴方向 |
|------|------|----------|---------------|
| **Per-task Uncertainty** | §3.2 | $u_i = \text{Std}(s_{i,1}, \ldots, s_{i,K})$ | 增加 step-level 方差指标, 度量路由决策稳定性 |
| **Per-task Difficulty** | §3.2 | $d_i = -\text{Mean}(s_i)$ | 增加 task-level 难度评分, 辅助路由决策 |
| **Priority Score** | §3.3.1 | $p_i = \frac{1}{2}(\tilde{u}_i + \tilde{d}_i)$ | 用于 SDR 的 skill 演化优先级排序 |
| **Elbow Index** | §3.3.1 | $j^* = \arg\max_j \|\delta_j - \delta_{j+1}\| + 1$ | 自动确定需要优化的 skill 数量 |
| **Selective Reflection Gain** | Table 3 | Ours - Full-Validation | 度量选择性策略 vs 全量策略的增益 |
| **Within-task Contrast** | §3.3.2 | $H_i$ vs $L_i$ (中位数切分) | 用于 SDR 的双反馈对比分析 |
| **Cross-task Synthesis** | §3.3.2 | Evidence $\mathcal{E}$ 跨任务综合 | 用于 SDR 的 Meta-RL 跨 episode 综合 |
| **Skill Convergence** | Figure 4 | 连续轮次修改量递减 | 用于 SDR 的 skill 演化收敛检测 |
| **Module-level Modification** | Figure 4 | 每轮每模块修改次数 | 用于 SDR 的 skill 组件级追踪 |
| **Evolution Cost** | Table 6 | 验证集进化开销 | SDR 应增加训练/进化阶段成本统计 |

### 11.3 SDR 独有指标（Skill-MAS 无）

| 指标 | SDR 类别 | 定义 | Skill-MAS 缺失原因 |
|------|---------|------|-------------------|
| **Skill Hit@1** | A | Skill 检索 Top-1 准确率 | Skill-MAS 不做 skill 检索 |
| **MRR@10** | A | Skill 检索 MRR | 同上 |
| **Routing Entropy** | C | 模型选择分布熵 | Skill-MAS 不做模型路由 |
| **Routing Collapse Rate** | C | 单模型占用率阈值 | 同上 |
| **Pareto Frontier Coverage** | C | 成本-性能前沿覆盖 | Skill-MAS 隐式但未显式度量 |
| **Pre/Post Exec Match** | E | 预执行与后执行评估一致性 | Skill-MAS 只有后置反思 |
| **Plan F1 / Exec F1** | E | 规划质量与执行质量 | Skill-MAS 无分步评估 |
| **Attribution Rate** | F | 失败归因到具体模块的精度 | Skill-MAS 无显式失败归因 |
| **Discovery Failure** | F | 未发现所需 skill 的比例 | Skill-MAS 不做 skill 发现 |

### 11.4 融合建议

**Skill-MAS → SDR 可引入的 5 个指标**：

1. **Step-level Uncertainty**：对同一 step 多次路由采样，计算模型选择分布的方差。高方差 = 路由器对该 step 不确定 → 应触发 skill 演化
2. **Task Difficulty Score**：用任务平均成功率反推难度，辅助 Rubar 的条件匹配
3. **Priority-driven Skill Evolution**：用 $p_i = \frac{1}{2}(\tilde{u}_i + \tilde{d}_i)$ 排序需要演化的 skill，肘部截断
4. **Within-skill Contrastive Analysis**：对同一 skill 的高/低分组执行轨迹做对比，提取失败模式
5. **Evolution Cost Tracking**：显式统计进化阶段的 token 消耗，与推理成本分开报告

**SDR → Skill-MAS 可引入的 3 个指标**：

1. **Routing Collapse Detection**：检测 Meta-Skill 是否总是生成相同拓扑（MAS 崩溃）
2. **Module-level Failure Attribution**：将失败归因到 Task Decomposition / Agent Engineering / Workflow Topology 具体模块
3. **Dual Feedback (Pre + Post)**：在 MAS 生成前预评估编排策略质量，生成后后评估执行质量

---

## 附录 A: Skill-MAS 完整指标清单

| # | 指标 | 类别 | 公式/来源 | 论文位置 |
|---|------|------|----------|---------|
| 1 | Avg.Perf | 主性能 | 4基准均值 | Table 1 |
| 2 | Avg.Cost | 成本 | 测试集推理开销 ($) | Table 1 |
| 3 | Per-Benchmark Score | 主性能 | DRB/HLE/BCP/VITA 单项 | Table 1 |
| 4 | Per-task Score | 分布统计 | $s_{i,k}$ 单次 rollout 得分 | §3.2 |
| 5 | Per-task Mean | 分布统计 | $\bar{s}_i = \text{Mean}(s_{i,1..K})$ | §3.2 |
| 6 | Per-task Uncertainty | 分布统计 | $u_i = \text{Std}(s_{i,1..K})$ | §3.2 |
| 7 | Per-task Difficulty | 分布统计 | $d_i = -\bar{s}_i$ | §3.2 |
| 8 | Normalized Uncertainty | 选择性反思 | $\tilde{u}_i = \text{minmax}(u_i)$ | §3.3.1, Eq.3 |
| 9 | Normalized Difficulty | 选择性反思 | $\tilde{d}_i = \text{minmax}(d_i)$ | §3.3.1, Eq.3 |
| 10 | Priority Score | 选择性反思 | $p_i = \frac{1}{2}(\tilde{u}_i + \tilde{d}_i)$ | §3.3.1 |
| 11 | First-order Difference | 选择性反思 | $\delta_j = p_{(j)} - p_{(j+1)}$ | §3.3.1, Eq.4 |
| 12 | Elbow Index | 选择性反思 | $j^* = \arg\max_j \|\delta_j - \delta_{j+1}\| + 1$ | §3.3.1, Eq.4 |
| 13 | Cross-LLM Transfer Δ | 迁移性 | 跨模型同任务增益 | Table 2 Panel A |
| 14 | Cross-Task Transfer Δ | 迁移性 | 同模型跨任务增益 | Table 2 Panel B |
| 15 | Full Transfer Δ | 迁移性 | 跨模型+跨任务增益 | Table 2 Panel C |
| 16 | Transfer Heatmap | 迁移性 | 列归一化 Δ 可视化 | Figure 3 Left |
| 17 | Selective Reflection Gain | 消融 | Ours - Full-Validation | Table 3 |
| 18 | Multi-task Learning Score | 消融 | 多任务进化后性能 | Table 3 |
| 19 | Rollout Scaling | 消融 | K=3/5/7 性能曲线 | Figure 3 Right |
| 20 | Evolution Cost | 成本 | 验证集进化开销 ($) | Table 6 |
| 21 | Best Round Selection | 演化 | $S^* = \arg\max_r \text{val\_score}$ | Algo.1 |
| 22 | Module-level Modification | 演化 | 每轮每模块修改追踪 | Figure 4 |

---

## 附录 B: Skill-MAS 三模块 Skill 结构

### 初始 Meta-Skill（S^(1)）

```yaml
name: unified_meta_agent_skill
description: "A foundational meta-agent skill for generating Multi-Agent Systems (MAS)."
tags: [meta-agent, task-decomposition, agent-engineering, workflow-orchestration]
inputs: [user_query]
```

**三模块**：
1. **Task Decomposition (What)**: Intent & Scope → Sub-task Breakdown → Dependency Mapping → Success Criteria
2. **Agent Engineering (Who)**: Role Profiling → Instruction Design → Input Context Framing
3. **Workflow Topology (How)**: Architectural Topology → Dataflow & State Management → Executable Generation

### 演化后 Meta-Skill（S*）特征

- 从通用框架 → 操作性规范
- 引入显式结构约束（bounded parallelism, merge/synthesis stages, capability-boundary splitting）
- 添加决策和验证规则（two-tier validation, retry protocol, threshold reduction）
- 增加可靠性控制（constraint-aware reasoning, verification gates, backtracking）

---

## 附录 C: 基线方法对比

| 基线 | 范式 | 核心方法 | Skill-MAS 差异 |
|------|------|---------|---------------|
| EvoAgent | 推理时 | 进化算法扩展单Agent→MAS | 有搜索、无累积 Skill 文档 |
| AOrchestra | 推理时 | 层次任务分解→动态节点属性 | 无经验积累 |
| AFlow | 推理时 | MCTS 搜索 workflow | per-query 重跑搜索, 成本高 |
| MAS2 | 训练时 | 训练小模型生成自纠错 workflow | 经验在权重, 难迁移到 frontier LLM |
| MAS-Orchestra | 训练时 | GRPO 优化 function-calling 编排 | 同上 |

---

*文档生成日期: 2026-07-06*
*数据来源: Skill-MAS 原始论文 PDF (32 pages)*
