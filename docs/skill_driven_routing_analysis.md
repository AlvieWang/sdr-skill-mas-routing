# Skill-Driven Dynamic Routing（SDR）

## ——基于Skill的Tool-Agent场景动态路由：指标分析与新度量提案

---

## 0. 背景与动机

基于已有的两个路由方案——**Rubar**（确定性评分标准路由）和**RL-PER**（RL预训练外部路由器），本文档分析当前方案在**skill层面**的缺失，并调研近期skill-based agent系统（SkillOpt、SkillRouter、SkillOrchestra、Arbor、Ares、ToolTree、LaMer、PawBench）所使用的指标，提出一个**skill驱动的动态路由框架**，并识别在skill维度上可以观察到的新指标变化。

**核心问题**：如果我们用skill去调度tools、或者去调度step-level的情况，能在哪些指标上看到新的变化？

---

## 1. Skill-based工作全景：指标体系调研

### 1.1 SkillOpt（Microsoft Research, 2026）

**核心思想**：将skill文档本身作为可训练参数（文本空间优化），冻结目标模型，通过Rollout→Reflect→Edit→Gate循环优化skill。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **Skill质量门控** | Selection Gate Score | 留出验证集上的选择得分，决定是否接受候选编辑 |
| | Train Rollout Score | 训练批次上的执行得分 |
| | Test Performance | 未见测试集上的最终性能 |
| **Skill迁移性** | Cross-model Transfer | 技能跨模型迁移增益（如+15.2分，GPT-5.4→GPT-5.4-nano） |
| | Cross-framework Transfer | 技能跨框架迁移增益（如+31.8分，Codex→Claude Code） |
| | Self-optimization Transfer | 模型自优化增益（+10.4分） |
| **Skill优化过程** | Bounded Edit Budget | "文本学习率"=4 edits/step，防止破坏性重写 |
| | Rejection Buffer | 被拒绝的编辑作为负反馈（消融显示+1.6-5.5分贡献） |
| | Meta-skill | 优化器侧跨epoch记忆（消融显示+0.8-22.5分贡献） |
| **Skill覆盖度** | 52个模型×基准×框架cell | 在所有设置中达到最佳或并列最佳 |

**关键洞察**：SkillOpt证明skill文档可以像模型权重一样被优化，且优化后的skill可跨模型/框架迁移，是可复用的制品。

### 1.2 SkillRouter（arXiv 2603.22455, 2026）

**核心思想**：两阶段retrieve-and-rerank系统，从~80K候选技能中为每个query检索最相关技能。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **路由准确率** | Hit@1 | 排名第一的技能是否为正确技能 |
| | MRR@10 | 前10名中最高排名正确技能的倒数排名 |
| | nDCG@10 | 前10名的归一化折损累积增益 |
| **检索覆盖** | Recall@K (K∈{10,20,50}) | 前K名中恢复的真实技能比例 |
| | FC@10 (Full Coverage) | 前10名是否包含所有真实技能 |
| **下游性能** | Overall Success | 端到端任务成功率 |
| | Recovery vs. Oracle | 相对无技能→金标准技能提升的恢复比例（SkillRouter恢复71%） |
| **效率** | Latency (p50) | 路由延迟（1.2B模型495.8ms vs 16B模型2900.1ms） |

**关键发现**：完整技能文本（包含body）是关键路由信号——移除body导致路由准确率下降31-44个百分点。0.6B微调编码器超越13×规模的8B基础编码器。

### 1.3 SkillOrchestra（arXiv 2602.19672, 2026）

**核心思想**：基于显式、可迁移skill抽象的技能感知编排，通过Skill Handbook建模agent能力与成本。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **Skill级能力建模** | Beta-Bernoulli成功概率 $\phi_{A,\sigma}$ | 每个agent在每个skill上的成功后验分布 |
| | Agent Profile | 模式条件化能力 + 技能条件化成功率 + 成本特征 |
| **路由质量** | Accuracy | 在QA/推理/数学基准上的绝对准确率 |
| | Inference Cost | Token消耗 + 延迟 |
| | Training Cost | 学习路由策略所需的数据量（比RL少700×） |
| **利用率** | Utilization Balance | 查询在不同模型间的分配均匀性 |
| | Routing Collapse | RL方法98%时间调用同一模型；SkillOrchestra均衡分配 |
| **Pareto最优性** | Pareto Frontier Position | 性能-成本权衡的前沿位置 |
| **Skill迁移** | Cross-orchestrator Transfer | Skill Handbook跨编排器迁移无需重训 |
| | Skill Refinement | 技能分裂（高方差时拆分）与合并（不可区分时合并） |

**关键发现**：显式skill建模避免了RL路由器的"路由崩溃"问题（98%调用同一模型），且比RL方法少2-3个数量级训练数据。

### 1.4 Arbor（RUC + Microsoft, 2026）

**核心思想**：通用自主研究agent，skill作为markdown playbook按需加载，驱动Idea Tree的生长。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **实验评估** | Dev/Test Split | Executor在dev split迭代，仅在held-out test超过merge_threshold才合并 |
| | Merge Threshold | 留出测试增益百分比阈值（如5.0%） |
| **Idea Tree** | Tree Growth | 假设树的生长深度和广度 |
| | Novelty Check | idea-check返回novel/partial-overlap/prior-art-exists |
| **Skill加载** | When-to-apply | 触发条件——agent何时应加载并遵循该skill |
| **性能** | Pass Rate | Terminal-Bench 2.0: 77.36% (+7.55) |
| | Accuracy | BrowseComp: 67.67% (+22.34) |
| | Steps | Optimizer Design: 3237.5 (+2.63%) |
| | Medal Rate | MLE-Bench Lite: 86.36% Any-Medal, 77.27% Gold |

**关键发现**：Skill在精确时刻注入指导（而非埋没在system prompt中），对抗LLM的已知失败模式。

### 1.5 Ares（arXiv 2603.07915, 2026）

**核心思想**：per-step动态推理努力选择，轻量级路由器根据交互历史预测每步最低合适推理级别。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **效率** | Reasoning Token Reduction | 最多52.7%推理token减少 |
| **质量保持** | Task Success Rate | "minimal degradation"（极小性能下降） |
| **对比基线** | vs. Fixed High-effort | 固定高推理努力模式 |
| | vs. Low-effort | 低推理努力导致显著性能下降 |
| | vs. Random Selection | 无法保持准确性或提供有意义的成本降低 |

**关键发现**：不同步骤需要不同推理深度——简单步骤（打开URL）用低effort，复杂步骤（导航复杂网站）用高effort。

### 1.6 ToolTree（ICLR 2026）

**核心思想**：基于MCTS的工具规划框架，双反馈（预执行评估+后执行评估）+双向剪枝。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **工具规划质量** | Tool F1 | 工具选择F1分数 |
| | Arg F1 | 参数预测F1分数 |
| | Plan F1 | 整体工具调用序列的规划质量 |
| | Exec F1 | 工具执行结果的正确性 |
| **效率** | Pass Rate | 正确解决方案比例（ToolBench: 69.04%） |
| | Win Rate | 与基线对比的胜出比例 |
| | Efficiency = 边际增益/秒 | 最佳性能-时间平衡在32-64步 |
| | Token Cost | 令牌开销（完整版18.2k vs 基线24.3k） |
| **鲁棒性** | Hallucination Rate | 无效工具调用比例 |
| | Judge Error Rate | LLM-as-judge的评判错误率 |
| **可扩展性** | Tool Library Scale | 14→10,014工具，性能仅降1.62% |
| **搜索质量** | Node Expansion | 扩展节点中位数（~95） |
| | Rollout Count | 中位rollout数（~47） |

**关键发现**：后执行评估对准确率贡献最大（-7.50分），说明执行反馈对引导搜索至关重要。双反馈（前瞻+后顾）比单一反馈更有效。

### 1.7 LaMer（Meta-RL, arXiv 2512.16848, 2025）

**核心思想**：通用Meta-RL框架，让LLM agent在测试时主动探索并从环境反馈中学习。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **任务性能** | Sokoban | +11%性能提升 |
| | MineSweeper | +14%性能提升 |
| | Webshop | +19%性能提升 |
| **泛化能力** | Cross-task Transfer | 在完全未见过的任务上表现出更好泛化性 |
| | Exploration Quality | 主动探索策略的质量 |

**关键发现**：Meta-RL让agent学习到通用的探索策略（而非仅适配特定任务），测试时通过反思进行上下文策略自适应，无需梯度更新。

### 1.8 PawBench（通义实验室, 2026）

**核心思想**：Model × Harness协同评估基准，将模型能力和Harness能力放到同一张评测表里。

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **联合性能** | Model × Harness Score | 模型与框架的协同表现 |
| **Harness Gap** | Harness差异 | 同一模型在不同harness上的分数差（最高11.5分） |
| **五维切片** | Scenario | 办公/软件工程/安全对齐等 |
| | Capability | Logic/Math/Code/Tool_Use/**Skill_Use**/Planning/Self_Verification |
| | Complexity | L1(1-2步)/L2(3-5步)/L3(>5步，含分支或回溯) |
| | Modality | text/multimodal |
| | Environment | closed/open |
| **失败归因** | 6种失败来源 | 模型推理/工具缺失/技能发现弱/工作空间感知差/网络脆弱/完成检查宽 |

**关键发现**：Skill_Use是最困难的能力（平均47.2分），技能发现能力弱是agent失败的关键来源之一。

---

## 2. 现有Rubar/RL-PER的指标局限分析

### 2.1 Rubar现有指标

| 维度 | 现有指标 | 局限 |
|------|---------|------|
| **路由质量** | 条件匹配准确率 | 仅有"满足/不满足"二元判定，缺乏skill级别的细粒度评估 |
| **成本** | Token Cost降低75-85% | 未区分不同skill的cost-effectiveness |
| **质量** | Pass@1保持>90% | 未分解到skill级别的pass rate |
| **TTT** | Rubric迭代频率 | 缺乏skill迁移和skill演化的度量 |
| **技能矩阵** | 7×3能力矩阵 | 静态矩阵，未考虑skill的动态性和迁移性 |

### 2.2 RL-PER现有指标

| 维度 | 现有指标 | 局限 |
|------|---------|------|
| **路由质量** | RL累积折扣回报 | 回报信号是全局的，未分解到skill级别 |
| **成本** | Token成本降低65-80% | 同上，缺乏skill-level成本分析 |
| **质量** | Pass@1保持>85% | 同上 |
| **训练** | RL训练稳定性（KL regularization） | 未度量skill-level的RL收敛质量 |
| **泛化** | Meta-RL快速适应 | 缺乏skill迁移的量化指标 |

### 2.3 关键Gap总结

| Gap | 说明 | 来源系统启示 |
|-----|------|-------------|
| **G1: 缺乏Skill级路由准确率** | Rubar/RL-PER都没有度量"这个步骤的skill需求被正确识别了吗" | SkillRouter的Hit@1/MRR@10 |
| **G2: 缺乏Skill迁移性度量** | 没有评估优化后的路由策略能否迁移到新任务/新模型池 | SkillOpt的cross-model/framework transfer |
| **G3: 缺乏利用率平衡度量** | RL-PER可能存在路由崩溃（总是选择同一模型） | SkillOrchestra的Utilization Balance |
| **G4: 缺乏Skill演化追踪** | Rubar的Rubric迭代没有追踪skill的分裂/合并/演化 | SkillOrchestra的Skill Refinement |
| **G5: 缺乏Pareto前沿度量** | 成本-质量权衡没有在前沿空间中定位 | SkillOrchestra的Pareto Frontier |
| **G6: 缺乏双反馈度量** | 没有预执行和后执行的双信号评估 | ToolTree的dual feedback |
| **G7: 缺乏失败归因** | 失败未分解到skill级别的原因 | PawBench的六维失败归因 |
| **G8: 缺乏探索质量度量** | Meta-RL的探索策略没有被量化 | LaMer的Exploration Quality |

---

## 3. Skill-Driven Dynamic Routing（SDR）框架提案

### 3.1 核心思路

将Rubar的**确定性Rubric**和RL-PER的**RL路由器**统一在**skill抽象层**之下：

```
                    ┌─────────────────────────────────┐
                    │     Skill Abstraction Layer       │
                    │  (统一skill表示 + skill级度量)     │
                    └──────────┬───────────┬───────────┘
                               │           │
                    ┌──────────▼──┐  ┌─────▼─────────┐
                    │  Rubar侧     │  │  RL-PER侧      │
                    │  Rubric→     │  │  RL Router→    │
                    │  Skill条件   │  │  Skill策略     │
                    └──────────┬──┘  └─────┬─────────┘
                               │           │
                    ┌──────────▼───────────▼───────────┐
                    │    Meta-RL Skill Transfer Layer    │
                    │  (跨任务skill迁移 + 快速适应)       │
                    └──────────────┬────────────────────┘
                                   │
                    ┌──────────────▼────────────────────┐
                    │    Tool/Step-Level Routing         │
                    │  (4B/7B/14B模型池 + 工具集)        │
                    └───────────────────────────────────┘
```

### 3.2 Skill定义

在SDR中，skill不再仅是Rubar的7维能力矩阵，而是一个**多层次的、可演化的技能体系**：

```
Skill = (name, description, when_to_apply, capability_profile, cost_profile, evolution_state)
```

| 字段 | 来源 | 说明 |
|------|------|------|
| name | SkillOrchestra | 技能标识符 |
| description | SkillOpt | 自然语言描述（可被文本优化） |
| when_to_apply | Arbor | 触发条件（何时激活此skill） |
| capability_profile | SkillOrchestra | 每个模型在此skill上的Beta-Bernoulli成功概率 |
| cost_profile | RL-PER | 每个模型在此skill上的token成本和延迟 |
| evolution_state | SkillOpt | 技能的演化状态（稳定/待分裂/待合并/新建） |

### 3.3 路由决策流程（融合Rubar + RL-PER + Skill）

```
Step t: 输入步骤上下文 s_t
  ↓
[Skill识别层] —— 借鉴SkillRouter的两阶段检索
  1. Bi-encoder检索: 从skill registry中取top-K候选skill
  2. Cross-encoder重排: 精确排序候选skill
  3. 输出: 当前步骤的活跃skill集合 Σ_t
  ↓
[Skill条件匹配层] —— 借鉴Rubar的Rubric
  4. 对每个活跃skill σ_i ∈ Σ_t:
     - 评估Rubric条件 c(s_t, σ_i) → 满足/不满足
     - 匹配优先级规则（Specificity优先）
  5. 输出: (skill需求, 推荐模型档位)
  ↓
[RL路由决策层] —— 借鉴RL-PER的RL路由器
  6. RL Router R_θ(4B)接收:
     - 步骤上下文 s_t
     - 活跃skill集合 Σ_t
     - skill级能力画像 {φ_{M,σ} | σ ∈ Σ_t}
     - skill级成本画像 {ψ_{M,σ} | σ ∈ Σ_t}
  7. 输出: 模型选择分布 π(M_k | s_t, Σ_t)
  ↓
[双反馈评估层] —— 借鉴ToolTree的dual feedback
  8. 预执行评估: r_pre(s_t, M_k) — 预测此模型在此skill上的有用性
  9. 执行: M_k处理s_t
  10. 后执行评估: r_post(s_t, M_k, output) — 评估实际贡献
  ↓
[Meta-RL更新层] —— 借鉴LaMer的Meta-RL
  11. 跨episode累积skill级经验
  12. 反思式上下文策略自适应（无梯度更新）
  13. LoRA更新RL Router（可选，有梯度更新）
```

### 3.4 与Rubar和RL-PER的融合关系

| 原方案 | 在SDR中的角色 | 改进点 |
|--------|-------------|--------|
| **Rubar Rubric** | Skill条件匹配层 | Rubric条件从7维静态矩阵→动态skill registry |
| **Rubar Memory Store** | Skill级经验存储 | 从SQLite FTS→增加skill演化追踪 |
| **Rubar TTT** | Skill演化机制 | 从Rubric迭代→skill分裂/合并/新建 |
| **RL-PER Router** | RL路由决策层 | 从全局RL→skill条件化RL |
| **RL-PER Reward** | Skill级奖励分解 | 从全局reward→skill-level reward |
| **RL-PER Meta-RL** | Meta-RL更新层 | 从任务级迁移→skill级迁移 |

---

## 4. 新指标体系：Skill维度上的度量变化

### 4.1 核心新指标总览

以下是引入skill维度后，我们可以在以下指标上观察到**新的变化**：

#### A类：Skill级路由准确率指标

| 新指标 | 定义 | 对标系统 | 预期变化 |
|--------|------|---------|---------|
| **Skill Hit@1** | 路由器识别的top-1 skill是否为正确skill | SkillRouter | Rubar的7维匹配→细粒度skill命中 |
| **Skill MRR@10** | 前10候选skill中正确skill的倒数排名 | SkillRouter | 评估skill检索质量 |
| **Skill Recall@K** | 前K候选中恢复真实skill的比例 | SkillRouter | 评估skill覆盖度 |
| **Skill FC@10** | 前10是否包含所有必需skill（多skill步骤） | SkillRouter | 多skill步骤的全覆盖 |
| **Skill Conditioned Routing Accuracy** | 给定正确skill后，模型选择是否正确 | SkillOrchestra | skill→model映射的准确率 |

**预期变化**：
- Rubar的"条件匹配准确率"是粗粒度的（7维×3模型=21种组合），SDR可以度量数百种skill的精确命中率
- RL-PER的"累积折扣回报"是全局的，SDR可以分解到每个skill的回报贡献

#### B类：Skill迁移与适应性指标

| 新指标 | 定义 | 对标系统 | 预期变化 |
|--------|------|---------|---------|
| **Cross-task Skill Transfer** | 在任务A上学习的skill在任务B上的增益 | SkillOpt (+15.2) | 度量Meta-RL的skill迁移效果 |
| **Cross-model Skill Transfer** | 模型池变化后skill的保持率 | SkillOpt (+15.2) | 新模型加入时的快速适配 |
| **Cross-framework Skill Transfer** | 从SWE-bench→WebArena的skill迁移 | SkillOpt (+31.8) | 跨场景泛化 |
| **Skill Adaptation Speed** | 新任务上前N步达到稳定skill命中率所需步数 | LaMer | Meta-RL的快速适应量化 |
| **Skill Exploration Quality** | 在未见任务上的主动探索得分 | LaMer (+11-19%) | 探索vs利用的平衡 |

**预期变化**：
- Rubar的"TTT频率"是全局的（每100步/每10步），SDR可以度量每个skill的独立演化速度
- RL-PER的"Meta-RL快速适应"缺乏量化，SDR可以用Skill Adaptation Speed度量

#### C类：利用率与稳定性指标

| 新指标 | 定义 | 对标系统 | 预期变化 |
|--------|------|---------|---------|
| **Skill-level Utilization Balance** | 每个skill在各模型间的分配均匀性 | SkillOrchestra | 防止路由崩溃 |
| **Routing Entropy** | 路由决策的熵（越高越均衡） | SkillOrchestra | 量化路由多样性 |
| **Routing Collapse Rate** | 单一模型被选择的频率超过阈值(如95%)的比例 | SkillOrchestra (RL 98%) | RL-PER的路由崩溃检测 |
| **Skill-level Cost-Effectiveness** | 每个skill的单位成本通过率 | RL-PER + SkillOrchestra | skill级性价比 |
| **Pareto Frontier Coverage** | skill级帕累托前沿的覆盖面积 | SkillOrchestra | 多目标权衡的质量 |

**预期变化**：
- Rubar的"Token成本降低75-85%"是全局的，SDR可以分解到每个skill的成本贡献
- RL-PER可能存在路由崩溃（SkillOrchestra发现RL方法98%调用同一模型），SDR可以检测和缓解

#### D类：Skill演化与质量指标

| 新指标 | 定义 | 对标系统 | 预期变化 |
|--------|------|---------|---------|
| **Skill Refinement Rate** | 单位时间内skill分裂/合并/新建的次数 | SkillOrchestra | skill体系的活跃度 |
| **Skill Stability** | 某skill的能力画像在N步内的方差 | SkillOrchestra Beta-Bernoulli | skill的可靠性 |
| **Skill Coverage** | skill registry覆盖的任务模式比例 | SkillRouter (80K skills) | skill体系的完备性 |
| **Skill Quality Gate** | 新skill通过验证门控的通过率 | SkillOpt (Selection Gate) | skill质量保证 |
| **Skill Velocity** | skill优化的收敛速度（epoch到稳定的步数） | SkillOpt (2-4 epoch) | 优化效率 |

**预期变化**：
- Rubar的"Rubric迭代"是规则级别的，SDR可以追踪skill从发现→稳定→分裂/合并的完整生命周期
- RL-PER的"RL训练稳定性"是全局的，SDR可以度量每个skill的RL收敛质量

#### E类：双反馈与搜索质量指标

| 新指标 | 定义 | 对标系统 | 预期变化 |
|--------|------|---------|---------|
| **Pre-execution Skill Match** | 预执行评估与实际skill需求的匹配度 | ToolTree (r_pre) | 前瞻准确性 |
| **Post-execution Skill Contribution** | 后执行评估的skill级贡献 | ToolTree (r_post) | 后顾准确性 |
| **Feedback Gap** | 预执行与后执行评估的差距 | ToolTree (消融-7.50) | 预测偏差 |
| **Skill-level Plan F1** | skill序列的规划质量 | ToolTree (Plan F1) | 多skill步骤的规划 |
| **Skill-level Exec F1** | skill执行结果的正确性 | ToolTree (Exec F1) | skill执行质量 |

**预期变化**：
- Rubar只有"条件匹配"（预执行），没有后执行验证，SDR可以增加双反馈
- RL-PER的"reward"是执行后的，SDR可以增加预执行预测

#### F类：失败归因与诊断指标

| 新指标 | 定义 | 对标系统 | 预期变化 |
|--------|------|---------|---------|
| **Skill-level Failure Attribution** | 失败归因到具体skill的比例 | PawBench (6种归因) | 精确定位失败原因 |
| **Skill Discovery Failure Rate** | 因skill发现弱导致的失败比例 | PawBench (Skill_Use 47.2) | skill检索质量 |
| **Skill-Model Mismatch Rate** | skill需求与所选模型能力不匹配的比例 | PawBench | 路由错误归因 |
| **Harness-Skill Interaction Score** | 模型×harness在特定skill上的协同表现 | PawBench (Harness Gap) | 联合评估 |

**预期变化**：
- Rubar/RL-PER的失败是全局的（Pass@1通过/不通过），SDR可以归因到具体skill
- PawBench发现Skill_Use是最难的能力（47.2分），SDR可以专门度量skill发现质量

### 4.2 新旧指标对照矩阵

| 评估维度 | Rubar旧指标 | RL-PER旧指标 | SDR新指标 | 新增信号 |
|---------|-----------|-------------|----------|---------|
| **路由准确率** | 条件匹配率 | 累积折扣回报 | Skill Hit@1/MRR@10/FC@10 | 细粒度skill命中 |
| **成本效率** | Token降低75-85% | Token降低65-80% | Skill-level Cost-Effectiveness | 每个skill的性价比 |
| **质量保持** | Pass@1 >90% | Pass@1 >85% | Skill-conditioned Pass@1 | 每个skill的通过率 |
| **迁移性** | 未度量 | Meta-RL快速适应 | Cross-task/model/framework Transfer | 量化迁移增益 |
| **稳定性** | 规则冲突率 | RL训练稳定性 | Routing Collapse Rate + Entropy | 路由崩溃检测 |
| **演化** | Rubric迭代频率 | LoRA更新 | Skill Refinement Rate + Velocity | skill生命周期追踪 |
| **反馈** | 单一（条件匹配） | 单一（reward） | Dual Feedback (pre+post) | 前瞻+后顾 |
| **失败归因** | 无 | 无 | Skill-level Failure Attribution | 精确定位 |
| **Pareto** | 无 | 无 | Pareto Frontier Coverage | 多目标权衡定位 |

---

## 5. Meta-RL在SDR中的角色：Skill级跨任务迁移

### 5.1 Meta-RL与Skill的结合点

LaMer证明Meta-RL可以让agent学习到通用的探索策略，在未见任务上快速适应。在SDR中，Meta-RL的角色从"任务级适应"升级为"**skill级适应**"：

| Meta-RL维度 | 传统RL-PER | SDR中的Meta-RL |
|------------|-----------|---------------|
| **适应目标** | 整体路由策略 | 每个skill的独立路由策略 |
| **迁移粒度** | 任务→任务 | skill→skill（可跨任务） |
| **适应信号** | 全局reward | skill-level reward分解 |
| **适应速度** | N步达到稳定 | 每个skill独立适应速度 |
| **探索策略** | 全局探索 | skill级探索（对不熟悉的skill探索更多） |

### 5.2 Skill级Meta-RL形式化

**传统RL-PER的Meta-RL目标**：
$$\theta^* = \arg\max_\theta \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})}\left[ \mathbb{E}_{\pi_{R_\theta}}\left[ \sum_{t=1}^{T} \gamma^{t-1} r_t \mid \mathcal{T} \right] \right]$$

**SDR的Skill级Meta-RL目标**：
$$\theta^* = \arg\max_\theta \mathbb{E}_{\sigma \sim p(\sigma)}\left[ \mathbb{E}_{\pi_{R_\theta}}\left[ \sum_{t=1}^{T} \gamma^{t-1} r_t^{(\sigma)} \mid \sigma \right] \right]$$

其中 $p(\sigma)$ 是skill分布，$r_t^{(\sigma)}$ 是skill $\sigma$ 在步骤 $t$ 的分解奖励：

$$r_t^{(\sigma)} = \alpha \cdot Q(s_t, M(t), \sigma) - \beta \cdot C(M(t), \sigma) - \gamma \cdot L(M(t), \sigma)$$

### 5.3 Skill级Meta-RL的三个阶段

| 阶段 | 操作 | 对标系统 |
|------|------|---------|
| **阶段1: Skill级SFT冷启动** | 在Open-AgentRL数据上，为每个skill学习初始路由策略 | SkillOpt (SFT) + RL-PER (SFT) |
| **阶段2: Skill级RL强化** | 在RL训练中，reward按skill分解，每个skill独立优化 | SkillOrchestra (Beta-Bernoulli) + RL-PER (RL) |
| **阶段3: Meta-RL跨任务Skill迁移** | 在新任务上，通过反思式上下文自适应，快速迁移已学skill的策略 | LaMer (Meta-RL) + SkillOpt (cross-transfer) |

### 5.4 Skill级迁移的量化

借鉴SkillOpt的迁移评估方法，SDR可以度量以下迁移场景：

| 迁移场景 | 定义 | 预期增益 | 度量方法 |
|---------|------|---------|---------|
| **Skill→新任务** | 在SWE-bench上学的debug skill在WebArena的debug步骤上 | +10-15分 | Skill-conditioned Pass@1差异 |
| **Skill→新模型池** | 模型从4B/7B/14B→3B/7B/14B | +5-10分 | Beta-Bernoulli后验迁移 |
| **Skill→新框架** | SWE-agent框架→OpenHands框架 | +15-20分 | Cross-framework Transfer |
| **Meta-RL→未见任务** | 完全未见任务类型上的快速适应 | +11-19% | LaMer式探索质量 |

---

## 6. 在Tool-Agent场景下的具体应用

### 6.1 场景设定：SWE-bench + WebArena联合评估

| 维度 | SWE-bench | WebArena | 联合 |
|------|-----------|----------|------|
| **Skill类型** | 代码生成/调试/测试/验证 | 网页交互/检索/表单填写/导航 | 7+4=11种skill |
| **工具集** | 代码执行器/文件系统/搜索 | 浏览器/表单/API/数据库 | 10+种工具 |
| **步骤深度** | L3(>5步) | L2-L3 | 复杂长链 |
| **Skill切换** | 频繁（每3-5步切换） | 中等（每5-8步切换） | 高频切换 |

### 6.2 Skill→Tool→Model的三层路由

```
Step t: "修复auth模块的SQL注入漏洞"
  ↓
[Skill识别] → Σ_t = {debug, code_gen, verify}
  ↓
[Tool匹配] → debug需要(code_search), code_gen需要(code_editor), verify需要(test_runner)
  ↓
[Model路由] → debug→M_S(7B), code_gen→M_M(14B), verify→M_T(4B)
  ↓
[执行] → 7B搜索漏洞 → 14B生成修复 → 4B运行测试
  ↓
[双反馈] → r_pre: 预测各步有用性 → r_post: 评估实际贡献
  ↓
[Skill更新] → debug skill的Beta-Bernoulli后验更新 → Meta-RL累积经验
```

### 6.3 关键新指标在场景中的体现

| 指标 | SWE-bench中的体现 | WebArena中的体现 | 联合迁移 |
|------|------------------|-----------------|---------|
| **Skill Hit@1** | 7种skill的识别准确率 | 4种skill的识别准确率 | 跨场景skill泛化 |
| **Skill-level Cost-Effectiveness** | debug skill的性价比 | 检索skill的性价比 | 跨场景成本对比 |
| **Routing Collapse Rate** | 14B被过度使用的比例 | 7B被过度使用的比例 | 路由多样性 |
| **Cross-task Skill Transfer** | — | — | SWE的verify skill→WebArena的验证步骤 |
| **Dual Feedback Gap** | 预测vs实际的偏差 | 同左 | skill级预测校准 |
| **Skill-level Failure Attribution** | 哪个skill导致失败 | 同左 | 跨场景失败模式 |

---

## 7. 新增信号的预期变化幅度

基于调研数据，以下是我们预期在引入skill维度后能观察到的新信号变化幅度：

### 7.1 路由准确率维度

| 指标 | 当前（无skill） | 预期（有skill） | 变化幅度 | 依据 |
|------|---------------|----------------|---------|------|
| 路由命中率 | ~70%（粗粒度条件匹配） | ~85-90%（skill级命中） | +15-20pp | SkillRouter: 74% Hit@1 |
| 多skill步骤覆盖 | 未度量 | FC@10 ~60-70% | 新增信号 | SkillRouter多skill场景 |
| skill→model匹配 | ~80%（7维矩阵） | ~90%（细粒度画像） | +10pp | SkillOrchestra Beta-Bernoulli |

### 7.2 成本效率维度

| 指标 | 当前 | 预期 | 变化幅度 | 依据 |
|------|------|------|---------|------|
| Token成本降低 | 75-85%(Rubar) / 65-80%(RL-PER) | 80-90% | +5-10pp | Skill级精准路由减少浪费 |
| 推理token减少 | 未度量 | ~50%+ | 新增信号 | Ares: 52.7% |
| 利用率平衡 | 未度量 | 熵 >2.0 bits | 新增信号 | SkillOrchestra: RL熵≈0（崩溃） |
| 路由崩溃率 | 未度量 | <5% | 新增信号 | SkillOrchestra: RL 98%→<5% |

### 7.3 迁移与适应性维度

| 指标 | 当前 | 预期 | 变化幅度 | 依据 |
|------|------|------|---------|------|
| 跨任务迁移 | 未度量 | +10-15分 | 新增信号 | SkillOpt: +15.2 |
| 跨模型迁移 | 未度量 | +5-10分 | 新增信号 | SkillOpt: +15.2 |
| 适应速度 | "快速适应"（无量化） | 5-10步达稳定 | 新增信号 | LaMer: 反思式适应 |
| 探索质量 | 未度量 | +11-19% | 新增信号 | LaMer: +11-19% |

### 7.4 演化与诊断维度

| 指标 | 当前 | 预期 | 变化幅度 | 依据 |
|------|------|------|---------|------|
| Skill演化速率 | 未度量 | 2-4 epoch收敛 | 新增信号 | SkillOpt: 2-4 epoch |
| 失败归因精度 | 0%（无归因） | 80%+可归因 | 新增信号 | PawBench: 6维归因 |
| 双反馈偏差 | 无预执行 | <15%偏差 | 新增信号 | ToolTree: judge error 25.8% |
| Pareto覆盖 | 无 | >80%前沿覆盖 | 新增信号 | SkillOrchestra: Pareto最优 |

---

## 8. 实施路线图

### Phase 1: Skill Registry构建（借鉴SkillRouter + SkillOpt）

1. 从SWE-smith 66K轨迹中提取初始skill registry
2. 使用SkillRouter的两阶段检索建立skill检索系统
3. 使用SkillOpt的文本空间优化持续改进skill文档

### Phase 2: Skill条件化路由（借鉴Rubar + SkillOrchestra）

1. 将Rubar的7维技能矩阵替换为动态skill registry
2. 使用SkillOrchestra的Beta-Bernoulli建模每个模型在每个skill上的能力
3. Rubric条件从静态阈值→skill级动态匹配

### Phase 3: Skill级RL训练（借鉴RL-PER + LaMer）

1. RL-PER的reward按skill分解
2. 引入LaMer的Meta-RL跨episode训练
3. 度量skill级RL收敛质量

### Phase 4: 双反馈与诊断（借鉴ToolTree + PawBench）

1. 引入ToolTree的预执行+后执行双反馈
2. 建立PawBench式的五维失败归因体系
3. 度量skill级失败模式

### Phase 5: 跨任务Skill迁移评估

1. 在SWE-bench上训练，在WebArena上评估skill迁移
2. 度量Cross-task Skill Transfer增益
3. 建立skill迁移基准

---

## 9. 总结

### 核心论点

通过引入skill抽象层，Rubar和RL-PER可以从**模型级路由**升级为**skill级路由**，在以下维度获得新的可观测信号：

| 维度 | 新信号 | 来源 |
|------|--------|------|
| **精度** | Skill级路由准确率（Hit@1/MRR/FC） | SkillRouter |
| **效率** | Skill级性价比 + 路由崩溃检测 | SkillOrchestra |
| **迁移** | Cross-task/model/framework Skill Transfer | SkillOpt |
| **演化** | Skill生命周期追踪（分裂/合并/收敛速度） | SkillOpt + SkillOrchestra |
| **反馈** | 双反馈（预执行+后执行）skill级评估 | ToolTree |
| **诊断** | Skill级失败归因 | PawBench |
| **探索** | Meta-RL skill级探索质量 | LaMer |
| **权衡** | Skill级Pareto前沿覆盖 | SkillOrchestra |

### 关键创新

1. **Skill作为路由的一等公民**：不再仅是7维矩阵，而是可演化、可迁移、可度量的独立实体
2. **Rubar + RL-PER的skill级融合**：Rubric提供skill条件，RL Router学习skill→model映射，Meta-RL实现skill迁移
3. **双反馈skill级评估**：预执行预测skill需求，后执行验证skill贡献
4. **PawBench式失败归因**：将失败从"通过/不通过"升级为"哪个skill的哪个环节失败"
5. **Meta-RL skill级迁移**：从任务级适应→skill级适应，实现更细粒度的跨任务泛化

---

*文档版本：V1*
*日期：2026-07-04*
