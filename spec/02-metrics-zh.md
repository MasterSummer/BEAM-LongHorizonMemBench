# 02 — 指标数学与记分卡

> **状态**：v1 规范。任务 17、18、19 和 22 精确实现了本定义。
> **v1 范围**：维度2（目标导向利用）、维度3（目标漂移）、维度4（检索质量）
> 和维度7（通过 Memory ROI 的 Token/资源效率）。维度1、5、6、8、9 推迟到
> 扩展点存根（见第5节）。

## 0. 当前 long-horizon qualification 的主测量合同

后续章节保留早期 v1 的通用记分卡定义；当前 Software long-horizon 论文不以
Memory ROI 或检索 recall 作为 headline。主科学对象是：在 workspace 与 task
memory 两条持久通道并存时，系统能否跨越长因果依赖链，维持当前权威任务状态，
并让该状态继续控制下一步可执行行为。

### 0.0 Long-horizon span 与 interaction tier

Long-horizon 不以 token 数计量。每条轨迹报告 effective transition、handoff、
dependency depth、state/event distance，以及 `policy_evaluated`、
`frozen_replay`、`environment_generated` 的独立计数。进一步报告：

- `minimum_decision_causal_span` / `maximum_decision_causal_span`；
- `semantic_effect_coverage`；
- `consumed_prefix_effect_fraction`；
- `anti_padding_verified`；
- `policy_conditioned_future_step_count`；
- `policy_steps_with_downstream_effect_count`；
- `policy_dependent_decision_count`；
- `policy_dependency_coverage`；
- `interaction_mode`；
- `online_long_horizon_agent_execution_supported`。

当前 matched/horizon release 的 interaction mode 是
`replay_backed_critical_decision`。因此 long-horizon 估计量表示审计过的长程
因果前缀对关键状态决策的影响，不能写成被测 policy 在线完成了完整轨迹。
≥200 门槛使用被评分决策的 effective causal ancestors；每个计数步骤必须产生
唯一 semantic effect 并消费声明前驱效果。Episode 总长度、digest-only 链、
重复 observation 和未进入决策因果祖先的尾部步骤均不能满足门槛。

### 0.1 C1：同决策、workspace-adjusted 的状态控制

每个 counterfactual group 固定 terminal request、current-state semantics、
action catalog、gold action、opaque option map、workspace semantics、package 和
hidden checker，只改变上游历史为 `static`、`evolution` 或
`hierarchical_conflict`。对 memory condition `m`、history `h` 和 matched
static `s`：

```text
G(m,h) = Y(m,h) - Y(workspace,h)
identified_construct_effect(m,h) = G(m,h) - G(m,s)
```

主报告同时给出 short/medium/long 的 joint horizon-dose amplification，但不得
把 transition、dependency depth 和 handoff 同时变化的结果解释为 pure handoff
effect。

`oracle_current_state` 是 terminal solvability control；`full_context` 是
history availability/interpretability control。两者必须对每个 policy、每个
matched group、三个 history variant 分别通过。Oracle 失败时不能声称任务可解；
full-context 失败时不能把完整历史解释困难归因为 memory selection。
每个 SCEU 还必须将所有会使候选 action 有效或无效的当前 state 纳入 required
closure；replace/revoke、authority 与 scope precedence 是所有 condition 共享
的任务语义，而不是 oracle-only 提示。

### 0.2 C2：goal-relative longitudinal drift

Headline 不是单点错误率，而是从“先遵守、后违背”识别出的行为漂移。四类
programmatic violation 为 `constraint_loss`、`plan_deviation`、`stale_state`
和 `local_over_global`。单点 violation 与 adherence-anchored onset、survival、
persistence、recovery 必须分开；没有同一 state-lineage prior-adherence anchor
的 trajectory 不进入 longitudinal drift 分母。Category-only legacy trajectory
只能描述；oracle-current-state/full-context 在同 lineage 上发生 onset 时，不得
将该类别解释为 memory-specific drift。

### 0.3 C3：同一决策的可观测故障链

对 workspace 中不显式可恢复、但当前 action 必需的状态，重建：

```text
stored -> backend-retrieved -> model-visible
       -> intervention evidence -> checked behavior
```

最早有证据支持的失败分别为 storage、retrieval、exposure 或 utilization。
Storage provenance 不可观察时输出 `storage_evidence_unavailable`，不能当作
storage failure。Visible 也不等于 used；只有 repeat-stable、state-targeted
intervention 才提供 detected unique causal-effect lower bound。无 effect 不能证明
unused，因为其他 memory object 或 workspace 可能冗余地提供相同状态。
只要 repeat-stable intervention 改变 action 或 checker，就记为检测到 unique
causal effect；`beneficial`、`harmful` 与 `causal_direction_ambiguous` 描述的是
方向，不改变“存在可观测因果影响”的判断。旧字段 `behaviorally_used` 仅保留为
该 lower bound 的兼容名称。

为证明最终准确率会掩盖机制差异，定义 outcome-equivalent fault-profile
divergence。令 `OE` 为相同 policy、readout、episode、SCEU、opportunity、
checkpoint、current-state contract、selected action 和 correctness 的
memory-condition pairs，`F` 为 earliest stage 加 utilization subtype：

```text
D_OE = sum[(F_a != F_b) for (a,b) in OE] / |OE|
```

`D_OE > 0` 表示相同 checked outcome 隐藏了不同 observed repair target。Pair
仅是同一决策内的 dependent descriptive comparison，不是新的独立样本；
`D_OE = 0` 也不能据此宣称系统等价。

---

## 1. Memory ROI — 核心指标

Memory ROI 回答：*记忆系统是否改善了性能，代价是什么？*
它是按系统、按任务族、按赛道计算的反事实比率，绝不是一个单一的
裸排行榜数值。

### 1.1 定义

对于每个 episode `e`、种子 `s` 和实验条件（系统）`c`：

```
score(c, e, s)    = 任务分数 ∈ [0, max_achievable(e)]   （见第4节）
gain(c, e, s)     = score(c, e, s) − score(no_memory, e, s)
```

增益下限为 `max(0, gain)`。负增益保留用于 ROI 计算
（产生负 ROI），但增益下限也被报告以区分
"无害"与"真正有益"的系统。

**归一化增益**将反事实改进限制在 [-1, 1] 范围内，以防止
由于异常大的增益而导致任何单个 episode 主导聚合结果：

```
normalized_gain(c, e, s) = clamp(
    gain(c, e, s) / max(ε, max_achievable(e) − score(no_memory, e, s)),
    −1,
    1
)
```

其中 `ε = 0.001`（当 no_memory 已达到最大值时防止除以零）。
`max_achievable(e)` 是 episode 的已知满分，由任务族的真实答案确定。

**成本向量** → **标量成本**：每个 episode-条件运行产生一个 `CostVector`
（见第 1.3 节）。向量通过已声明的转换表
（`configs/cost_weights.yaml`）标量化为 **token 当量**：

```
cost(c, e, s) = scalarize(CostVector(c, e, s))
```

**Memory ROI** 在 episode 和种子上聚合：

```
mean_gain(c) = mean_{e,s} normalized_gain(c, e, s)
mean_cost(c) = mean_{e,s} cost(c, e, s)

ROI(c) = mean_gain(c) / mean_cost(c)
```

ROI 以 **bootstrap 置信区间**（95%，有种子，n=10,000 次重采样）在
配对 (episode, seed) 单位上报告。单位：每 token 当量成本的增益（通常
按每 1,000 token 报告以提高可读性）。

### 1.2 边界情况策略

| 情况                                    | 策略                                                                                                                      |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **无记忆对照组**                            | ROI = `N/A`（非 0，不报告）。无记忆条件本身就是用来计算所有其他系统增益的基线；为其计算 ROI 在定义上是不成立的。                                                       |
| **成本接近零**（`mean_cost < ε_cost`）       | ROI 标记为 `undefined-lowcost`，**绝不**标记为 `+inf` 或 `NaN`。帕累托前沿和成本向量分解仍会报告；增益单独报告。`ε_cost = 1.0`（平均每 episode 少于 1 token 当量）。 |
| **负增益**（`gain < 0`）                   | 产生负 ROI。这是允许的并被报告——这意味着记忆系统损害了性能。始终伴随增益下限（下限为 0）和帕累托上下文，以免负 ROI 系统被误读为"优于基线"。                                           |
| **max_achievable ≈ score(no_memory)** | `normalized_gain` 中的 `ε` 钳位防止除以零。完全饱和的基线（无改进空间）无论原始增益如何都产生接近零的归一化增益——反映该任务未测试记忆。此类 episode 被标记供分析者关注。                   |
| **未定义条件**                             | 如果 no_memory 超时或崩溃，该 (episode, seed) 的 episode 被排除在配对增益计算之外；系统的原始分数 + 成本仍被报告。记录为部分配对。                                   |
| **NaN / +inf**                        | 禁止。任何会产生 `+inf` 或 `NaN` 的计算路径必须改为发出 `undefined-lowcost` 或 `N/A`。                                                        |

### 1.3 CostVector

每个 `CostVector` 记录在给定条件下运行 episode 的全生命周期成本。
所有字段在整个 episode 上累计：

| 字段                        | 单位    | 描述                                                                                  |
| ------------------------- | ----- | ----------------------------------------------------------------------------------- |
| `agent_input_tokens`      | token | 馈入智能体 LLM 的 token（提示 + 上下文）。                                                        |
| `agent_output_tokens`     | token | 智能体 LLM 生成的 token。                                                                  |
| `mem_internal_in_tokens`  | token | 记忆系统内部 LLM 调用消耗的 token（如 Mem0 提取、Cognee cognify、Letta 区块编辑）。在 `memory_scope()` 内计数。 |
| `mem_internal_out_tokens` | token | 记忆系统内部 LLM 生成的 token。                                                               |
| `embedding_tokens`        | token | 为向量存储嵌入的总 token（输入侧）。                                                               |
| `embedding_calls`         | 次数    | 嵌入 API 调用次数。                                                                        |
| `storage_bytes`           | 字节    | 记忆后端使用的峰值或累计存储。                                                                     |
| `retrieval_latency_ms`    | 毫秒    | `search()` 调用的总挂钟延迟。                                                                |
| `write_latency_ms`        | 毫秒    | `add_memory()` 调用的总挂钟延迟。                                                            |
| `update_latency_ms`       | 毫秒    | `update_memory()` / `delete_memory()` 调用的总挂钟延迟。                                     |
| `reflection_tokens`       | token | 显式 `reflect()` / 整合调用消耗的 token（在 `memory_scope()` 下计数）。                             |
| `num_retrieval_calls`     | 次数    | `search()` 调用次数。                                                                    |

**排除项**：一次性数据集生成 token、判定器 token 和表面渲染 token 不计入系统 CostVector。每项在其自身作用域下运行。

**标量化**：`scalarize(cv, weights, sheet) -> float` 产生一个单一的 token 当量数值。延迟和存储使用 `configs/cost_weights.yaml` 中的转换表进行转换（例如 1 毫秒延迟 = 0.1 token 当量；1 KB 存储 = 0.01 token 当量）。标量化权重可按运行配置固定，并在每个记分卡中记录。

### 1.4 计算示例

考虑一个 3-episode 任务族，所有 episode 的 `max_achievable = 1.0`。

| Episode | no_memory 分数 | 系统 A 分数 | max_achievable − no_memory | gain | normalized_gain | 系统 A 成本（token 当量） |
|---------|---------------|------------|---------------------------|------|-----------------|----------------------|
| e1 | 0.30 | 0.70 | 0.70 | +0.40 | clamp(0.40/0.70, −1, 1) = 0.571 | 5200 |
| e2 | 0.45 | 0.60 | 0.55 | +0.15 | clamp(0.15/0.55, −1, 1) = 0.273 | 4800 |
| e3 | 0.20 | 0.90 | 0.80 | +0.70 | clamp(0.70/0.80, −1, 1) = 0.875 | 5100 |

```
mean_gain(A)   = (0.571 + 0.273 + 0.875) / 3 = 0.573
mean_cost(A)   = (5200 + 4800 + 5100) / 3 = 5033.3
ROI(A)         = 0.573 / 5033.3 = 1.14 × 10⁻⁴ 每 token
               ≈ 每百万 token 当量 114
```

对 3 个配对差异的 bootstrap CI（95%）可能产生每百万 token `[92, 136]`。
系统 A 的增益下限为 `mean(max(0, gain)) / mean(cost)`，在此例中等于 ROI（所有增益为正）。

**负增益示例**：系统 B 的 mean_normalized_gain = −0.15，mean_cost = 4000：

```
ROI(B) = −0.15 / 4000 = −3.75 × 10⁻⁵ 每 token
       ≈ 每百万 token 当量 −37.5
```

报告为"负 ROI（系统损害了性能）"，增益下限 = 0。

**成本接近零示例**：系统 C 的 mean_cost = 0.5（< ε_cost = 1.0）：

```
ROI(C) = undefined-lowcost（而非 +inf）
```

增益（如有）单独报告；成本向量分解被保留。

---

## 2. 目标漂移与行为稳定性（维度3）

目标漂移衡量当前权威任务状态对行为的控制是否随长期轨迹而退化。判定依据是版本化状态和可执行 action checker，而不是 LLM judge。

### 2.1 定义

每个 episode 携带具有 authority、scope、version 和 validity window 的目标、约束、计划与决策。每个 continuation opportunity 预注册其 drift-eligible categories，未进入某类别风险集的决策不进入该类别分母。

四类 canonical drift-compatible violation 为：

1. `constraint_loss`：仍然有效的全局约束不再控制行为；
2. `plan_deviation`：动作偏离当前有效计划，包括过早采用未来版本；
3. `stale_state`：已撤销、替换或失效的旧状态重新控制行为；
4. `local_over_global`：局部子目标、低权威决策或 scoped exception 错误覆盖全局目标或高权威约束。

必须区分：

- **drift-compatible violation**：某个 eligible action 触发上述程序化错误类别；
- **observed longitudinal drift**：同一 episode、policy、condition、readout、类别和 state lineage 中，先在较早的 distinct checkpoint 观察到 adherence，随后该 lineage 才出现 violation。

第一次被观察时已经错误的行为只能记作 violation，不能声称发生了纵向 drift。有效状态更新后的正确行为变化属于 adaptation，不记作 violation 或 drift。

### 2.2 公式

对 episode `e`、实验 cell `s`、类别 `c`、state lineage `l` 和 distinct
checkpoint `t`，定义：

```
E(e,s,c,l,t) = 该 checkpoint 对类别 c 的 lineage l eligible
V(e,s,c,l,t) = eligible 且 action checker 检测到类别 c 对 lineage l 的 violation

violation_rate(c) = Σ V(e,s,c,l,t) / Σ E(e,s,c,l,t)

D(e,s,c,l,t) = V(e,s,c,l,t) 且存在 t' < t，
               E(e,s,c,l,t')=1 且 V(e,s,c,l,t')=0
```

只有同一 lineage 存在 prior-adherence anchor 且之后还有 eligible checkpoint
的 trajectory 才是 `drift_evaluable`。observed drift incidence 的分母只包含
这些 trajectory；没有 anchor 的 trajectory 保留在 violation 报告中，但不以
0 填入 drift 分母。

每个 episode/cell/category 输出：

- violation incidence 和 first-violation session；
- 是否有 adherence anchor、是否 drift-evaluable；
- first observed-drift session；
- distinct-checkpoint persistence；
- valid-update / fresh-reminder recovery；
- adherence-anchored drift-free survival。

episode 是统计单位，SCEU 和 lineage 只是 episode 内重复观测。matched
static/evolution/conflict 只有一个共同终点时，可以报告
`drift-compatible violation excess`，但不能将该单点差值称为 longitudinal
drift onset。

### 2.3 边界情况策略

| 情况 | 策略 |
|------|------|
| 有效 update 后改变动作 | 正确 adaptation，不处罚；以 checkpoint 的 current authoritative state 为准。 |
| 首次 eligible checkpoint 已违规 | 记作 violation；没有 prior adherence，不记 observed drift。 |
| 同一 checkpoint 多个 probe | 合并成一个时间点，不能制造 persistence 或 onset。 |
| 不同 state lineage 的 adherence 与 violation | 分成不同 trajectory；前者不能为后者提供 onset anchor。 |
| 同一 SCEU/category 对应多个 focal lineage | 数据生成时拆成多个 SCEU；报告拒绝将一个 category flag 广播到多个 lineage。 |
| 只有 category、没有 lineage 的旧记录 | 保留为 descriptive violation/trajectory，不进入 C2 evidence-ready 结果。 |
| oracle/full-context 在同 lineage 也发生 drift | 标记 control contamination；不得声称该类别是 memory-specific effect。 |
| 类别不 eligible | 不进入该类别分母。 |
| 没有后续 eligible checkpoint | 可以记录 adherence/violation，但不进入 longitudinal survival risk set。 |
| revoke 后 reopen | 按 validity window 重放；reopen 后正确使用不算 stale state。 |
| matched triplet 单一终点 | 报告 violation excess；纵向 onset 只来自具有多个 checkpoint 的 trajectory release。 |

### 2.4 计算示例

某个 `stale_state` trajectory 有三个 distinct eligible checkpoints：

```
t=2: 使用当前 v1，V=0，建立 adherence anchor
t=8: v2 已生效但仍使用 v1，V=1，observed drift onset=8
t=12: fresh reminder 后使用 v2，V=0，recovered=true
```

若 t=2 已错误使用尚未生效的 v2，则该点是 `plan_deviation` violation；只有在之后先观察到正确 adherence、再发生新的 violation 时，才能得到 observed drift onset。

---

## 3. 检索质量（维度4）

检索质量衡量记忆系统返回相关项目的效果，与智能体查询质量解耦。

### 3.1 定义

两种测量模式，分别报告：

**内生检索**：智能体在 episode 期间发起的 `search()` 调用。
按每个探针步骤的已知相关 `fact_id` 集评分。
*弱智能体 + 强记忆* 将显示低内生 p@k，因为智能体
发出了糟糕的查询——但这是智能体的问题，不是记忆的问题。

**Oracle 检索**：一组固定的、与智能体无关的基准查询，带有
已知的相关记忆 ID，直接对 `adapter.search()` 发出。这隔离了
记忆系统的检索质量。

### 3.2 公式

对于单个查询 `q`，带有黄金相关集 `R_q`（大小 `|R_q|`）和返回的
top-k 结果 `S_{q,k}`：

```
precision@k(q) = |S_{q,k} ∩ R_q| / k
recall@k(q)    = |S_{q,k} ∩ R_q| / |R_q|
```

对于查询集合 `Q`：

```
mean_precision@k = (1/|Q|) × Σ_{q∈Q} precision@k(q)
mean_recall@k    = (1/|Q|) × Σ_{q∈Q} recall@k(q)
```

**context_relevance**：在探针步骤时有效（未撤回）的返回项目比例。
计算公式为 `|{i ∈ S_{q,k} : i.fact is current}| / k`。

默认 `k = 10`。每次运行可配置。

每种模式报告为三元组：`(p@k, recall@k, context_relevance)`。

### 3.3 边界情况策略

| 情况 | 策略 |
|------|------|
| **空黄金集**（`|R_q| = 0`） | recall@k = N/A；precision@k 始终为 0。标记为非检索探针。 |
| **空结果集**（`|S| = 0`） | p@k = 0，recall@k = 0。 |
| **结果少于 k 个** | p@k 的分母仍为 `k`（*请求的* k），而非返回数量。这惩罚返回太少结果的系统。 |
| **重复结果** | 在交集前去重（每个 `memory_id` 计数一次）。 |
| **无 oracle 探针定义** | Oracle 检索 = N/A；仅内生。 |
| **Episode 中无检索调用** | 两种模式 = N/A。表示智能体从未使用记忆搜索（用于利用分析）。 |

### 3.4 计算示例

Oracle 探针的黄金相关集：`R = {m1, m2, m3, m4}`，`|R| = 4`。
系统返回 top-10：`S = {m1, m5, m2, m6, m7, m8, m9, m10, m11, m12}`。
`k = 10`。

```
|S ∩ R| = 2   (m1, m2)
precision@10 = 2 / 10 = 0.20
recall@10    = 2 / 4  = 0.50
```

如果 `m8` 在此探针步骤被撤回，`context_relevance = 9/10 = 0.90`
（10 个结果中 1 个无效）。

**内生 vs oracle 分割示例**：

| 模式 | 系统 | p@10 | recall@10 | 解释 |
|------|------|------|-----------|------|
| 内生 | 弱智能体 + FakePerfect | 0.15 | 0.12 | 智能体查询不佳 |
| Oracle | FakePerfect（同次运行） | 0.98 | 0.95 | 记忆系统检索完美 |

该分割确认记忆系统很强但智能体是瓶颈——
防止错误结论"记忆系统失效"。

---

## 4. 任务性能与目标导向利用（维度2）

### 4.1 定义

**任务分数**（`task_score`）：episode 探针中正确回答的比例，
由任务族 `Checker` 聚合（见 `spec/04-datasets.md`）。对于开放式（程序化不可检查）的探针，稀疏判定器提供分数；
判定器贡献单独报告并有上限。

```
task_score(c, e, s) = ( Σ deterministic_probe_scores + Σ judge_probe_scores ) / |probes|
```

`task_score ∈ [0, 1]`。当所有探针都有已知黄金答案时，`max_achievable(e) = 1.0`；对于仅有判定器探针的 episode，可能更低（按 episode 声明）。

**利用率**（`utilization_rate`）：在正确答案需要早期会话中获得的事实的探针（跨会话依赖）中，智能体正确回答的比例。所需事实可在当前会话上下文中获得的探针被排除——它们不测试记忆。

```
utilization_rate(c, e, s) =
    |{p ∈ cross_session_probes : p.correct}| / |cross_session_probes|
```

其中 `cross_session_probes` 是黄金答案依赖于 ≥1 个来自探针所在会话之前的会话的事实的探针子集。

**随时间改进**：可选的每 episode 趋势分析——随着更多记忆积累，智能体的任务分数是否跨会话增加？报告为补充图表，非核心数值。

### 4.2 边界情况策略

| 情况 | 策略 |
|------|------|
| **无跨会话探针** | utilization_rate = N/A。所有探针均可从上下文回答。 |
| **仅判定器探针** | 判定器提供 `[0,1]` 分数 + 理由。判定器分数被标记并在独立的 `judge_contribution` 字段中报告。任何复合中的判定器权重上限为 20%。 |
| **超时 / 崩溃** | 探针答案 = 0（任务失败）。运行被包含，不丢弃。 |
| **智能体答案为空** | 评分为不正确（0）。不视为缺失数据。 |
| **探针答案格式不匹配** | 如果智能体答案无法被检查器解析，探针评分为 0 并记录为 `format_error`。Episode 继续。 |

### 4.3 计算示例

6 个探针、3 个会话的 episode：

| 探针 | 会话 | 跨会话？ | 智能体正确？ | 分数 |
|------|------|---------|------------|------|
| p1 | s1 | 否（事实在上下文中） | 是 | 1.0 |
| p2 | s2 | 是（需要 s1 的事实） | 是 | 1.0 |
| p3 | s2 | 否 | 否 | 0.0 |
| p4 | s3 | 是（需要 s1、s2 的事实） | 否 | 0.0 |
| p5 | s3 | 是（需要 s1 的事实） | 是 | 1.0 |
| p6 | s3 | 否（仅判定器） | 判定器：0.7 | 0.7 |

```
task_score = (1.0 + 1.0 + 0.0 + 0.0 + 1.0 + 0.7) / 6 = 3.7 / 6 = 0.617

跨会话探针：p2, p4, p5 → 3 个探针，2 个正确
utilization_rate = 2 / 3 = 0.667

判定器贡献：0.7 / 3.7 = 0.189（18.9%，在 20% 上限内）
```

同一 episode 上的无记忆对照组将 p2、p4、p5 答错（事实不在上下文中）→ utilization_rate = 0/3 = 0.0，确认这些探针确实需要记忆。

---

## 5. 记分卡 — v1 聚合

### 5.1 原始 9 维聚合（v1 已废弃）

原始设计提出了 9 维加权聚合：

```
LHMSB = 0.25×G + 0.20×M + 0.15×D + 0.10×R + 0.10×T + 0.05×R2 + 0.05×E + 0.05×S + 0.05×A
```

此公式在 v1 中**已废弃**。v1 仅深入实现 4 个维度；原始聚合假设所有 9 个维度都被评分，在有 5 个维度未评分时会产生误导性数字。9 维公式保留为文档记录，作为未来版本的延迟设计目标。

### 5.2 v1 记分卡

v1 记分卡报告 4 个独立指标，加上跨维度的 Memory ROI 核心。没有从这些指标中计算出单一的聚合数值——每个指标都带有自己的 CI 报告，并通过 Memory ROI 核心进行上下文定位。

**记分卡表（每系统、每族、每赛道）**：

| 指标 | 类型 | 单位 | 报告形式 |
|------|------|------|---------|
| **Memory ROI** | 核心（跨维度） | 每 10³ token 当量的增益 | Bootstrap CI，增益下限，帕累托，`undefined-lowcost` 标记 |
| **任务分数** | 维度2 | [0, 1] | Bootstrap CI，判定器贡献 % |
| **利用率** | 维度2 | [0, 1] | 以跨会话探针为条件 |
| **漂移指数** | 维度3 | [0, 1] | 每类分解，判定器比例 |
| **检索 p@10** | 维度4 | [0, 1] | 内生 + oracle 模式 |
| **检索 recall@10** | 维度4 | [0, 1] | 内生 + oracle 模式 |
| **成本向量** | 维度7（支撑） | token 当量 + 原始字段 | 完整向量分解，不折叠 |

**原生赛道**是主要排行榜。**受控赛道**（跨系统固定内部模型）单独报告，绝不混合。敏感性条件（`fake_perfect`、`fake_bad`）在独立的校准章节中报告，不进入排行榜。

**帕累托前沿**：系统绘制在（mean_cost, mean_gain）坐标轴上，帕累托最优前沿被高亮显示。仅凭 ROI 标量值在极低成本下实现极小增益的系统不能"获胜"，如果被具有更大增益和可接受成本的系统支配的话。

### 5.3 可扩展性

维度1（记忆演进）、维度5（时间推理）、维度6（鲁棒性）、维度8（可扩展性）和维度9（抽象）被记录为扩展点。当未来版本实现新维度时，其指标作为独立行添加到记分卡中；一旦所有维度被评分，9 维聚合可以重新启用。v1 记分卡设计故意是扁平的（无加权和），以避免评分的和未评分的维度聚合造成扭曲。
