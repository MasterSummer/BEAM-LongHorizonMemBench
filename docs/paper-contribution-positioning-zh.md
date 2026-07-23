# BEAM 论文贡献定位：识别 Long-horizon Task-State Control

## 1. 中心科学问题

BEAM 用持续目标、有效步骤、因果依赖和 context reset 操作化
`long-horizon`。本文研究的问题是：

> 当 Agent 围绕一个持续总体目标，在有边界的任务环境中经历数百至数千个
> 相互依赖的有效步骤和多次 context reset 后，memory system 能否维持当前、
> 有效且具有正确 authority/scope 的任务状态，并在 workspace 之外继续控制
> 正确的下一步行为？如果行为失败，故障最早发生在写入、检索、暴露，还是
> 已经看到但未形成正确行为？

本文的观测对象是 **memory-supported delayed task-state control under
competing persistent channels**。`workspace` 与 task memory 都能跨 session
持久化，但二者是不同的
信息通道：workspace 保存文件、日志、测试结果和当前产物；memory system
自行选择写入、更新、删除和检索任务状态。论文要估计的是 memory 相对于
workspace 的边际行为价值，而不是把二者共同提供的信息归功于 memory。

第一项贡献应表述为：**一个用于识别 long-horizon task-state control 的反事实
诊断 benchmark**。
对决策 $d$，其最小观测模型是：

```text
latent task history H_d ──> native store M_d ──> retrieval R_d ──> exposure V_d
          │                                                        │
          └──────────────── workspace W_d ─────────────────────────┤
                                                                   ▼
                                                    action A_d ──> checker Y_d
```

BEAM 固定终端的 current-state contract、workspace 语义、action catalog 和
checker，再改变产生该终端的历史或 memory channel。该设计估计同一下一步
决策中，memory 在 workspace 之外维持 authoritative state 行为控制的程度，
并定位控制损失最早出现的环节。

## 2. Long-horizon 的定义与当前实验层级

Long-horizon task 满足五个条件：

1. **持续目标**：全部步骤服务于同一个有完成条件的总体目标；
2. **有界环境**：Agent 在同一个任务环境及其工具、文件和状态空间中行动；
3. **长决策跨度**：存在数百至数千个有效 Agent/环境步骤，而不是文本填充；
4. **因果依赖链**：早期决策、约束和中间结果会改变后续可执行动作及其有效性；
5. **任务级持久状态**：context reset 后，Agent 必须依靠 workspace、task
   memory 或完整历史恢复进度、决策、结果和约束。

其中“有效步骤”必须满足反填充契约：每个计数步骤产生唯一、可命名的语义任务
效果，消费其声明前驱的效果，并且位于至少一个被评分 continuation decision 的
因果祖先集合中。只有 effect digest、重复 observation、未被后续步骤消费的尾部
工作，均不能进入 ≥200 的 long-horizon 判定。报告同时给出 episode 总步骤和
每个被评分决策的 causal-ancestor span，门槛以后者为准。

任务长度与 policy 评测密度必须分开报告。BEAM 的当前 matched/horizon/
longitudinal
release 是 `replay_backed_critical_decision`：前缀具有可重放的有效步骤 DAG、
多次 handoff 和状态演化，memory backend 按 session 写入，但被比较的 policy
只在预注册的关键 continuation decision 上作答。因此当前结果支持：

> 在审计过的长程因果前缀之后，memory system 对关键任务状态决策的支持能力。

当前结果不支持：

> 被测 policy 在线执行了完整的数百至数千步闭环 Agent rollout。

报告必须输出 `interaction_mode`、有效步骤数、policy-evaluated 步骤数、
policy-conditioned 后续步骤数，以及是否存在
`policy action → later state/workspace → later policy decision` 依赖。只有达到
`online_long_horizon_agent_execution` gate 的新 release 才能使用强版在线
闭环表述。论文据此把 claim 限定在可识别的 long-horizon memory 机制。

## 3. 三项贡献是一套识别设计

### C1. Counterfactual identification of workspace-adjusted long-horizon state control

BEAM 提出 state-first、可冻结、可重放的 long-horizon memory-system benchmark。
gold 由某一 checkpoint 的当前 authoritative state closure 及其对应的可执行
continuation 构成。状态具有版本、有效时间、scope、authority 和依赖，
可发生 replace、revoke、invalidate、reopen 和 priority/scope change。

SCEU 的 closure 必须包含所有会区分候选 action 有效性的**当前**状态；缺少其中
任何一项，任务会在 API 调用前失败。replace/revoke precedence、authority
precedence 与 scope 不得越界泛化属于所有 condition 共享的稳定任务规则；
oracle 只额外获得最小当前状态事实，而不能获得 memory condition 看不到的
判定语义。这样 control 失败才可能归因于信息通道，而不是 evaluator contract
漏项或 condition-specific prompt。

实验在同一个终端决策上固定 current state、workspace 语义、action catalog、
opaque option mapping、正确 action 和 programmatic checker，仅改变上游历史：

- `static`：所需状态一直稳定；
- `evolution`：旧状态被更新、撤销或失效；
- `hierarchical_conflict`：局部收益与全局目标、scope 或高 authority 约束冲突。

同时设置以下持久通道 controls：

- `workspace_only`：memory 的 lower-bound control；
- `full_context`：完整历史可得性 control；
- `oracle_current_state`：当前状态可解性 upper bound；
- memory conditions：Flat retrieval、Mem0、A-MEM、MemOS。

主要估计量采用 workspace-adjusted matched penalty。令

```text
G(m, h) = Y(memory=m, history=h) - Y(workspace_only, history=h)
```

则 state-evolution 的主要估计量为：

```text
G(m, evolution) - G(m, static)
```

hierarchical conflict 同理。该 difference-in-differences 回答的是“历史需要
更新或 authority resolution 时，memory channel 在 workspace 之外额外丢失了
多少行为控制”，而不是“更难的问题准确率是否更低”。

Long-horizon specificity 由同决策 horizon-dose panel 检验。short、medium、
long 只改变有效前缀步骤、dependency depth 和 handoff 的联合剂量，终端任务
保持不变。主要 diagnostic 是 workspace-adjusted evolution/conflict penalty
从 short 到 long 的增量。若该增量为零，论文仍可报告状态维护与故障归因结果，
但不得宣称故障被 horizon 特异性放大。

**C1 可证伪条件**：oracle 无法解决任务；full context 同样失败；workspace-
only 已能恢复全部状态；matched terminal contract 不一致；固定 action/option
可以解决多数样本；或 long-dose penalty 不高于 short control。

### C2. Adherence-anchored, goal-relative behavioral drift

BEAM 不把任意错误或输出变化称为 drift。drift 必须相对于 episode 当前有效的
总体目标、计划和约束定义，并由 programmatic checker 判定：

- `constraint_loss`：持续约束逐渐失去行为影响；
- `plan_deviation`：当前计划偏离有效的总体目标，包括过早采用未来计划；
- `stale_state`：已撤销或替代的旧状态重新控制行为；
- `local_over_global`：局部子目标或局部收益错误覆盖全局目标/高 authority 约束。

单个 checkpoint 的错误只是 `drift-compatible violation`。只有在更早的独立、
同一 state lineage、同类别 eligible checkpoint 已观察到 adherence，之后该
lineage 才出现 violation，才能确定 longitudinal drift onset。遵守约束 C1 不能
为之后偏离计划 P2 提供 anchor；不同版本若属于同一演化 lineage，则由 evaluator
state graph 显式关联。每个 SCEU/category 只能有一个 focal lineage；若同一类别
涉及多个 state，生成器必须拆成多个 SCEU，不能把一个 category flag 广播到多个
lineage。主要输出包括：

- category-eligible violation rate；
- adherence-anchored first-drift handoff；
- drift-free survival；
- violation persistence；
- valid update / fresh reminder 后的 recovery。

episode 是统计单位，checkpoint/SCEU 是 episode 内重复观测。这个定义使 drift
成为 long-horizon 特有的“原先被正确执行的状态约束是否随时间失去控制力”，
而不是静态错误率的别名。

**C2 可证伪条件**：没有同 lineage 的 prior-adherence anchor；checker 的正负例
不能校准；oracle-current-state 或 full-context control 产生同类 onset；或 drift
标签不能由当前 state validity 与 lineage 重建。

为使这一定义在数据层面可证伪，v0.13 使用独立 longitudinal release，而不从
v0.11 的单一终端点外推 onset。每个 16-session episode 有 13 个预注册决策；对
四类 drift 中的每一类，必须存在同 lineage 且严格有序的
`anchor < ordinary challenge < update/reminder recovery` 三点窗口。最终 recovery
checkpoint 专门补齐旧 12-decision 轨迹中 `stale_state` 与
`local_over_global` 缺少后续恢复观测的问题。删掉该 checkpoint 会在任何 writer
或 policy API 调用前使 C2 design audit 失败。这里通过的是**可识别性设计**，不是
系统已经发生 drift 或成功 recovery 的经验结论；后者仍须从 episode 内实际
adherence 轨迹和干净的 oracle/full-context controls 得到。

### C3. Same-decision causal memory-channel fault localization

BEAM 在完全相同的 SCEU 上重建：

```text
required current state
  → stored
  → backend-retrieved
  → model-visible
  → intervention evidence
  → executable behavior
```

从而区分四类表面上可能具有相同错误 action 的原因：

1. **没存好**：workspace-absent 的 required state 在可观察 store 中不存在；
2. **存了但没检索到**：required state 已存储但不在 native retrieval 中；
3. **检索到但模型没看到**：backend retrieval 中存在，但最终 prompt/exposure
   中不存在；
4. **模型看到了但没有形成正确行为**：required state 全部 visible，行为仍错。

第四类不能只依靠 prompt inspection 推断“没有使用”。BEAM 对指定 memory
object 做 repeat-stable leave-one-out/replacement intervention：

- intervention 不改变 action/checker：
  `visible_without_detected_unique_causal_effect`；这不排除 redundant 或
  compensated use；
- intervention 改变行为但原行为仍错：
  `visible_causally_influential_but_wrong`；
- probe 不完整或重复不稳定：`visible_use_evidence_incomplete`。

因此代码中的 `behaviorally_used` 是可观测干预得到的 **unique causal-effect
lower bound**，而不是模型自述或对内部推理的直接访问。无 action/checker
变化只能写成 `no detected unique causal effect`，不能写成 memory definitely
unused。没有 storage provenance 时输出
`storage_evidence_unavailable`，不能偷换成 storage failure。native/exact 与
inventory-inferred provenance 必须分轨报告。

进一步地，BEAM 在同一 policy、readout、episode、SCEU、checkpoint、required
state、selected action 和 correctness 下配对不同 memory conditions，计算
`outcome_equivalent_fault_profile_divergence`。正值说明相同 end-task outcome
会掩盖不同 repair target，这是 endpoint accuracy 无法给出的诊断信息。

**C3 可证伪条件**：stage 不是在同一决策上比较；native lifecycle 不可观察且
inventory diff 也不可归因；retrieved 与 visible 无法分离；干预不稳定；或没有
outcome-equivalent aligned pair。

### 3.1 贡献层级：不要把三个组件包装成三个“first”

论文应明确区分主贡献、纵向 construct 和诊断工具：

| 层级 | Long-horizon 中新增的科学对象 | 必要反事实/时间条件 | 主要估计量 | 单独不足以证明什么 |
|---|---|---|---|---|
| **Primary: C1** | 多次 reset 与状态演化后，task memory 在 workspace 之外仍能否控制同一个可执行决策 | 同 terminal SCEU 的 static/evolution/authority-conflict history；matched workspace/full-context/oracle；short/long causal dose | workspace-adjusted construct penalty；horizon amplification | 更长 transcript 上的 raw accuracy drop 不能识别 memory-channel 或 horizon effect |
| **Construct: C2** | 某一 authoritative state lineage 曾经控制行为、后来何时失去控制 | 同 lineage prior adherence、later violation、recovery window；oracle/full-context clean | onset、drift-free survival、persistence、recovery | 单点错误、不同 state 的前后错误或 control 同样失败都不是 memory-specific drift |
| **Instrument: C3** | 在 C1/C2 的同一关键决策上，控制损失最早出现在哪个可观察环节 | 同 SCEU 的 store/retrieval/exposure trace 与 registered intervention | conditional yields、earliest supported stage、unique-effect lower bound、fault-profile divergence | C3 本身不是“首次 lifecycle benchmark”，也不能证明模型内部绝对没有使用 memory |

因此论文的 novelty 不能写成三个松散功能的并集。最强表述是：C1 定义并识别
long-horizon state-control loss，C2 给出该 loss 的纵向事件语义，C3 使该事件可
定位、可修复。若 C1 的 matched controls 与 horizon dose 没有完成，C2/C3 仍可
作为 measurement contribution 和 post-hoc diagnosis，但不能反过来证明
long-horizon-specific empirical effect。

### 3.2 Long-horizon specificity 的三级证据

为避免把“任务发生在长轨迹中”误写成“效应由 horizon 导致”，结果必须分级：

1. **Task-level applicability**：关键 SCEU 位于审计过的长因果前缀之后，所需
   state 跨越多次 reset/handoff；当前完成实验最多支持这一层。
2. **Mechanism identification**：同 terminal decision 下，evolution/conflict
   相对 static 的 memory penalty 在扣除 workspace 后仍存在；需要 matched C1
   experiment 和 clean full-context/oracle gates。
3. **Horizon specificity**：上述 workspace-adjusted penalty 在 long causal dose
   相对 short control 增大；需要 v0.12 panel。只有这一层通过，才能写
   “the failure is amplified by horizon”。

这个分级允许 null result：如果第三层为零，BEAM 仍贡献 state-control 的评测
对象、C2 operationalization 与 C3 diagnosis，但论文不能靠 session 数把机制
称为 long-horizon-specific。

## 4. 论文的三条 Research Questions

### RQ1

在同一终端决策上，task memory 相对于 workspace-only 提供多少额外行为控制；
当历史需要 state evolution 或 authority resolution 时，该边际控制是否下降，
并是否随 joint horizon dose 放大？

主要指标：behavior gain beyond workspace、workspace-adjusted evolution/conflict
penalty、oracle-gap closure、short-to-long amplification，以及按
workspace recoverability、state age、handoff 和 dependency depth 的分层结果。

### RQ2

当前 authoritative state 对行为的控制何时开始丢失、是否持续，以及 update 或
reminder 能否恢复？

主要指标：eligible violation、adherence-anchored onset、drift-free survival、
persistence、recovery；四类 drift 分开报告。

### RQ3

对于同一个错误或正确 continuation，故障最早可以定位在存储、检索、暴露还是
利用阶段；相同 endpoint outcome 是否隐藏不同 memory repair target？

主要指标：write coverage/selectivity、current-state storage P/R/F1、stale
retention、update/delete responsiveness、stored→retrieved 与
retrieved→visible conditional yield、causal-use lower bound、earliest failure
stage 和 outcome-equivalent fault-profile divergence；其中 causal-use 指
repeat-stable intervention 识别出的 unique-effect lower bound。

## 5. BEAM 改变的测量对象与识别变量

如果只是迁移，设计会是“增加历史长度，然后问同一个事实问题”。BEAM 改变了
gold unit、干预变量和因果估计量：

| 维度 | 长版事实 MemoryBench | BEAM |
|---|---|---|
| Gold | 历史事实/答案 | 当前 authoritative state closure + executable action |
| 时间语义 | 事实仍然成立 | 版本、撤销、失效、scope、authority 会变化 |
| Long-horizon 操作化 | prompt/session 更长 | 有效步骤 DAG、handoff、state age、同决策 horizon dose |
| 外部持久状态 | 通常并入 context | workspace 是独立竞争通道和 matched control |
| 机制识别 | 长样本与短样本不是同一任务 | 同一终端决策的 static/evolution/conflict 反事实历史 |
| 输出 | factual answer | programmatically checked continuation |
| Drift | 某时点答错 | prior adherence 后的 goal-relative onset/persistence/recovery |
| 故障分析 | recall/retrieval error | 同决策 stored→retrieved→visible→causal use→behavior |
| 因果证据 | 检索文本与答案相关 | memory-object intervention + outcome-equivalent pairing |

论文的中心主张可写成：

> BEAM holds the executable decision fixed and counterfactually changes the
> state history and persistent information channel, thereby identifying how
> long-horizon state evolution and authority conflict degrade memory-supported
> behavioral control beyond workspace, and where that loss first appears in
> the native memory-to-action pipeline.

### 5.1 与最新相邻工作的重叠压力测试

截至当前，不能再把“多 session 行动”“状态演化”“retrieved-but-unused”中的任意
一个单点当作独占创新。论文必须主动承认这些重叠，再说明 BEAM 改变的是识别
设计：

| 相邻工作已经做到 | BEAM 不应声称 | BEAM 仍需证明的增量 |
|---|---|---|
| [MemoryArena](https://arxiv.org/abs/2602.16313) 在相互依赖的多 session agentic tasks 中把 memory acquisition 与后续行动耦合 | first multi-session/action-oriented memory benchmark | 固定同一终端决策，显式扣除 workspace 通道，并在该决策内定位 memory channel 故障 |
| [AMA-Bench](https://arxiv.org/abs/2602.22769) 使用 agent trajectory、latent state/causality，并以 needle ablation 区分 construction 与 retrieval 损失 | first state-first/causal-trajectory benchmark；first construction-vs-retrieval diagnosis | 对 native store、backend retrieval、最终 exposure、干预证据与 executable behavior 做同决策对齐；不把 QA accuracy 当作最终任务状态控制 |
| [Mem2ActBench](https://arxiv.org/abs/2601.19935) 评估 memory-grounded tool action，并报告 retrieval miss 与 retrieved-but-unused | first memory-to-action benchmark；first retrieval/use decomposition | 不把 retrieved+wrong 直接标成 unused；用 targeted intervention 给出 unique causal-effect lower bound，并额外区分 stored、retrieved 与 model-visible |
| [MemConflict](https://arxiv.org/abs/2605.20926) 与 [Memora](https://arxiv.org/abs/2604.20006) 评估动态冲突、过时 memory 和 retrieval/utilization gap | first update/conflict/stale-memory benchmark | 用 version、scope、authority 和 dependency closure 定义当前可执行状态；以 prior adherence 后的 onset/persistence/recovery 定义 drift，而非某次冲突 QA 错误 |
| WorldMemArena、AgingBench 等工作覆盖 lifecycle 或阶段诊断 | first lifecycle/failure-stage benchmark | 把 workspace control、反事实 matched history、native provenance、exposure 和 causal intervention 绑定到同一个 executable decision，并报告证据不足类别 |

BEAM 的 novelty 来自以下联合识别设计：

```text
same executable decision
+ counterfactual state history
+ workspace as a competing persistent channel
+ adherence-anchored longitudinal consequence
+ native-trace and intervention-based fault localization
= identifiable integrity of the long-horizon state-control channel
```

其中任何一项缺失，都必须缩小 claim：

- 没有 matched terminal decision，只能报告不同题型的 performance gap；
- 没有 workspace-only 对照，不能把共享持久信息的收益归因于 task memory；
- 没有 prior-adherence anchor，只能报告 violation，不能报告 drift onset；
- 没有 lifecycle provenance，不能报告 storage failure；
- 没有 retrieved/visible trace，不能区分 retrieval 与 exposure；
- 没有稳定的 targeted intervention，不能把 visible+wrong 写成“看到了但没用”；
- 没有 same-decision short/long panel，不能说观察到的机制是 horizon-specific。

### 5.2 论文主线

三个 contribution 应被写成一条因果故事：

1. **识别对象**：C1 建立 memory 相对于 workspace 的 delayed state-control
   estimand；
2. **纵向后果**：C2 衡量该控制何时从原先 adherence 演化为 goal-relative
   drift；
3. **可修复诊断**：C3 在同一个决策上找出控制链最早断在 write、retrieve、
   expose，还是 behavior formation。

这样，drift 不是额外加的一组错误标签，fault localization 也不是一般 telemetry；
二者分别回答 state-control channel 的“何时失效”和“在哪里失效”。

### 5.3 三条不可约的贡献表述

第一条需要写明评测单元与识别估计量。建议固定为以下三条：

1. **Decision-matched, workspace-adjusted state-control benchmark.** BEAM 以
   State-Conditioned Execution Unit（SCEU）为评测单元：一个 checkpoint 的
   当前权威状态闭包、workspace、可执行决策、动作集合与
   programmatic checker。静态、演化和 authority-conflict 历史在同一 SCEU
   终端上配对，并以 workspace-only 扣除外部持久状态已经提供的控制。新意在
   评测对象和识别估计量，不在 transcript 长度。
2. **Adherence-anchored loss of long-horizon control.** BEAM 将 drift 定义为
   Agent 已经在更早 checkpoint 证明会遵守的总体目标、持续约束或当前计划，
   在后续长因果链中失去行为影响。anchor 与 failure 必须属于同一 state
   lineage；onset、persistence 与 recovery 是定义的一部分；单点答错不是
   drift。这使 drift 成为 state control 的纵向失效，而非额外错误标签。
3. **Same-decision causal fault localization.** BEAM 将 native store provenance、
   backend retrieval、最终 model exposure、memory-object intervention 和
   executable behavior 对齐到同一 SCEU。贡献不是首先提出某一个 failure stage，
   而是使“没存、没取、没看到、看到但未形成正确行为”成为互斥、证据分级、
   可追溯到 repair target 的同决策诊断。

这三条不能拆成三个互不相关的 feature。C1 给出“什么控制丢了”，C2 给出“何时
开始丢”，C3 给出“在哪一段丢”。论文中的 benchmark contribution 应被称为
**an identification and diagnosis framework for memory-supported state
control**，而不是 “a long-horizon version of a memory benchmark”。

### 5.4 审稿人替代解释测试

论文主张只有在以下替代解释被实验排除后才成立：

| 可能的审稿意见 | BEAM 必须给出的反证 |
|---|---|
| 只是 context 更长 | 同一 terminal SCEU 的 short/medium/long causal-dose panel；长度不能靠不同题目难度产生 |
| workspace 本身已经含有答案 | 同一 SCEU 的 explicit/derivable/absent recoverability 与 workspace-only subtraction |
| 只是测试检索 | native store、backend retrieval 与 final model-visible IDs 分开记录，并接到 checked action |
| “用了”只是根据答案猜的 | repeat-stable targeted intervention；未探测和无检测效应分别保留 |
| drift 只是普通错误 | 同 state-lineage、同类别 prior-adherence anchor，随后 onset，并报告 persistence/recovery；oracle/full-context 必须干净 |
| 状态演化只是又一种题型 | static/evolution/conflict 共享 terminal state、workspace 语义、动作、opaque option map 与 checker |
| 最终分数已经足够 | 在同动作、同正确性的 aligned pairs 中证明 fault profile 可以不同 |

任何一行缺少证据，都应缩小对应 claim，而不是用更多 episode 掩盖设计缺口。

## 6. 投稿时允许与禁止的表述

允许：

- a benchmark of memory-supported critical decisions after audited
  long-horizon task prefixes；
- counterfactually identified, workspace-adjusted state-control degradation；
- goal-relative, adherence-anchored longitudinal drift；
- earliest supported failure stage and a detected unique-causal-effect lower
  bound；
- same-decision horizon amplification under a joint transition/handoff dose。

禁止：

- the first memory lifecycle benchmark；
- the first action-oriented memory benchmark；
- the first behavioral drift benchmark；
- the tested agent executed 257 online decisions（当前 release 不成立）；
- visible memory was definitely unused（只能说 no detected unique causal
  effect；redundant/compensated use 未被排除）；
- storage failure when provenance is unavailable；
- pure handoff effect（当前 panel 同时改变 transitions、depth 和 handoffs）；
- backend ranking when oracle/full-context/readiness gates fail。

## 7. 结果段的判断顺序

每个实验报告和论文结果必须按以下顺序解释：

1. terminal solvability：oracle 是否通过；
2. history availability：full context 是否通过；
3. workspace insufficiency：workspace-only 是否留下 memory-reliant gap；
4. shortcut resistance：always-action/always-option 是否低于阈值；
5. matched mechanism effect：workspace-adjusted evolution/conflict penalty；
6. horizon specificity：long 相对 short 的额外 penalty；
7. longitudinal consequence：drift onset、persistence、recovery；
8. repair target：storage/retrieval/exposure/utilization attribution；
9. uncertainty and unit：episode、counterfactual group 或 horizon panel，禁止将
   dependent physical members 当成独立样本。

只有前置 gate 通过，后续解释才允许升级。这样论文的贡献由可证伪的识别链条
支撑，而不是由 benchmark 名称中的 `long-horizon` 支撑。

## 8. 已完成实验与新 contribution contract 的时间边界

分析代码升级不能改变实验当时冻结的 claim。对已经完成的 server run：

1. 使用 run manifest 中的原始 clean commit 生成或验证 canonical report；
2. 保留原始 `experiment_design_audit.json` 及其 hash，禁止用新 schema 覆盖；
3. 新增的 interaction-tier、fault-profile 或 contribution audit 若在实验完成后
   定义，必须标记为 `post_hoc_scope_audit` 或 exploratory analysis；
4. 如果原始 pre-call contract 已经冻结相同的 matched/horizon estimand，仅报告
   代码发生变化，则可以从 immutable task results 和 prefix artifacts 做零 API
   重聚合；
5. 如果原始数据没有 static/evolution/conflict matched members、horizon-dose
   members、prior-adherence checkpoints 或同决策 intervention，则不能靠重聚合
   补出对应 contribution evidence，必须运行新的冻结 experiment；
6. canonical report 与 post-hoc report 使用不同目录、不同 aggregation commit
   和明确的 analysis-timing 标签，不能覆盖同一个结果目录。

新实验的 pre-call `analysis_contract` 必须逐项冻结三类 claim：C1 的
workspace-adjusted matched/horizon estimand，C2 在该 release 上允许的 endpoint
或 lineage-longitudinal scope，以及 C3 的 trace、provenance 与
repeat-stable neutral-replacement intervention。只冻结 C1 而在看过结果后补写
C2/C3，不属于 `pre_specified` evidence。

### 8.1 2026-07-22 已完成 GPT-only calibration 能支持什么

已完成的 clean source run 使用 5 个 episode、600 个 SCEU 结果和 741 个
memory lifecycle events。它通过了 artifact/hash validation，但 measurement
readiness 未通过：sham action-flip 上界为 0.0563（阈值 0.05），且 oracle 在
`local-only`、`global-local-conflict` 及两个 scenario strata 上未达到预设门槛。
因此它不能作为 confirmatory backend ranking，也不能证明 C1 的 matched
long-horizon effect。

进一步审计表明，这两个 control failure 至少部分来自 v0.8 冻结任务契约本身：
local conflict 的旧 SCEU 没有把当前 P2/U1 完整纳入 action-relevant closure，
且 authority/scope precedence 当时不是所有 condition 共享的稳定公开规则。
因此不能把这些 control 错误解释成 memory-system drift。旧 v0.8 数据与报告
保持原样，只作为带此限制的 calibration；修复仅进入新 release，并由
`current_action_state_contract_complete` 与 shared-governance prompt 测试在 API
调用前 fail closed。

原报告和其精确冻结 dataset 已做只读 hash 对齐。零 API、
`post_hoc_exploratory` 的 C2+C3 重聚合得到：

- 由 evaluator state graph 重建了 200 条 state-lineage trajectory，其中 194
  条具有 prior-adherence 后续窗口；34 条 category-lineage trajectory 出现
  observed drift，对应去重后的 20 个 onset decisions；
- 18 条 trajectory 具备 valid-update/fresh-reminder recovery 机会，其中 14 条
  恢复；
- memory conditions 出现 22 条 observed-drift trajectories，但 oracle 与
  full-context 各有 2 条，并都涉及 `constraint_loss` / `local_over_global`；因此
  四类 drift 的整体 memory-specific effect 不成立；
- `plan_deviation` 与 `stale_state` 在本次 post-hoc 描述中没有对应的
  oracle/full-context drift，可作为后续 confirmatory 设计的候选，不可在 5 个
  episode 上升级为总体效应；

- 600 个已观察决策中，154 个是 workspace-absent 的 memory-reliant decisions；
- earliest supported stage 包括 13 个 storage failures、10 个 retrieval
  failures、26 个 exposure failures 和 24 个 utilization failures；
- 166 对 outcome-equivalent aligned memory-condition pairs 中，56 对具有不同
  fault profile，描述性比率为 0.337；
- 41 对选择同一错误 action 的 pairs 中，27 对具有不同 fault profile。

这组结果直接支持两个有限但重要的描述：**当前冻结轨迹中存在可由同一
state-lineage prior adherence 锚定的 longitudinal transitions；在相同 checked
outcome 下，endpoint accuracy 的确会隐藏不同的 observed repair target。**
前者不能整体归因于 memory，因为两个强 control 在部分 drift 类别上也失败；
后者不能支持“总体 33.7% 的所有 long-horizon failures 都能如此分解”之类的
总体推断，因为 pairs 在 episode/SCEU 内相关、只有 5 个 episode，且两个
估计量都是实验完成后定义的。

对三项贡献的证据状态应写成：

| Contribution | 当前证据状态 | 当前允许的论文表述 |
|---|---|---|
| C1 state-control identification | design implemented；旧 run 只有 calibration signal | benchmark/estimand contribution；不得声称已识别 long-horizon amplification |
| C2 longitudinal drift | 200 条 lineage-backed post-hoc trajectories；存在 onset/persistence/recovery signal，但 oracle/full-context 在两类 drift 上污染 | 可报告 state-lineage-anchored descriptive trajectories；不得声称整体 memory-specific drift effect |
| C3 same-decision localization | trace、provenance、intervention 和 frozen gold 可重聚合；有非零 outcome-equivalent divergence | post-hoc descriptive evidence that identical outcomes hide different observed repair targets |

对应只读产物位于 source run 的 sibling directories；canonical report 保持不变。

v0.13 的 longitudinal generator、freeze/verify/regen、pre-call C2/C3 design
audit 与系统 planner 已在本地实现并通过；它不改变上表所列“已完成实验”的证据
等级，因为尚未产生新的 server policy/backend result。正式 C2 结果应来自独立的
5-episode calibration 与 50-episode confirmatory longitudinal splits，并以 episode
为统计单位；同一 episode 的 13 个 checkpoint 只能作为重复观测。

## 9. 推荐标题与摘要首句

标题优先强调诊断对象，而不是泛称 long-horizon benchmark：

> **BEAM: Counterfactual Diagnosis of Memory-Supported State Control in
> Long-Horizon Tasks**

或：

> **Beyond Recall: Diagnosing Memory as a State-Control Channel in
> Long-Horizon Agents**

摘要首句建议直接给出缺口：

> Existing benchmarks increasingly test memory in long and agentic
> trajectories, but an incorrect continuation still does not reveal whether
> the required task state was never stored, not retrieved, omitted from the
> model input, or visible yet behaviorally ineffective, while persistent
> workspace artifacts make memory's marginal contribution ambiguous.

因此，“实验已完成”只表示无需重新调用 policy 的前提可能已经满足，并不自动
表示所有后来提出的 claim 都已被实验前设计支持。论文中应分别标注：

- `design contribution`：数据结构、控制条件和可证伪 measurement contract；
- `calibration evidence`：链路可运行且产生非退化信号；
- `identified empirical effect`：matched controls、solvability/readiness gates
  和正确统计单位全部通过；
- `confirmatory evidence`：还需满足 disjoint split、预先冻结分析和样本量要求。
