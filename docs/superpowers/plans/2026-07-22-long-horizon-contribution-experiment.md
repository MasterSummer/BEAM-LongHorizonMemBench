# Long-Horizon Contribution and Experiment Completion Plan

## Objective

将 BEAM 的论文对象从“更长的 memory benchmark”明确改造成“持续任务中的状态维护与行为控制评测”，并让三项 contribution 都有独立、可审计、可证伪的代码和实验产物：

1. state-first、workspace-aware 的 long-horizon benchmark；
2. long-horizon behavioral drift；
3. 同一决策上的 `stored → retrieved → visible → intervention evidence →
   behavior` 故障归因。

这三点不是三个互不相干的 novelty claim。主 claim 是
“counterfactually identified delayed task-state control under competing
persistent channels”；drift 与 fault localization 是使该主 claim 可观察、
可定位、可证伪的两类 measurement protocol。不得声称首次提出 action-oriented
memory、memory lifecycle、behavioral decay 或 write/retrieval/use stage diagnosis。

## Evidence boundary

旧 v0.10 Software 数据是 16-session、12-opportunity 的受控关键决策轨迹；
v0.13 longitudinal release 是 16-session、13-opportunity，并增加最终同 lineage
recovery decision。v0.11--v0.13 的长前缀有可审计的 causal/effect-step DAG，但
被比较 policy 仍只执行 registered critical decisions。因此论文可以声称评测了
审计过的 long-horizon causal prefix 之后的 memory-supported decisions，不能声称
该 policy 在线逐步执行了数百至数千步。

进一步区分 task span 与 policy interaction。当前 v0.11/v0.12/v0.13 必须标记为
`replay_backed_critical_decision`；只有至少一个 policy decision 进入后续
state/workspace 并成为后续 policy decision 的祖先，才可标记
`sparse_closed_loop`；只有至少 200 个 causally linked policy-evaluated steps
才可标记 `online_long_horizon_agent_execution`。interaction tier 必须进入
task-span artifact、pre-call design audit、report summary 和 contribution
evidence，防止有效前缀步骤数被误写成在线 Agent 决策数。

当前单 episode server smoke 证明工程链路可运行，但 measurement readiness 仍为 false。它不能支持 backend ranking 或 drift 的总体实证结论。

## Contribution-to-evidence contract

Each total report must emit `contribution_evidence.json` and
`contribution_evidence.md`. The artifact maps C1--C3 to applicable claim scope,
required gates, estimands, evidence counts, source artifacts, and claim
boundaries. It is rebuilt during validation, and therefore cannot be made
`ready` by manually editing the report. `ready` denotes measurement-contract
completeness only, not a positive effect or confirmatory conclusion.

For an experiment that has already completed, generate a separate
`completed_report_audit.json/.md` with `audit-completed-report`. The audit must
preserve the source report tree hash, source analysis timing, failed gates, and
the strongest artifact level available for C1--C3. Missing pre-call contracts
remain `undeclared_legacy`; they must never default to `pre_specified`. The audit
output is a sibling of the canonical report and is always labelled
`post_hoc_scope_audit` or `post_hoc_exploratory`.

在任何 writer/policy API 调用前，`plan-systems` 还必须生成并绑定
`experiment_design_audit.json`。该 policy-free audit 检查 matched triplet、
gold action/opaque option、workspace recoverability、terminal archetype、drift
checker 正负例、memory-reliant decision、future-state exclusion 和 task-step
effect chain。每个 current action-relevant state 必须进入 SCEU required
closure；所有 condition 共享 replace/revoke、authority 和 scope 规则。
单 group 仅标为 `diagnostic_only`；至少三组且全部通过才标为
`ready_for_calibration`。

同一 pre-call artifact 必须分别冻结 C1--C3 的 analysis contract，而不是只冻结
C1 后把 C2/C3 在报告阶段补写为 `pre_specified`：C1 固定 workspace-adjusted
matched/horizon estimand；C2 固定本 release 只支持 endpoint violation 还是满足
lineage gate 后的 longitudinal onset/survival；C3 固定同一 SCEU 上的 trace 顺序、
exact/inferred provenance 双轨和 repeat-stable neutral-replacement intervention。
任何 contribution 缺少自己的 pre-call contract，都只能标记为 undeclared 或
post-hoc。

### C1. Counterfactually identified delayed task-state control benchmark

必须输出以下 evaluator-side construct，而不是仅以 token/session 数代表 horizon：

- handoff count；
- oldest required-state age；
- latest decision-event distance；
- dependency depth；
- relevant transition count；
- workspace recoverability composition；
- construct kind：static recall、state evolution、hierarchical conflict、fresh control。

主实验必须在相同 policy、action catalog 和 checker 下比较 workspace-only、full-context、oracle-current-state、Flat retrieval 和 memory systems。报告 behavior score、current-state resolution、conflict resolution、gain beyond workspace 和 oracle-gap closure。

对于 matched release，`full_context` 和 `oracle_current_state` 不是仅供画图的
上下界。每个 policy 必须在全部 counterfactual group 和 static/evolution/
hierarchical-conflict 三个 variant 上达到预注册正确率门槛。Oracle 失败表示
任务可解性未建立；full-context 失败表示完整历史解释仍是混杂；任一失败都
禁止将该 cell 的缺陷归因为 memory channel。

### C2. Behavioral drift

四类 canonical drift 为：

- `constraint_loss`；
- `plan_deviation`；
- `stale_state`；
- `local_over_global`。

所有 rate 使用 category-specific eligible denominator，并严格拆分两类结果：

- `drift-compatible violation`：单点 action 触发 canonical drift flag；
- `observed longitudinal drift`：较早 distinct eligible checkpoint 已在同一
  state lineage 上观察到 adherence，随后该 lineage 才出现 violation。

首次观测即错误不能被称为 drift onset。除 violation rate 外，输出按
handoff/age/construct 分层的 adherence-anchored drift curve、first-drift
session、persistence 和 reminder/update recovery。统计推断以 episode 为单位，
SCEU 与 state lineage 仅作为 episode 内重复测量。category-only legacy
trajectory 只能描述，不能通过 C2 gate；oracle-current-state 与 full-context
必须覆盖相同 lineage 且不出现 onset，才能声称 memory-specific drift。

### C3. Decision-aligned causal fault attribution

对同一个 SCEU 定义当前必需状态、memory-reliant 状态、stored、retrieved、visible、probed 和 causally used 集合。按最早失败阶段归因：

1. `storage_evidence_unavailable`（不能误报为 storage failure）；
2. `storage_failure`；
3. `retrieval_failure`；
4. `exposure_failure`；
5. `utilization_failure`；
6. `behavior_success_causal`；
7. `behavior_success_without_detected_unique_causal_effect`；
8. `behavior_success_unprobed`；
9. `not_memory_reliant` / `no_memory_channel`。

“看到了且具有可识别的独立行为影响”只由 repeat-stable、state-targeted
counterfactual intervention 判定；模型自述和单次 action flip 不构成
causal-use 证据。无 effect 不得写成 unused，因为 workspace 或其他 memory
object 可能提供冗余信息，policy 也可能补偿该删除。

新增 `outcome_equivalent_fault_profile_divergence`：仅在 policy、readout、
episode、SCEU、opportunity、checkpoint、required/current state 和最终 action
相同的 memory-condition pair 上，判断 earliest supported stage 或 utilization
subtype 是否不同。它用于证明相同 end-task outcome 会掩盖不同 repair target；
pair 是 dependent descriptive diagnostic，不新增统计样本，0 也不解释为等价。

## Implementation phases

### Phase 1. Long-horizon construct profiling

- 新增纯 evaluator-side construct profiler，不修改冻结 episode schema 和 plan hash。
- 从 `EpisodePlan + SCEU + ContinuationOpportunity` 推导 horizon/transition/dependency/workspace 指标。
- 将 profile 写入 normalized metric observations。
- 添加 construct/horizon scorecard 与单元测试。

### Phase 2. Same-decision fault localization

- 在 prefix inventory 上映射 stored state IDs。
- 将 retrieved、visible、probed、used memory IDs 映射到 evaluator state IDs。
- 只对 workspace-absent 的当前必需状态计算 primary memory-reliant funnel；derivable 状态作为 sensitivity track。
- 输出 conditional stage yields 和 first-failure distribution。
- 输出 `fault_profile_divergence.{json,md}`，并由 validator 从
  `decision_attribution.jsonl` 精确重建；C3 至少需要一个 outcome-equivalent
  aligned pair 才能 evidence-ready。
- exact 与 inferred storage provenance 继续分轨，不能合并为 exact 结论。

### Phase 3. Drift-over-horizon reporting

- 按 absolute handoff band 和 construct kind 汇总 behavior/drift。
- 分别输出 episode-level first-violation 与 adherence-anchored first-drift、
  persistence、survival、recovery records。
- 增加 workspace-vs-oracle paired drift differences。
- 增加报告 JSON/CSV/Markdown 和 artifact validation。

### Phase 4. Matched construct dataset release

状态：代码与本地数据闭环已完成，服务器机制实验尚未运行。v0.10 仍是主
ranking release；v0.11 是独立的 matched-mechanism release，两者不得合并。

- 在每个 counterfactual group 内固定 final-session decision、action catalog、
  gold action、opaque option mapping、checker-relevant terminal predicates、
  workspace shape 和 memory-object budget，并生成：
  - static-recall control；
  - ordered state-evolution condition；
  - hierarchical-conflict condition。
- 独立操纵 handoff distance、transition count、dependency depth 和 workspace recoverability。
- 保持 opaque options，禁止 future-state leakage。
- 升级 dataset release；重新 freeze、verify 和 regen-check。
- 每个 16-session member 至少记录 256 个 public prefix transitions；每步
  记录 execution mode、前驱、state/workspace reference、effect digest、唯一
  semantic effect 及其 consumed predecessor effects。加载时同时验证 digest
  DAG 与 semantic-effect DAG；≥200 门槛计算 target decision 的 effective
  causal ancestors，禁止用未消费 observation、episode 总长度或 post-decision
  step 充当 horizon。
- 在 group 之间轮换 `current-v2 offline`、`current-v1 offline` 与
  `authorized scoped cloud` 三种 terminal archetype，同时平衡 opaque gold
  option；3-group 及以上 release 强制执行 always-action ≤ 0.50 和
  always-option ≤ 0.40 gate。
- 输出 `matched_construct_contrasts.jsonl` 与
  `matched_construct_scorecard.{csv,md}`。相对 matched static 的 raw
  state-evolution penalty、hierarchical-conflict penalty、correctness penalty
  与 endpoint drift-compatible violation excess 均为 secondary description，
  不能替代 workspace-adjusted primary analysis。
- 同时输出以 workspace-only 为第二层对照的 difference-in-differences：
  `state_evolution_penalty_excess_over_workspace` 与
  `hierarchical_conflict_penalty_excess_over_workspace`。这是机制 claim 的
  primary estimand；raw construct penalty 只能描述难度，不能排除 workspace
  surface 随历史变化造成的混杂。
- 输出 `matched_construct_statistics.{json,md}`：先在 triplet 内计算差值，
  再以 counterfactual group 为统计单位做 bootstrap CI、paired sign-flip 和
  Holm correction；三个 physical members 永远不作为三个独立样本。
- 在任何 writer/policy call 前，将上述 primary/secondary estimands、
  workspace adjustment、effect direction、endpoint-only drift scope、统计
  单位与 multiplicity scope 写入 `experiment_design_audit.analysis_contract`
  并绑定 run identity。它是内部 hash-bound pre-specification，不冒充外部
  public preregistration；standard trajectory release 明确标为 N/A。

已实现并通过单元测试：三联组同 final checkpoint/request/action
catalog/gold action/opaque option mapping/continuation scope/terminal
condition/prefix shape/workspace shape；三个 archetype 的 gold 均通过隐藏
software checker；v0.11 generate、freeze、verify、regen-check；task-span、
shortcut-resistance 与 matched-outcome readiness gates。

### Phase 4.5. Same-decision horizon-dose diagnostic

状态：v0.12 生成、freeze、verify、regen-check、planner contract、report、
statistics 与 artifact validation 均已实现并通过离线 vertical slice；尚未在
正式 policy/backend matrix 上运行，因此仍不属于已完成的实证结果。

- 对同一 semantic seed 和 terminal archetype 生成 4/8/16-session panel；
- 每个 horizon 含 static/evolution/hierarchical-conflict triplet，共 9 个
  physical members，但统计单位是一个 `horizon_panel`；
- 固定 terminal request、current state semantics、workspace semantics、actions、
  gold、opaque option mapping、scope、package 与 hidden checker；
- 只允许 effective transitions、dependency depth、session handoffs 和对应事件
  时间发生变化；
- 结构总步数为 65/129/257，terminal decision causal-ancestor span 为
  64/128/256，handoff 为 3/7/15；只有 long member 达到当前 ≥200-step
  long-horizon floor；
- primary diagnostic：
  `state_evolution_horizon_amplification_excess_over_workspace` 与
  `hierarchical_conflict_horizon_amplification_excess_over_workspace`；
- positive 表示 memory condition 的 construct penalty 从 short 到 long 的增幅
  大于 workspace-only 的对应增幅；null/negative 结果必须保留，并限制
  “long-horizon-specific” claim；
- 该 panel 同时改变 transition count 和 handoff count，只识别 joint horizon
  dose。拆分两者需未来 2×2 factorial，不得从本 panel 推断 pure handoff effect。
- dataset release：`software-matched-horizon-panels-v0.12.0`；配置：
  `configs/experiments/systems_controlled_gpt_only_horizon_v012.yaml`；
- planner 将 9 个 physical members 锁成 1 个 `horizon_panel`，任何截断直接
  fail before API；报告禁用 episode-level 与 within-dose-triplet inference；
- calibration 采用 3 panels / 27 physical episodes / 189 evaluation tasks；
  `horizon_panel_statistics.json` 才是该 release 的 inferential artifact。

### Phase 4.6. Longitudinal trajectory release

状态：v0.13 generator、freeze/verify/regen、pre-call design audit、planner 和本地
反例测试已经实现；正式 server policy/backend matrix 尚未运行。

- 每个 episode 保留 16 sessions、256 个 public causal-prefix steps，并在 13 个
  registered critical decisions 上评分；总 effective steps 为 269；
- 对 `constraint_loss`、`plan_deviation`、`stale_state` 和
  `local_over_global` 分别冻结一个同 lineage 的
  `anchor < ordinary challenge < update/reminder recovery` 窗口；
- 最终 `opp-final-lineage-recovery` 补齐 stale-state 与 local-over-global 的后续
  recovery 观测；删掉它必须使 `c2_longitudinal_recovery_design` fail；
- 每个 memory-reliant SCEU 必须声明至少一个 current、action-relevant 的
  intervention target；删掉 target 必须使 `c3_intervention_target_contract` fail；
- gold action 分布为 7/3/3，best always-action accuracy 为 7/13，低于 0.60 gate；
- planner 记录 `primary_analysis_unit=episode`，13 个 SCEU 不能增加统计单位；
- dataset release：`software-longitudinal-trajectories-v0.13.0`；配置：
  `configs/experiments/systems_controlled_gpt_only_longitudinal_v013.yaml`；
- calibration/confirmatory 分别至少为 5/50 个独立 episode。

### Phase 5. Calibration and confirmatory experiment

已完成旧 run 的证据回收不修改下述 future experiment 顺序。2026-07-22 的
v0.8.0 GPT-only calibration 已通过 source hash audit，并与其 exact frozen
dataset 对齐。新增的 `reanalyze-completed-report` 只读路径在零 API 调用下输出
600 条 decision attribution；其中 154 条 memory-reliant rows 覆盖 storage、
retrieval、exposure 与 utilization 四种 earliest stage。166 对
outcome-equivalent pairs 中有 56 对 fault profile 不同。它还重建了 200 条
state-lineage trajectories：194 条 drift-evaluable、34 条 observed drift、20
个去重 onset decisions，18 条 recovery-evaluable 中 14 条恢复；但 oracle 与
full-context 各有 2 条 drift，污染 `constraint_loss` / `local_over_global`。
该结果固定标为 `post_hoc_exploratory`，只支撑 C2/C3 的描述性可行性，不替代
v0.11/v0.12 的 pre-specified C1/C2 evidence，也不建立整体 memory-specific
drift effect。

旧 v0.8 的 local conflict SCEU 还遗漏了当前 plan/update 的 action-relevant
state，且 authority/scope precedence 没有作为所有 condition 共享的 task
semantics。旧冻结数据不回写；这些错误不得解释为 backend drift。新 release
必须先通过 `current_action_state_contract_complete` 和 shared-governance prompt
测试，才允许发起服务器 policy calls。

1. 本地 4-session CI；
2. zyd connectivity smoke；
3. 5-episode v0.10 calibration，覆盖不同 semantic scenario 和 phase schedule；
4. 修正所有 measurement gates；
5. 冻结与 calibration 不重叠的 50-episode v0.10 confirmatory ranking split；
6. 先运行 3-group v0.11 server calibration，再冻结 30-group/90-physical-
   episode v0.11 matched mechanism split；
7. 使用 Slurm array 运行 GPT-only controlled/native tracks；
8. v0.10 做 episode-clustered ranking/sensitivity，v0.11 做 paired-by-group
   construct penalty、endpoint violation excess 和 attribution-stage change；
   v0.10 的 longitudinal onset/survival 与 v0.11 的 matched endpoint effect
   不混合。
9. 独立运行 3-panel v0.12 calibration，检验同一决策下 construct penalty 是否
   随 joint horizon dose 放大；该结果不与 v0.10/v0.11 pooled，也不解释为
   pure handoff effect。
10. 独立运行 5-episode v0.13 calibration；C2 的四类 lineage/recovery coverage、
    C3 intervention target、oracle/full-context drift contamination 和 attribution
    stability 通过后，再冻结 50-episode disjoint confirmatory split。

v0.11 使用独立配置
`configs/experiments/systems_controlled_gpt_only_matched_v011.yaml`。planning
阶段必须核对 dataset manifest release，并拒绝任何切断 triplet 的
`--episode-limit`。run/report manifest 同时记录 physical episode 数、
counterfactual group 数和 primary analysis unit。机制实验继续运行 causal-use
所需的 state-targeted leave-one-out/replacement probes，但关闭与主 estimand
无关的 +1/+5/+20 count-load probes。服务器校准为 3 groups/9 physical/63
evaluation tasks，confirmatory 为 30 groups/90 physical/630 evaluation tasks。
`analysis_phase` 必须写入 immutable run identity 并由 worker 重验：3-group
v0.11 使用 `calibration`，30-group v0.11 使用 `confirmatory`；标准及 v0.13
longitudinal release 对应至少 5/50 independent episodes。该阈值只防止错误标注，不替代 disjoint freeze、
preregistration、measurement gates 或统计不确定性。

## Acceptance gates

- 每个 SCEU 都有可重建的 long-horizon construct profile；
- early opportunity 不把 future state 计为 required current state；
- memory condition 的每个 SCEU 都能产生 first-failure stage 或明确的 not-applicable reason；
- stage yields 使用条件 denominator，不把未存储状态计入 retrieval denominator；
- `behaviorally_used` 仅表示稳定干预识别出的 unique causal-effect lower
  bound；无 effect 不得解释为 unused；
- violation 与 observed longitudinal drift 已分离；onset/survival 必须有
  同一 state-lineage 的 prior-adherence anchor，统计单位仍为 episode；
- category-only legacy trajectory 不通过 C2 gate，oracle-current-state 与
  full-context 对同 lineage 的 onset 必须为零；
- v0.13 的每类 drift 必须在 policy call 前具备 distinct same-lineage
  anchor/challenge/recovery 三点窗口，且 checkpoint/SCEU 不增加统计单位；
- static/evolution/conflict matched controls 通过相同 action/checker contract；
- matched mechanism inference 以 counterfactual group 为单位，并输出 CI、
  effect size、paired sign-flip 与 terminal-archetype sensitivity；
- matched full-context 与 oracle terminal-contract gates 对每个 policy 和三个
  history variant 均通过；缺少 control row 不是 N/A，而是 C1 证据不完整；
- 至少 3 个 matched group 时三个 terminal gold action 和三个 opaque gold
  position 均被覆盖，固定 action/position baseline 不超过预注册阈值；
- 每个声称 long-horizon span 的 terminal decision 至少有 200 个
  effect-chain-verified causal ancestors；每个计数步骤产生唯一 semantic
  effect、消费声明前驱效果，并明确拆分 policy/frozen/environment step count；
- 每个 SCEU 的 current action-relevant state 均包含在 required closure，且
  stable governance semantics 对 workspace/full-context/oracle/memory conditions
  完全一致；
- 每个 task-span profile 必须输出 interaction mode、policy-conditioned future
  step、policy-dependent later decision 和 online-execution support；当前
  matched/horizon/longitudinal release 的 claim 必须限定为 replay-backed critical decision；
- horizon diagnostic 的 terminal state/workspace/action/checker signature 跨
  short/medium/long 必须各只有一个，统计单位必须为 `horizon_panel`，且不得
  将 joint dose 写成 pure handoff effect；
- report validation、dataset regen-check、ruff、mypy 和相关 pytest 全部通过；
- contribution evidence 可由底层 artifacts 确定性重建；matched release 的
  standard per-episode control gates 为 N/A 时，必须由 group-level matched
  replacements 通过，不能将 N/A 解释为豁免；
- `fault_profile_divergence.json` 可由 decision rows 确定性重建，并明确报告
  aligned pair、outcome-equivalent pair 和 divergence rate；
- paid API 调用开始前，experiment design audit 已绑定到 run identity，并且
  三组校准的 `balanced_mechanism_design_ready=true`；worker 会重新计算该
  audit，拒绝事后替换；
- calibration/confirmatory phase 已绑定 run identity；matched 统计单位按
  group 计数，3 个 physical triplet members 不得充当 3 个独立单位；
- 5-episode calibration 的 measurement readiness 为 true 后，才允许启动 50-episode confirmatory run。
