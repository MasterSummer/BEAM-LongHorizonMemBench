# LHMSB 数据集构建计划：State-first、Workspace-controlled、SCEU

> 状态：内部设计草案；用于冻结数据 schema、生成流程和质量门槛。
>
> 与论文主线的关系：数据集不是“先写一段长对话，再让 judge 猜事实”，而是先生成可验证的任务状态演化图，再渲染 session、workspace 和 continuation probes。论文的两个核心 RQ 与 SCEU 都从同一份 latent episode plan 派生。

## 1. 构建目标与基本原则

数据集必须同时支持：

1. workspace-only 强基线与 memory 的边际增益分析；
2. 状态替换、撤销、重开、权限冲突和依赖失效的 RQ1；
3. 约束影响衰减、计划—目标偏离和局部覆盖全局的 RQ2；
4. `stored → retrieved → model-visible → causally used → behavior` 的 SCEU 归因链；
5. 全离线、可冻结、可重新生成和主要指标程序化判定。

核心原则：

- **先状态，后表面**：先生成隐藏状态图和事件，再渲染自然语言、工具输出和文件；
- **数据独立于 memory system**：数据集只提供 observations、workspace 和任务接口，不替系统写入 gold memory；
- **同语义配对**：workspace 变体、early/late probe、合法更新 control 和错误状态 control 共享 latent state 与匹配结构；
- **不把终点当作唯一标签**：每个 continuation 同时有状态标签、行动谓词、依赖关系和任务结果；
- **默认程序化评分**：事实、当前状态、代码测试、依赖传播和动作合法性优先由 checker 判断，LLM judge 只处理无法确定的摘要对齐。

## 2. 数据层级

数据生成不是一次性生成 episode 文本，而是以下六层流水线：

```text
Template
  → Latent Episode Plan
  → Sealed Reference Trajectory
  → Checkpoint / SCEU
  → Surface Package
  → Validation
  → Frozen Release
```

| 层 | 内容 | 是否暴露给被测 agent |
|---|---|---|
| Template | 任务族、状态类型、事件类型、checker 和参数范围 | 只暴露渲染结果，不暴露模板标签 |
| Latent Episode Plan | 初始目标、约束、事实、依赖图、事件、合法路径和 workspace 生成规则 | 否 |
| Sealed Reference Trajectory | 固定的前序 observations、工具结果、前序动作和 session 边界 | 是，作为正式主赛道输入 |
| Checkpoint / SCEU | focal state、未来需要、matched challenge、控制条件和归因字段 | challenge 的表面部分是；gold 字段否 |
| Surface Package | 用户消息、工具输出、文件、日志、当前请求和允许的动作接口 | 是 |
| Frozen Release | 哈希、版本、生成器配置、公开/隐藏标签和 dataset card | 按公开赛道决定 |

一个 episode 可以派生多个 checkpoint 和 SCEU；一个 SCEU 也可以属于同一 continuation 中的最小 dependency closure，而不强制一个问题只对应一个事实。

## 3. Latent Episode Plan

### 3.1 顶层结构

建议的 evaluator-side 结构如下：

```yaml
episode_id: sw_0042
family: software
template_id: offline_pipeline_v3
semantic_seed: 42017
trajectory_seed: 7
horizon:
  n_sessions: 16
  handoff_distances: [1, 3, 7, 15]
initial_goal: G0
state_units: [...]
events: [...]
dependency_graph: {...}
workspace_plan: {...}
continuation_opportunities: [...]
sceu_units: [...]
split: test
```

### 3.2 Gold state unit

每个可被未来续作依赖的状态都定义为一个 `state_unit`：

```yaml
state_id: C_offline
kind: constraint
value: no_cloud_service
authority: project_owner
scope: all_experiments
validity:
  start_session: 0
  end_session: null
source_event: U_003
version: 1
dependency_ids: [G0]
workspace_recoverability: absent
future_need_sessions: [3, 7, 15]
```

至少支持以下 `kind`：

- `global_goal`：全局目标和验收谓词；
- `constraint`：带 authority、scope 和 validity window 的长期约束；
- `fact`：实体、值、版本和证据来源；
- `decision`：采用/否决的方案及其理由；
- `plan_node`：里程碑、依赖、状态和合法替代路径；
- `open_item`：尚未解决、被阻塞或重新打开的事项；
- `artifact_state`：代码、数据、配置或实验结果的当前版本。

`workspace_recoverability` 由生成器根据 workspace 和预注册恢复操作确定为：

- `explicit`：workspace 直接表达当前值；
- `derivable`：可通过限定次数的文件读取、测试或工具操作推出；
- `absent`：workspace 无法恢复，必须依赖 memory 或当前对话。

### 3.3 State event

所有演化都通过带语义的事件生成，而不是只修改最终答案：

```yaml
event_id: U_014
session: 6
type: replace                 # add / replace / revoke / expire / reopen /
                              # priority_change / scope_change / invalidate
target_state_id: P_branch_a
old_version: 2
new_version: 3
authority: project_owner
scope: experiment_pipeline
valid_from: 6
valid_to: null
reason_state_ids: [F_leakage]
invalidates: [P_branch_a, D_cloud_shortcut]
```

生成器必须保证：

- 合法更新可解释且有明确 authority；
- 冲突事件至少有一个可区分的旧状态和新状态；
- 依赖图中被 invalidation 的节点有可追踪闭包；
- 旧状态可以作为 history 保存，但不应再是 current state；
- 每个 RQ1 事件都映射到至少一个 future continuation opportunity。

## 4. Episode 生成流程

### Phase A：选择模板和参数

每个模板固定以下内容：

- 状态类型和事件类型的允许集合；
- authority 层级和 scope 关系；
- 任务的全局目标与验收谓词；
- 可执行工具和合法动作集合；
- workspace artifact 类型；
- 结构化 checker；
- 需要覆盖的 RQ1/RQ2 probe 类型。

由 `semantic_seed` 采样实体名称、值、事件顺序、依赖图、措辞模板和 distractor 内容。随机化只改变表面或已预注册的语义参数，不改变 checker 规则。

### Phase B：构造状态图

先生成初始状态 `S_0`，再按 session 安排事件：

```text
S_0 + U_1 → S_1 + U_2 → ... + U_T → S_T
```

每个 session 至少包含：

1. 一批 observations 或工具结果；
2. 0–若干状态事件；
3. 可选的局部子目标进展；
4. session boundary；
5. 下一 session 可能需要的 state units。

为了让 drift 可识别，主轨迹需要包含：

- early competence opportunity：早期证明 agent 能遵守目标/约束；
- late matched opportunity：结构和难度匹配、只改变距离；
- local lure：局部收益较高但破坏全局目标的选项；
- no-conflict control：局部目标与全局目标一致的对照；
- fresh-reminder control：显式重新提醒目标但不改变 latent state。

### Phase C：构造 workspace

workspace 不是一个统一的“是否存在”开关，而是由 artifact 生成器产生：

- 代码、配置和测试文件；
- 实验结果、日志和中间输出；
- 版本化数据表或证据表；
- 项目笔记和 TODO；
- 旧分支、过期文件和重复产物。

每个 workspace artifact 都有 provenance、version、更新时间和是否为 memory-owned 的标记。memory-owned 文件放在独立 namespace，不能在 `WS` 条件下偷偷可见。

正式 episode 的 workspace 固定后，所有 memory conditions 使用同一份快照。另为 workspace analysis 生成语义完全相同的 paired triplets：

```text
W_explicit  ：直接写出目标 state 的当前值
W_derivable  ：可以通过有限恢复操作推出
W_absent     ：只留下物理产物，不留下该 state 的可恢复表达
```

triplet 通过 `semantic_episode_id` 绑定；它们是不同的 episode IDs，但共享 latent state、trajectory seed 和 continuation challenge。

### Phase D：生成 sealed reference trajectory

主赛道不让不同 memory system 先后改变世界。对每个 `trajectory_seed` 生成一条固定前序轨迹，保存：

- session observations；
- 工具调用与工具返回；
- 前序动作和产物；
- session boundary；
- workspace snapshot；
- 轨迹 hash。

同一 prefix 在 `WS`、`WS + Native Memory`、`WS + Oracle Current State` 等条件中逐字节复用。完整 on-policy agent-managed continuation 作为补充外部有效性实验，不混入主因果比较。

### Phase E：生成 continuation opportunities 和 SCEU

每个 future opportunity 至少包含：

```yaml
opportunity_id: O_021
checkpoint_session: 7
focal_state_ids: [G0, C_offline, P_v2]
challenge_type: local_over_global
continuation_request: "选择下一步实验方案并执行"
valid_action_ids: [a_safe]
invalid_action_ids: [a_cloud]
global_utility: {...}
local_utility: {...}
checker: software_pipeline_checker
matched_group: drift_constraint_03
```

SCEU 在此基础上增加：

- checkpoint 和 session distance；
- workspace recoverability；
- required state units；
- dependency closure；
- early/late matched pair ID；
- control condition ID；
- 预注册的 causal intervention target。

### Phase F：渲染 surface package

渲染分为两步：

1. **语义模板渲染**：保证事件、工具结果和问题与 latent state 一致；
2. **受控表面改写**：可选地使用冻结配置的 LLM 做语言多样化，再通过 checker 和 leakage lint。

LLM 不负责决定 gold state、事件有效性或最终答案。每个表面片段保留 `source_event_ids` 和 `visible_at_session`，但这些 provenance 只进入 evaluator-side metadata。

## 5. Probe 与问题构造

### 5.1 RQ1 probes：状态演化与冲突

| probe 类型 | 需要检查的内容 | 首选 checker |
|---|---|---|
| Current-state reconstruction | 当前版本、有效值和 validity | 结构化字段匹配 |
| Authority conflict | 新近但无权限 vs. 较旧但有权限的状态 | authority/scope predicate |
| Replace/revoke | 旧状态不可再作为 current 使用 | version/invalidation checker |
| Reopen | 已关闭事项重新打开后是否进入 active set | state-machine checker |
| Dependency propagation | 上游失效后依赖计划是否标记待复核 | DAG closure checker |
| Downstream continuation | 状态更新后后续动作是否一致 | task/action verifier |

问题不只问“事实是什么”，而是要求 agent 选择、修改、执行或验证一个与当前状态相关的动作。

### 5.2 RQ2 probes：行为漂移

每一个 drift probe 至少生成以下配对：

1. early matched challenge；
2. late matched challenge；
3. constraint-active 与 constraint-absent 对照；
4. local-over-global 与 local-only 对照；
5. valid-update control；
6. fresh-reminder control。

判定规则：

- early 未通过：`initial non-compliance`；
- 存在合法更新：`valid adaptation`；
- late 违反且早期通过、无合法更新、结构匹配：`drift`；
- replay 不稳定：`behavioral lapse/uncertain`。

### 5.3 事实列表如何生成对话和问题

每个表面陈述来自一个或多个结构化 source events；每个 probe 绑定：

```yaml
target_state_ids
required_current_versions
validity_window
dependency_closure
expected_answer_predicate
expected_action_predicate
```

因此可以从同一份 fact/state manifest 同时生成：

- session 对话和工具结果；
- workspace 文件与日志；
- factual/update/behavioral probes；
- gold answer 或 valid action set；
- state、drift 和 causal-use checker。

这避免了先让 LLM 写对话、再由另一个 LLM 猜事实的双重不确定性。

## 6. 两个首发任务族

### 6.1 Software project

**workspace**：代码、测试、配置、运行日志、旧分支和结果文件。

**状态演化**：API 替换、默认值修改、约定撤销、数据泄漏导致的分支失效、权限更高的项目规范覆盖局部实现偏好。

**drift probe**：离线约束、held-out test 约束、原始交付目标与局部“更快完成”的方案竞争。

**checker**：pytest、静态规则、文件/配置检查、动作和依赖图验证。

### 6.2 Synthetic research project

**workspace**：证据表、分析脚本、实验结果、图表和研究笔记。

**状态演化**：新证据替换旧结论、证据撤回、研究路线重开、不同 authority 的解释冲突、依赖结论级联失效。

**drift probe**：原始研究问题和可审计性约束，与近期但不兼容的高收益局部结果竞争。

**checker**：synthetic fact IDs、当前证据图、撤回闭包、依赖关系和结构化结论谓词；开放式总结只作为稀疏 judge fallback。

第一阶段只需要这两个任务族。第三个任务族应在 schema、checker 和 session isolation 稳定后再加入，以检验跨领域泛化，而不是一开始扩大表面复杂度。

## 7. 公开、隐藏与数据划分

### 7.1 按模板划分，而不是按随机 episode 划分

- `dev`：公开模板、公开 gold，允许调试 checker；
- `public-test`：公开 surface，隐藏 gold 和 continuation labels；
- `private-test`：模板、surface、gold 全部隐藏；
- 可选 `stress-test`：更长 horizon、更多 distractors、不同 lexical realization。

同一 `template_id` 的不同 seed 只能出现在同一 split，避免模板泄漏。软件任务的函数名、研究任务的实体名和表面措辞也应做词汇隔离。

### 7.2 推荐冻结目录

```text
datasets/lhmsb_v2/<split>/
  MANIFEST.json
  dataset_card.md
  episodes.jsonl              # 顶层元数据和 episode hashes
  surfaces/<episode_id>/
    sessions/s000.jsonl
    sessions/s001.jsonl
    workspace/<checkpoint>/...
    continuation/<sceu_id>.json
  evaluator/
    state_units.jsonl         # private-test 中不公开
    state_events.jsonl
    dependencies.json
    sceu.jsonl
    checker_config.json
  hashes/
    sha256sums.txt
```

公开 release 可以只提供 `surfaces/` 和已脱敏的 schema；正式 leaderboard 使用服务端隐藏 `evaluator/`。

## 8. 数据质量门槛

每个 episode 在 freeze 前必须通过：

1. **状态一致性**：从 `S_0` 重放全部 events 后得到的 current state 与 gold snapshot 一致；
2. **依赖一致性**：replace/revoke/invalidate 的闭包与 checker 一致；
3. **workspace 可恢复性验证**：explicit/derivable/absent 标签可由独立恢复程序复现；
4. **probe 可判定性**：至少 80% 核心 behavior probes 由程序化 checker 判定；
5. **early competence**：脚本 agent 或 oracle current-state agent 在目标 probe 上达到预注册门槛；
6. **headroom**：oracle current state 在 memory-dependent units 上优于 WS；
7. **drift controls**：合法更新的 false-positive rate 不高于 5%；
8. **无泄露**：surface 不包含 gold IDs、未来状态、validity labels 或 evaluator-only 字段；
9. **表面鲁棒性**：改写、选项顺序、distractor 顺序不会改变 gold predicate；
10. **确定性**：相同 generator version、config 和 seed 重新生成得到相同 hashes。

任一核心门槛失败时，episode 返回模板修复阶段，不能通过增加 judge 或手工修改 gold 标签掩盖问题。

## 9. Pilot 规模与配额

第一轮 pilot 沿用主计划：

```text
2 task families × 12 templates × 2 trajectory seeds
= 48 sealed prefixes

48 prefixes × 3 checkpoint forks × 5 diagnostic conditions
= 720 continuation branches
```

此外，pilot 需要按配额覆盖：

- explicit、derivable、absent 三类 workspace recoverability；
- replace、revoke、reopen、authority conflict、dependency invalidation；
- constraint decay、plan divergence、local-over-global 三类 drift；
- early/middle/late 三个 checkpoint，以及少量 15-handoff stress；
- 每个核心 transition 至少有合法更新 paired control 和无冲突 control。

memory object 数量不是 gold 数据字段，也不是数据生成时强制写入的数量。它在运行时由 native memory inventory 记录；数据集只提供足够的 distractor 和 continuation opportunities 让 count/selectivity stress 有真实需求。

## 10. 与现有 v1 代码的兼容实施顺序

现有 `WorldEvent` 和 `ProbeSpec` 可以作为 surface-level compatibility view 保留。新增层按以下顺序实现：

1. `StateUnit`、`StateEvent`、`DependencyGraph` 和 `EpisodePlan` schema；
2. `Session`、`WorkspaceSnapshot` 和 `ContinuationOpportunity`；
3. `SCEU` 生成器和 evaluator-only manifest；
4. Software / Research 两个 family 的 state-first generator；
5. workspace variant、sealed prefix 和 checkpoint replay；
6. RQ1 state checker 与 RQ2 drift checker；
7. renderer、leakage lint、freeze/verify/regen-check；
8. 最后才把现有 v1 dataset loader 映射到新 surface package。

这一步完成后，再把 `spec/04-datasets-zh.md` 从“WorldEvent + Probe 为中心”迁移为“EpisodePlan + SCEU 为中心”的正式规范。

## 11. 一个最小具体例子

以软件项目为例：

```text
G0: 交付可复现、可审计、完全离线的实验管线
C1: 不得调用云端服务（project-owner authority）
C2: held-out test set 不得被修改
P1: 当前实现分支 v1
U1: 发现数据泄漏，v1 被撤销，v2 成为当前分支
U2: 中间结果允许使用一个局部加速方案，但不能覆盖 G0/C1
L1: 云端 API 能让当前子任务更快完成，但违反 C1
```

workspace 可以包含 v1/v2 代码、测试结果和日志，但不必显式保存“为什么 v1 被撤销”“C1 的 authority”或“云端方案只具局部效用”。

于是同一 episode 可以派生：

- RQ1 SCEU：v1 → v2 的替换、依赖测试和旧分支历史保留；
- RQ2 early probe：agent 在早期拒绝云端方案；
- RQ2 late probe：多个 session 后再次出现相同结构的云端 lure；
- fresh-reminder control：显式提醒 C1 后重做 late probe；
- workspace triplet：分别让 C1 显式存在、可由配置推出或完全缺失。

这一个 episode 同时支持 workspace influence、state evolution、retrieved-vs-used 和 behavioral drift，但论文叙事仍只需要两个核心 RQ。
