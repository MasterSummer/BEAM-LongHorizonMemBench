# Long-Horizon Task Memory Benchmark：被测系统选择（修订版）

**日期**：2026-07-17  
**状态**：候选集，正式冻结前需完成 adapter qualification  
**修订原因**：上一版主要选择了在 LoCoMo、LongMemEval 等对话记忆评测中常见的 memory middleware。它们适合测用户事实、事件和偏好的保存与召回，但不能代表软件工程、网页操作和工具使用中的 task memory。

## 1. 结论

主实验应评测 **5 个近期 task-oriented memory systems**：

1. **ReMe**
2. **ACE**
3. **ReasoningBank**
4. **MemRL**
5. **Agent-KB**

它们都从 agent 执行轨迹、成功与失败经验、环境反馈或任务策略中构造 memory，并在 AppWorld、WebArena、SWE-bench、BigCodeBench、ALFWorld、GAIA 等 agentic tasks 上验证过。主实验不再使用 Mem0、Letta、Graphiti、Cognee 和 Hindsight 作为 headline systems。

完整实验分成两个不可混排的 track：

| Track | 被测对象 | 建议数量 | 比较原则 |
|---|---|---:|---|
| A. Non-parametric task memory | ReMe、ACE、ReasoningBank、MemRL、Agent-KB | 5 | 尽量固定 agent backbone、任务 surface 和反馈边界 |
| B. Learned memory policy | MemAct、AgeMem，MEM1 作为候选 | 2 | 使用论文发布的 checkpoint，单独报告，不和 Track A 排统一名次 |

主表包含：

```text
Workspace-only
Oracle-current-state
Flat retrieval baseline
ReMe
ACE
ReasoningBank
MemRL
Agent-KB
```

因此主表有 8 个 conditions，其中只有 5 个是完整 task-memory systems。

## 2. Benchmark 到底评测哪类 memory

[LoCoMo](https://arxiv.org/abs/2402.17753) 构造长程对话，并用问答、事件总结和多模态对话生成来评估记忆。它主要考察 agent 能否从对话历史中恢复人物事实和事件。

本项目评测另一类问题：

> Agent 在持续执行一个项目时，能否把目标、约束、计划、决策、失败经验和状态变更写成可用 memory，并在后续 continuation 中遵守当前状态，避免沿用失效分支或偏离全局目标？

因此，候选系统需要满足以下条件：

- memory 来自 task trajectory、tool result、test result 或环境反馈；
- 系统会提炼 workflow、strategy、experience、diagnosis 或 task state；
- memory 会影响后续 action，而非只回答关于过去的事实问题；
- 原论文至少包含一个 interactive、tool-use、web、software、embodied 或 multi-step agent benchmark；
- 开源实现允许记录 write、retrieve 和 model-visible memory；
- 系统能在 episode 内持续更新，而非只加载一份预先构建的静态知识库。

对话事实系统仍可用于跨应用迁移实验，但不应支撑论文关于 long-horizon task memory 的主要结论。

## 3. Track A：五个主实验系统

时间与社区数据采用同一口径：论文时间取 arXiv v1 首次提交日期，代码时间取 GitHub repository 创建日期；stars、forks 和最近 push 均为 **2026-07-16 UTC 的快照**。Stars 只描述公开关注度，不作为性能证据。仓库创建时间早于论文时间时，表示代码仓库先存在，论文随后公开。新增候选的 2026-07-17 核查见第 8 节。

| 系统 | 论文 / 代码时间 | GitHub 快照 | Native memory unit | 原始任务证据 | 本 benchmark 中的作用 |
|---|---|---|---|---|---|
| [ReMe](https://github.com/agentscope-ai/ReMe) | arXiv v1 2025-12-11；repo 2024-08-29 | 3,195 stars / 273 forks；latest push 2026-07-15 | 从成功、失败和比较中提炼的 procedural memory | [论文](https://arxiv.org/abs/2512.10696) 使用 BFCL-V3 和 AppWorld，并加入 utility-based add/prune | 检验系统能否保留有效 procedure、删除过时经验并控制 memory 数量 |
| [ACE](https://github.com/ace-agent/ace) | arXiv v1 2025-10-06；repo 2025-11-16 | 1,214 stars / 154 forks；latest push 2026-05-19 | 可增量编辑的 strategy/playbook item | [论文](https://arxiv.org/abs/2510.04618) 包含 AppWorld，并使用自然执行反馈进行 online adaptation | 检验增量更新能否减少 context collapse、冲突覆盖和长期 drift |
| [ReasoningBank](https://github.com/google-research/reasoning-bank) | arXiv v1 2025-09-29；repo 2026-02-26 | 438 stars / 56 forks；latest push 2026-07-10 | 从成功和失败轨迹蒸馏的 reasoning strategy | [论文](https://arxiv.org/abs/2509.25140) 在网页浏览和软件工程任务上评测 | 检验失败经验是否被正确写入，以及错误策略会不会在后续 session 持续影响行为 |
| [MemRL](https://github.com/MemTensor/MemRL) | arXiv v1 2026-01-06；repo 2026-01-12 | 141 stars / 14 forks；latest push 2026-05-02 | 带环境反馈效用值的 episodic strategy | [论文](https://arxiv.org/abs/2601.03192) 使用 HLE、BigCodeBench、ALFWorld 和 Lifelong Agent Bench | 检验 semantic relevance 与行为 utility 的差别，并支持 memory-level 因果归因 |
| [Agent-KB](https://github.com/OPPO-PersonalAI/Agent-KB) | arXiv v1 2025-07-08；repo 2025-06-11 | 445 stars / 30 forks；latest push 2025-08-19 | 结构化跨任务 experience、planning item 和 diagnostic feedback | [论文](https://arxiv.org/abs/2507.06229) 使用 GAIA、HLE、GPQA 和 SWE-bench | 检验共享经验的正迁移、knowledge interference 和局部经验覆盖全局约束 |

### 3.1 ReMe

本项目评测 ReMe 的 **procedural-memory pipeline**，不是仓库中的 LoCoMo 或 conversational-memory 配置。ReMe 包含三类适合本 benchmark 的行为：

- 从成功、失败和对比轨迹中提炼经验；
- 根据当前任务重用历史 procedure；
- 根据 utility 添加有效 memory，并移除过时 memory。

ReMe 对应 state evolution、memory-count scaling 和 write-to-continuation alignment。

### 3.2 ACE

ACE 把 context 组织成持续演化的 playbook，并通过 generator、reflector 和 curator 增量修改条目。它不依赖反复重写整份摘要，因此适合测试：

- 新约束到来后，旧策略是否仍残留；
- 局部经验是否错误覆盖全局规则；
- 多轮修改是否丢失早期仍有效的约束；
- playbook 变大后，新增条目是否仍有边际行为价值。

ACE 使用自然执行反馈的模式最适合本 benchmark。适配器不能把 evaluator gold 或隐藏 checker 标签传给 ACE。

### 3.3 ReasoningBank

ReasoningBank 同时从成功和失败经历中提取 reasoning memory。软件项目 episode 可以提供编译、测试、静态检查和工具错误等自然反馈，因此能产生有意义的成功或失败轨迹。

需要重点观察：

- 失败原因是否写成可泛化策略；
- 一次局部失败是否形成过强的全局禁令；
- 已失效的 v1 分支经验是否在 v2 continuation 中继续被检索；
- agent 是否把“当前任务成功”误当成“原始项目目标已满足”。

### 3.4 MemRL

MemRL 先按相关性检索，再用环境反馈学习 memory utility。它适合区分：

```text
retrieved because semantically similar
versus
retained because it improved behavior
```

每个 memory 需要记录：

- relevance-stage candidate；
- utility-stage selected item；
- model-visible item；
- action 后收到的公开环境反馈；
- utility update；
- leave-one-memory-out 后的行为变化。

这一机制与 `stored → retrieved → visible → behavior` 链条联系最直接。

### 3.5 Agent-KB

Agent-KB 从多种 agent framework 的轨迹中构造共享经验，并在 planning 和 diagnostic feedback 两个阶段调用 memory。它能测试跨任务经验是否造成 interference。

Agent-KB 的进入条件比其他四个系统更严格。正式适配前需要确认：

- 能否在本 benchmark 的 session boundary 上执行原生 experience construction；
- 能否从当前 episode 的已完成 session 中新增 memory；
- 是否允许从空 KB 开始，避免外部预训练经验污染 episode；
- planning 和 feedback 两阶段的 retrieved items 是否可记录。

如果它只能读取预构建 KB，不能进行 episode 内 native write，则不进入主榜。第五位置改由通过 qualification 的近期 task-memory system 补充；本轮优先检查第 8 节中的 MemOS，而不是把一个只读 KB 当成可写 memory system。

## 4. 为什么这五个系统具有代表性

五个系统覆盖 task memory 的五种关键机制：

| 机制 | 系统 | 对核心 claim 的贡献 |
|---|---|---|
| Procedural distillation and pruning | ReMe | 测量系统如何保留有效经验并删除过时经验 |
| Incremental playbook evolution | ACE | 测量状态更新、冲突合并和 context collapse |
| Success/failure reasoning memory | ReasoningBank | 测量错误经验如何形成、传播或被修正 |
| Feedback-weighted episodic utility | MemRL | 区分相关、可见和具有行为价值的 memory |
| Shared structured experience | Agent-KB | 测量正迁移、knowledge interference 和局部目标覆盖 |

这个组合对齐论文的两个主要 claim：

1. **State evolution and conflict resolution**：新状态写入后，系统是否更新、替换或抑制旧 memory；
2. **Long-horizon behavioral drift**：约束是否逐渐失去行为影响，计划是否偏离原始目标，局部子目标是否覆盖全局目标。

## 5. Track B：learned memory policy

Track B 包含直接把 memory operation 学成 agent action 的系统。它们与“agent 自己决定写什么”最接近，但更换了 agent policy 或模型 checkpoint。

| 系统 | 论文 / 代码时间 | GitHub 快照 | Memory behavior | 为什么单独报告 |
|---|---|---|---|---|
| [MemAct](https://arxiv.org/abs/2510.12635) | arXiv v1 2025-10-14；repo 2025-11-29 | 29 stars / 3 forks；latest push 2025-11-29 | 把 working-memory deletion 和 insertion 作为可学习 action | 使用专用 RL policy，主要管理工作上下文；无法和固定 backbone 的外部 memory system 做同一变量控制 |
| [AgeMem](https://arxiv.org/abs/2601.01885) | arXiv v1 2026-01-05；repo 2026-04-23 | 20 stars / 3 forks；latest push 2026-04-24 | Agent 自主选择 store、retrieve、update、summarize 和 discard | 通过分阶段 RL 学习 memory tool use；模型能力与 memory policy 绑定 |
| [MEM1](https://arxiv.org/abs/2506.15841) | arXiv v1 2025-06-18；repo 2025-07-14 | 325 stars / 21 forks；latest push 2026-01-03 | 每轮更新紧凑内部 state，联合执行 consolidation 和 reasoning | 依赖端到端训练，memory unit 与外部对象数量不直接对应 |

建议先在 diagnostic subset 上运行 MemAct 和 AgeMem。MEM1 作为替补或 supplement。

Track B 应报告：

- 发布 checkpoint 和训练设置；
- memory actions；
- 当前 memory object 或 internal state 的可测代理量；
- task success 和 drift；
- 与该系统自身 no-memory 或 full-context baseline 的差值。

论文不能把 Track B 与 Track A 合成单一排行榜，因为二者没有共享同一 agent backbone。

## 6. 其他近期系统的位置

| 系统 | 论文 / 代码时间 | GitHub 快照 | 位置 | 原因 |
|---|---|---|---|---|
| [MemEvolve](https://arxiv.org/abs/2512.18746) | arXiv v1 2025-12-21；repo 2025-12-21 | 253 stars / 28 forks；latest push 2026-05-05 | Meta-evolution supplement | 它同时改变 memory 内容和 memory architecture。被测对象随任务变化，适合研究架构演化，不适合固定架构主榜 |
| [MemQ](https://arxiv.org/abs/2605.08374) | arXiv v1 2026-05-08；repo 2026-05-08 | 10 stars / 0 forks；latest push 2026-05-13 | Qualification candidate / causal diagnostic | provenance DAG 与 memory credit assignment 高度契合本项目，但发布较新，工程成熟度和社区采用度仍低 |
| [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) | arXiv v1 2024-09-11；repo 2024-08-28 | 446 stars / 51 forks；latest push 2025-12-22 | Historical task-memory baseline | 它是 workflow memory 的经典任务型方法，并支持 online induction；只在需要历史锚点时运行，不占近期主系统名额 |
| [SkillWeaver](https://arxiv.org/abs/2504.07079) | arXiv v1 2025-04-09；repo 2025-04-10 | 141 stars / 18 forks；latest push 2025-04-14 | Web-specific supplement | 自动生成可重用 web skills，机制有价值，但领域范围比 software-project benchmark 更窄 |

## 7. 旧 LoCoMo 型系统如何处理

Mem0、Letta、Graphiti、Cognee 和 Hindsight 不进入主榜。它们可组成一个小型 **conversational-to-task transfer** 补充实验，用于回答：

> 面向人物事实和对话事件设计的 memory system，迁移到持续软件项目后会出现哪些 state-update 和 behavioral-drift failure？

这些系统的代码社区规模明显高于多数近期 task-memory 论文，但这个数字反映产品生态和使用者数量，不能替代任务型机制证据。

| 系统 | 代码首次公开 | GitHub 快照 | 最近 push | 补充实验中的位置 |
|---|---|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | repo 2023-06-20 | 60,965 stars / 7,095 forks | 2026-07-16 | fact-centric transfer baseline |
| [Letta](https://github.com/letta-ai/letta) | repo 2023-10-11 | 23,816 stars / 2,525 forks | 2026-07-03 | agent-managed memory transfer baseline |
| [Graphiti](https://github.com/getzep/graphiti) | repo 2024-08-08 | 28,799 stars / 2,903 forks | 2026-07-16 | temporal-validity transfer diagnostic |
| [Cognee](https://github.com/topoteretes/cognee) | repo 2023-08-16 | 27,978 stars / 2,765 forks | 2026-07-16 | generic graph/vector transfer baseline |
| [Hindsight](https://github.com/vectorize-io/hindsight) | repo 2025-10-30 | 18,464 stars / 1,130 forks | 2026-07-15 | reflection-oriented transfer diagnostic |

这个补充组最多选择 1 至 2 个系统：

- 一个主流 fact-memory system；
- 一个显式 temporal validity system。

补充实验不参与主排名，也不用于证明 task-memory systems 的总体优劣。

## 8. 2026-07-17 对新增候选的核查

“MemOSMemoryOS”不是一个系统，而是两个不同项目：**MemOS** 是 MemTensor 的 memory operating system，**MemoryOS** 是 BAI-LAB 的分层对话记忆系统。它们必须分开安装、适配和报告。本节的日期、stars、forks 和 latest push 是 **2026-07-17 UTC 快照**；stars 只表示公开关注度，不表示任务性能。

| 系统 | 论文 / 代码时间 | GitHub 快照 | 核心机制 | 与本 benchmark 的关系 | 结论 |
|---|---|---|---|---|---|
| [A-MEM](https://github.com/agiresearch/A-mem) | arXiv v1 2025-02-17；NeurIPS 2025；repo 2025-02-25 | 1,113 stars / 119 forks；latest push 2025-12-12 | Zettelkasten 式结构化 note、动态链接、增量 update | 有 native add/search/update/delete，但原始证据主要是 LoCoMo 对话 QA | 做 state-evolution/linking 诊断，不直接进主榜 |
| [MemOS](https://github.com/MemTensor/MemOS) | arXiv v1 2025-05-28；repo 2025-07-06 | 10,242 stars / 934 forks；latest push 2026-07-16 | MemCube、文本/激活/参数 memory 统一、feedback correction、技能演化 | 最接近可写、可检索、可编辑、可追踪的通用 memory OS | 最高优先级 qualification；通过后可替换 Agent-KB |
| [MemoryOS](https://github.com/BAI-LAB/MemoryOS) | arXiv v1 2025-05-30；EMNLP 2025 Oral；repo 2025-05-30 | 1,515 stars / 152 forks；latest push 2026-07-07 | short-/mid-/long-term 分层存储、更新和检索 | 适合测分层生命周期，但主要是个人对话记忆 | 低优先级 lifecycle/transfer baseline |
| [SimpleMem](https://github.com/aiming-lab/SimpleMem) | arXiv v1 2026-01-05；repo 2026-01-01 | 3,650 stars / 375 forks；latest push 2026-06-23 | semantic structured compression、online synthesis、intent-aware retrieval | 原生 memory unit 清晰，特别适合 memory-count、压缩和 selectivity | RQ5/效率诊断；冻结 text core，不混入 Omni/EvolveMem |
| [LightMem](https://github.com/zjunlp/LightMem) | arXiv v1 2025-10-21；ICLR 2026；repo 2025-06-11 | 969 stars / 88 forks；latest push 2026-07-16 | sensory/short-term/long-term 分层，sleep-time offline update | 适合测在线写入、离线 consolidation、延迟和成本 | lifecycle/efficiency 诊断，不作为 state-conflict 主系统 |
| [Mem0](https://github.com/mem0ai/mem0) | arXiv v1 2025-04-28；repo 2023-06-20 | 61,021 stars / 7,102 forks；latest push 2026-07-16 | 多 session fact extraction、consolidation、retrieval，含 graph variant | 工程成熟度最高，适合作为当前真实 backend qualification 和 fact-memory transfer anchor | 保留现有 qualification；不把它包装成 task-memory 主系统 |

### 8.1 A-MEM：机制很有价值，但任务证据不足

A-MEM 把每条记忆组织为带上下文描述、关键词、标签和链接的结构化 note，并允许新记忆加入后更新既有链接和属性。它的 `add_note`、`search_agentic`、`update` 和 `delete` 使 native object trace 比很多对话 middleware 更容易获得。[论文](https://arxiv.org/abs/2502.12110) 的主要实验仍是 LoCoMo 类对话记忆，因此它没有直接证明能处理 software episode 中的 authority、validity window、replace/revoke 或跨 session 计划约束。

本 benchmark 最适合用 A-MEM 回答：“新状态加入后，旧 note 是否被重新链接、抑制或错误共存？”这对应 state evolution and conflict resolution；它不是当前主榜的 task-performance 代表。还需记录 LLM 生成 note、链接和更新所产生的额外调用成本。

### 8.2 MemOS：本轮最值得升级为主榜候选

MemOS 不是单一的向量库，而是一个 memory OS。论文提出 [MemCube](https://arxiv.org/abs/2505.22101) 来统一 parametric、activation 和 plaintext memory；当前仓库提供 add/retrieve/edit/delete、可检查图结构、feedback correction、多个 memory cube 和技能演化等路径。这个接口集合最有希望覆盖本项目要求的 `stored → retrieved → visible → behavior` 链。

但它的范围也最大：不同 MemCube、插件和 memory 类型会改变被测对象。Qualification 必须冻结 **一个本地 profile**（建议 plaintext/local plugin），从空 namespace 开始，关闭云服务、参数 memory 和预构建 skill；否则 MemOS 的“系统能力”与其他 non-parametric systems 不可比。仓库 README 自报的多种 benchmark 结果只能作为工程线索，不能替代逐项的 native-write qualification。通过后，MemOS 是最合理的 Agent-KB 替代或扩展候选。

### 8.3 MemoryOS：清楚的层次基线，不是当前主线

MemoryOS 采用 short-term、mid-term、long-term 三层个人记忆，以及对应的 storage、updating、retrieval 和 generation 模块；官方实现和论文主要围绕 LoCoMo 对话 QA。[论文与仓库](https://arxiv.org/abs/2506.06326) 没有给出本 benchmark 所需的显式 authority、版本替换、invalidation 或任务执行反馈接口。因此它能帮助我们测“分层生命周期是否比平面摘要更稳”，但不能单独支撑 state evolution 或 behavioral drift 的主结论。

### 8.4 SimpleMem：RQ5 最强的新增诊断候选

SimpleMem 将对话压缩为语义结构化、可索引的 atomic memory units，再进行 online semantic synthesis 和 intent-aware retrieval。[论文](https://arxiv.org/abs/2601.02553) 报告的重点是 LoCoMo、MemBench 等记忆问答和推理效率，而不是 software task continuation；不过它的 object 粒度、压缩率和检索候选天然适合本项目的 **memory-count scaling and selectivity**。

实验时必须锁定仓库中的 **SimpleMem text core**。当前仓库还打包了 Omni-SimpleMem 和 EvolveMem；若不固定 commit，就无法判断性能来自压缩 memory、multimodal memory 还是 retrieval self-evolution。应暴露每个 atomic unit 的 ID、合并/更新历史、候选集合和 model-visible 集合，并把 token/API/runtime 作为辅助成本，而不是替代 object count 的主横轴。

### 8.5 LightMem：适合 lifecycle 和效率，不宜直接主榜

LightMem 用 sensory memory 过滤和压缩输入，用 topic-aware short-term memory 做会话内整合，再以 sleep-time offline update 写入长期记忆。[论文](https://arxiv.org/abs/2510.18866) 主要在 LongMemEval 和 LoCoMo 上展示召回、token、API 和运行时收益。它可用于检查“在线写入—离线 consolidation—后续 continuation”的生命周期和成本，但当前证据没有明确处理 state version、authority conflict 或旧分支 invalidation。

一个重要的工程信号是其 README 仍列出 “Coordinated Use of Context and Long-Term Memory Storage” 为 TODO。这与本项目要隔离 workspace/context 影响的 claim 直接相关，说明 LightMem 可以作为很好的诊断对象，却不应在该问题尚未闭合时被当作主榜结论的代表。

### 8.6 Mem0：保留为工程 anchor，而非任务记忆主证据

Mem0 的社区和工程成熟度在这六个系统中最高，适合作为当前真实 backend qualification、稳定性和 fact-memory transfer 的 anchor。[论文](https://arxiv.org/abs/2504.19413) 的主要问题设定是多 session 对话中的事实抽取、合并和检索，并非持续软件项目中的计划与约束演化。因此 Mem0 可以回答“通用 fact memory 迁移到 task continuation 会怎样”，但不能被解释为 task-memory 主系统的代表。

### 8.7 本轮决策和 qualification 顺序

| 研究目标 | 首选系统 | 进入条件 |
|---|---|---|
| 主榜新增通用 memory OS | MemOS | 本地单 profile 通过 native write/retrieve/edit/delete、visible trace 和 replay |
| RQ5 memory-count scaling/selectivity | SimpleMem | text core 的 atomic unit、merge history 和候选集合可记录 |
| State evolution and conflict resolution | A-MEM | note/link/update 版本可追踪，且不读取 evaluator gold |
| Lifecycle、latency、offline consolidation | LightMem | online/offline 阶段可重放，工作上下文可在 session boundary 清空 |
| 分层对话记忆 transfer | MemoryOS | 作为 LoCoMo-style hierarchical baseline 单独报告 |
| 当前真实后端工程锚点 | Mem0 | 沿用已有 qualification，不改变主榜定义 |

建议的执行顺序是：**Mem0（已有） → MemOS → SimpleMem → A-MEM → LightMem → MemoryOS**。如果下一阶段只能增加一个新系统：研究主线优先 MemOS；若只验证 RQ5，改选 SimpleMem；若只验证状态链接和冲突，改选 A-MEM。

## 9. 公平比较需要固定哪些变量

### 9.1 反馈边界

ACE、ReasoningBank、MemRL 和 ReMe 都可能使用执行反馈。系统只能看到 agent 在现实项目中能看到的信息：

- tool output；
- public tests；
- compiler 和 linter errors；
- user 或 environment 在当前 session 给出的反馈；
- workspace 文件和当前任务请求。

以下信息只供 evaluator 使用：

- latent state IDs；
- hidden valid-action labels；
- state dependency graph；
- future events；
- hidden software checker result；
- drift gold labels。

隐藏 checker 在 continuation action 完成后评分，但评分结果不能回流到同一次 episode 的 memory system，除非该 episode 明确设计为带公开反馈的后续 session。

### 9.2 写入者身份

实验必须区分三种 write origin：

```text
controller-generated
agent tool-call generated
learned-policy generated
```

ReMe、ACE、ReasoningBank、MemRL 和 Agent-KB 多由外部 controller 或框架完成蒸馏。MemAct 和 AgeMem 把 memory operation 纳入 agent policy。论文需要分别报告，不能统一描述成 “the agent autonomously writes memory”。

### 9.3 Native memory unit

RQ5 使用 **memory object 数量**，不使用 token 数作为主横轴。不同系统保留原生对象语义：

| 系统 | 一个 memory object 的建议定义 |
|---|---|
| ReMe | 一个独立 procedural memory item |
| ACE | merge 后的一个 playbook entry |
| ReasoningBank | 一个 reasoning strategy |
| MemRL | 一个可独立检索和更新 utility 的 episodic item |
| Agent-KB | 一个独立 planning 或 diagnostic KB item |

Token、字符数和 embedding 存储量作为辅助成本指标。不能把一整份 playbook 当作一个 object 来获得虚假的常数规模。

### 9.4 Retrieved、visible 和 used

每个系统统一记录：

```text
stored objects
→ retrieval candidates
→ selected/retrieved objects
→ model-visible objects
→ selected action
→ checker behavior
```

`used` 不通过模型自述判断。Evaluator 使用以下证据：

1. model-visible memory；
2. action 与 memory 所支持或违反的 state 对齐；
3. leave-one-memory-out intervention；
4. 删除该 memory 后 action、score 或 drift flag 是否变化。

MemRL 和 MemQ 的 utility/provenance 只能作为系统内部证据，不能替代外部行为干预。

## 10. Adapter qualification

候选系统先在冻结的 Software vertical slice 上运行资格测试。资格测试检查工程可评测性，不按分数淘汰系统。

### 必须通过

- episode 从空 memory namespace 开始；
- session boundary 后工作上下文清空；
- 系统按原生方式写 memory，不由 benchmark 代写理想摘要；
- 每次 write、update、delete、retrieve 可记录；
- memory object count 可计算；
- model-visible memory 可重建；
- 使用公开反馈，不读取 evaluator gold；
- 同一配置可重放；
- 代码版本、模型、embedder、reranker 和内部 LLM 可冻结；
- API 错误、重试和成本可记录。

### 系统特定检查

| 系统 | Qualification 风险 |
|---|---|
| ReMe | 需要锁定 procedural pipeline，避免误用 conversational benchmark wrapper |
| ACE | 需要把 playbook merge 后的条目暴露为可计数对象 |
| ReasoningBank | self-judgment 不能读取 hidden checker；需要保存成功与失败 memory 的来源 |
| MemRL | 环境 reward 必须来自公开反馈；需要暴露两阶段 retrieval 和 utility update |
| Agent-KB | 必须支持 episode 内 write；若只能使用预构建 KB，则退出主榜 |

## 11. 实验流程

### Stage 0：one-episode adapter qualification

使用冻结的 16-session Software exemplar。每个系统只跑一个 seed，检查 write、retrieve、visible、reset 和 object count。

### Stage 1：benchmark validity pilot

运行：

```text
Workspace-only
Oracle-current-state
ReMe
ACE
ReasoningBank
```

目标是确认：

- workspace absent 状态下，Oracle 与 Workspace-only 有稳定差距；
- 至少一个 task-memory system 能改变 continuation action；
- stale-state、constraint loss 和 plan drift 可以由程序 checker 区分；
- leave-one-memory-out 能产生方向合理的行为变化。

### Stage 2：Track A 主实验

运行 5 个 task-memory systems、一个 flat retrieval baseline 和两个 controls。所有系统使用同一批 frozen episodes、同一 continuation checkpoints 和相同 agent-visible feedback。

### Stage 3：Track B diagnostic subset

在分层抽取的 episode subset 上运行 MemAct 和 AgeMem。报告各自 checkpoint 下的绝对表现及相对自身 baseline 的提升。

### Stage 4：机制补充

按论文篇幅和预算选择：

- MemQ provenance/credit diagnostic；
- MemEvolve architecture-evolution supplement；
- 1 至 2 个 LoCoMo 型系统的 task-transfer test。

## 12. 论文中的推荐表述

英文：

> We evaluate five recent task-oriented memory systems that construct reusable knowledge from agent execution rather than conversational fact histories. The main track covers procedural refinement (ReMe), evolving playbooks (ACE), success-and-failure reasoning memories (ReasoningBank), feedback-weighted episodic utility (MemRL), and shared structured experience (Agent-KB). We evaluate learned memory policies such as MemAct and AgeMem in a separate track because they require specialized model training and do not share the same frozen agent backbone.

中文：

> 我们评测五个近期任务型记忆系统。它们从 agent 执行轨迹中构造可复用经验，而不是从长对话中抽取人物事实。主实验覆盖程序性经验精炼、增量 playbook、成功与失败推理记忆、反馈加权 episodic memory 和共享结构化经验。需要专用训练的 memory policy 另设实验轨道。

论文的系统范围应写成：

> 本研究覆盖近期开放源码的非参数 task-memory systems，以及一个分离的 learned-policy subset。研究结论不直接推广到闭源系统、参数记忆或所有对话记忆产品。

## 13. 冻结前决策

- [ ] ReMe 是否能只启用 procedural-memory pipeline；
- [ ] ACE playbook entry 的稳定计数和版本如何记录；
- [ ] ReasoningBank 在 software session 中采用哪种公开 success/failure signal；
- [ ] MemRL 的 utility update 是否能在 episode 内运行；
- [ ] Agent-KB 是否支持从空 KB 开始的 session-level native write；
- [ ] MemOS 是否冻结为单一 plaintext/local profile，并通过 native write/retrieve/edit/delete qualification；
- [ ] SimpleMem 是否固定 text core commit，且能暴露 atomic object count、merge history 和 retrieval candidates；
- [ ] A-MEM、LightMem 和 MemoryOS 是否分别只进入对应的机制诊断，不被误列为 task-memory 主榜；
- [ ] MemAct 和 AgeMem 的公开 checkpoint、GPU 需求与许可证是否可接受；
- [ ] Track A 是否能统一 agent backbone；
- [ ] 正式投稿前重新抓取 stars、forks 和 latest push，并在论文中保留快照日期；
- [ ] 所有系统是否能提供 `stored → retrieved → visible → behavior` trace。

## 14. 三系统精简版建议

如果需要把本轮新增候选压缩为 **3 个 memory systems**，建议选择：**MemOS、A-MEM、Mem0**。这不是删除原有 Track A 的五个 task-memory systems，而是对本轮六个新增候选的精简 panel。

| 系统 | panel 角色 | 覆盖的核心问题 | 为什么保留 |
|---|---|---|---|
| MemOS | 主系统候选 | workspace 隔离、写入/编辑/删除、state evolution | 机制范围最完整，社区活跃度高；通过本地 profile qualification 后可进入主榜 |
| A-MEM | state-evolution 诊断 | 动态链接、旧状态共存、冲突和 drift | 能直接观察新 note 如何改变旧 memory 的链接、属性和可见性 |
| Mem0 | 工程成熟度与迁移基线 | fact consolidation、跨 session 使用、context 缺失时的退化 | 社区和部署成熟度最高，可作为 conversational-to-task transfer anchor |

这三个系统形成互补，而不是重复比较：MemOS 代表通用 memory OS，A-MEM 代表显式结构演化，Mem0 代表高采用率的生产型 fact memory。SimpleMem 和 LightMem 暂时放到 RQ5/效率补充实验；MemoryOS 放到分层对话 transfer 实验。

如果论文明确要求 **三个都必须是 task-memory 主榜系统**，可以将 Mem0 替换为 SimpleMem；但这会失去一个高成熟度、可复现的外部工程参照。就当前论文的两个核心 claim——state evolution/conflict resolution 和 long-horizon behavioral drift——我建议保留 Mem0 作为补充 anchor，而不是把它伪装成 task-memory 主系统。

## 15. 主要来源

- [Remember Me, Refine Me: A Dynamic Procedural Memory Framework for Experience-Driven Agent Evolution](https://arxiv.org/abs/2512.10696)
- [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)
- [ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](https://arxiv.org/abs/2509.25140)
- [MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory](https://arxiv.org/abs/2601.03192)
- [Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving](https://arxiv.org/abs/2507.06229)
- [Memory as Action: Autonomous Context Curation for Long-Horizon Agentic Tasks](https://arxiv.org/abs/2510.12635)
- [Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents](https://arxiv.org/abs/2601.01885)
- [MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](https://arxiv.org/abs/2506.15841)
- [MemEvolve: Meta-Evolution of Agent Memory Systems](https://arxiv.org/abs/2512.18746)
- [MemQ: Integrating Q-Learning into Self-Evolving Memory Agents over Provenance DAGs](https://arxiv.org/abs/2605.08374)
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)
- [Evaluating Very Long-Term Conversational Memory of LLM Agents](https://arxiv.org/abs/2402.17753)
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)；[official repository](https://github.com/agiresearch/A-mem)
- [MemOS: An Operating System for Memory-Augmented Generation (MAG) in Large Language Models](https://arxiv.org/abs/2505.22101)；[official repository](https://github.com/MemTensor/MemOS)
- [Memory OS of AI Agent](https://arxiv.org/abs/2506.06326)；[official repository](https://github.com/BAI-LAB/MemoryOS)
- [SimpleMem: Efficient Lifelong Memory for LLM Agents](https://arxiv.org/abs/2601.02553)；[official repository](https://github.com/aiming-lab/SimpleMem)
- [LightMem: Lightweight and Efficient Memory-Augmented Generation](https://arxiv.org/abs/2510.18866)；[official repository](https://github.com/zjunlp/LightMem)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)；[official repository](https://github.com/mem0ai/mem0)
