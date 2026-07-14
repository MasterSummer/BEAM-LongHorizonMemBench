# 长程任务记忆系统 Benchmark 调研

> 调研截止：2026-07-12
>
> 目标：梳理长程记忆 benchmark 的任务范式、评测指标和常用 memory systems，并据此给出本项目 LongHorizonMemSysBench（LHMSB）的系统选型建议。

## 1. 结论摘要

当前工作大致分为两条路线：

1. **Memory recall / QA**：测试系统能否从很长的历史中找回事实并回答问题。代表工作有 LoCoMo、LongMemEval、MemBench、MemoryAgentBench、BEAM 和 EverMemBench。
2. **Agentic memory / task utility**：测试记忆是否真正改变后续 agent 的决策、计划和任务完成结果。代表工作有 AMA-Bench、MemoryArena、LongMemEval-V2 和 EMemBench。

本项目目前的多 session、任务完成、信息撤回、行为漂移和全生命周期成本设计，最接近 **MemoryArena + AMA-Bench**，而不是传统的 LoCoMo/LongMemEval QA benchmark。建议将 QA benchmark 作为 supporting diagnostics，将 task completion、goal drift 和 cost/ROI 作为主结果。

另外，公开领域已经存在一个 ICLR 2026 的 **BEAM（Beyond a Million Tokens）** benchmark，覆盖 128K–10M token 长对话。因此项目公开发布时不建议继续使用 BEAM 作为主缩写，推荐使用 **LHMSB** 或 **LongHorizonMemSysBench**，避免命名冲突。

## 2. 相关 benchmark 工作

| Benchmark | 核心设置 | 主要能力/指标 | 主要 memory baseline 或方法 | 与 LHMSB 的关系 |
|---|---|---|---|---|
| [LoCoMo](https://aclanthology.org/2024.acl-long.747/) | 10 条超长多 session 对话，平均约 588 turns | single-hop、multi-hop、temporal、open-domain、adversarial、event summarization | full-context LLM、long-context LLM、RAG over dialog/observation/summary | 适合作为传统 recall 对照，但不测真正的 task completion |
| [LongMemEval](https://arxiv.org/abs/2410.10813) | 500 个问题，用户-助手长期交互历史 | information extraction、multi-session reasoning、temporal reasoning、knowledge update、abstention | session decomposition、fact-augmented key、time-aware query expansion、RAG | 支持 knowledge update，但整体仍是 QA 范式 |
| [MemBench](https://aclanthology.org/2025.findings-acl.989/) | factual/reflective memory，participation/observation 两种场景 | accuracy、recall、capacity、temporal efficiency | FullMemory、RetrievalMemory、RecentMemory、GenerativeAgent、MemoryBank、MemGPT、SCMemory | 说明简单 retrieval baseline 和 capacity/efficiency 不能缺失 |
| [MemoryAgentBench](https://arxiv.org/abs/2507.05257) | 增量 multi-turn，包含 EventQA、FactConsolidation 等数据 | accurate retrieval、test-time learning、long-range understanding、conflict resolution | Mem0、Letta、Cognee、Zep、HippoRAG、MemoRAG、GraphRAG、RAPTOR、Self-RAG、embedding retriever | 与统一 adapter 接口较接近，可作为系统兼容性参考 |
| [BEAM](https://github.com/mohammadtavakoli78/BEAM) | 100 条对话，128K/500K/1M/10M token，2,000 个问题 | 10 类能力：abstention、contradiction、event ordering、update、instruction following、preference 等 | LIGHT：episodic memory、working memory、scratchpad | 主要测 token-scale 长程能力，不直接测 agent task utility |
| [EverMemBench](https://arxiv.org/abs/2602.01313) | 多人、多群组、跨主题、时间演化，超过 1M tokens | fine-grained recall、memory awareness、user profile understanding | 重点是复杂多人记忆场景 | 可作为未来 shared/group memory 扩展，不是当前单 agent 主线 |
| [AMA-Bench](https://arxiv.org/abs/2602.22769) | 真实和合成 agent trajectories，覆盖 Web、SWE、Text2SQL、Gaming、Embodied AI 等 | 从 trajectory 构造 memory，再检索证据并回答；支持任意 horizon | BM25、embedding memory、long-context、AMA-Agent | 与当前 research/software trajectory 设计最接近的工作之一 |
| [MemoryArena](https://arxiv.org/abs/2602.16313) | 多 session，后续子任务依赖早期行动和反馈 | end-to-end task success、跨 session 决策复用 | Letta、Mem0、MIRIX、Mem0-g、ReasoningBank、BM25、embedding、MemoRAG、GraphRAG | 当前最重要的对标工作，强调 memory-agent-environment loop |
| [LongMemEval-V2](https://arxiv.org/abs/2605.12493) | WebArena/WorkArena trajectories，最大约 115M tokens | static state、dynamic state、workflow、environment gotchas、premise awareness | no retrieval、RAG raw slices、RAG + notes、AgentRunbook-R/Codex | 可参考其 accuracy-latency frontier 和超大 haystack 设置 |
| [EMemBench](https://arxiv.org/abs/2601.16690) | 文本和视觉交互游戏，问题从 agent 自己的 trajectory 生成 | single/multi-hop、induction、temporal、spatial、logical、adversarial | context-only 与 persistent episodic memory agents | 适合未来增加 interactive/visual environment，不是 v1 必需 |
| [Cost and Accuracy of LTM](https://arxiv.org/abs/2601.07978) | LoCoMo 上的系统级成本和准确率比较 | accuracy、latency、CPU、RAM、disk I/O、network、TCO、Pareto | mem0、Graphiti、Cognee、RAG、full-context | 证明 cost-aware memory evaluation 可操作，但仍是 QA 而非 task completion |
| [LongMemCode](https://argosbrain.com/papers/longmemcode-benchmark) | 真实代码库上的结构化代码查询 | deterministic accuracy、P50/P95/P99 latency、cost/query | grep baseline、结构化 graph reference、第三方 adapter | 可为 software family 提供独立 code-memory retrieval diagnostics |

### 2.1 主要趋势

- LoCoMo、LongMemEval 等早期工作主要回答“能否召回正确事实”。
- MemoryAgentBench 开始把 memory 能力拆成 retrieval、test-time learning、long-range understanding 和 conflict resolution。
- AMA-Bench、MemoryArena 将 memory 放入 agent trajectory 和 environment loop，直接测后续行为和任务成功。
- LongMemEval-V2、LongMemCode 和 Cost/Accuracy-LTM 开始显式报告 latency、token、系统资源和准确率 frontier。
- 当前仍缺少一个同时覆盖 **task utility、staleness/drift、full-lifecycle cost、retrieval quality 和 capability differences** 的统一 benchmark，这正是 LHMSB 的主要空间。

## 3. Memory systems 分类

### 3.1 基础对照系统

这些系统不是“强 memory system”，但必须存在，否则无法解释复杂系统的增益来源：

| System | 作用 |
|---|---|
| `no_memory` | 跨 session 不保留状态，作为 counterfactual baseline |
| `full_context` | 尽可能保留全部历史，作为长上下文上界/高成本参照 |
| `recent_context` | 只保留最近若干轮，测试简单 recency 策略 |
| `bm25` | 低成本、可解释的 lexical retrieval；对代码符号、API 名称和 exact facts 很重要 |
| `dense_vector` | 标准 embedding retrieval |
| `hybrid_bm25_vector` | BM25 + dense vector 的透明混合 baseline |

### 3.2 生产级 memory systems

#### Mem0

典型机制：

- 写入时进行 fact extraction 和去重；
- vector storage + BM25 keyword + entity matching；
- 通过 entity linking 提升跨事实检索；
- 当前官方文档描述的新算法是 single-pass、ADD-only，旧事实和新事实可以同时保留。

参考：[Mem0 memory evaluation](https://docs.mem0.ai/core-concepts/memory-evaluation)、[Mem0 migration notes](https://docs.mem0.ai/open-source/features/graph-memory)。

对 LHMSB 的价值：适合作为主榜单中的 hybrid production baseline，同时能暴露“没有显式 update/delete 时，系统是否能正确处理 retraction”。

#### Letta / MemGPT

典型机制：

- persistent memory blocks，始终位于 agent context；
- archival memory，通过工具按需检索；
- agent 自己编辑 memory block；
- sleeptime/background agent 进行反思或整理。

参考：[Letta context hierarchy](https://docs.letta.com/guides/core-concepts/memory/context-hierarchy)、[Letta memory blocks](https://docs.letta.com/guides/core-concepts/memory/memory-blocks)。

对 LHMSB 的价值：这是 agent-integrated memory，不是单纯的 retrieval backend，适合测试主动记忆使用、行为稳定性和上下文成本。

#### Graphiti / Zep

Graphiti 是开源 temporal knowledge graph，Zep 是其托管生产系统。典型机制包括：

- entity/edge extraction；
- bi-temporal facts；
- fact invalidation 和历史保留；
- semantic、full-text、graph、time fusion retrieval。

参考：[Graphiti overview](https://help.getzep.com/graphiti/getting-started/overview)、[Zep vs Graphiti](https://help.getzep.com/zep-vs-graphiti)。

对 LHMSB 的价值：特别适合 research family 中的 change/retract、时间查询和证据图。主榜单建议使用 Graphiti；Zep 可放到 managed-service secondary track，因为内部成本通常不可完全观测。

#### Cognee

典型机制是多阶段知识构建流水线：写入时进行 chunk、entity extraction、图谱构建，之后通过 memify/improve 等过程重组织知识。

对 LHMSB 的价值：适合作为 graph/vector pipeline 代表。需要单独计入 cognify、memify、improve 等内部 LLM、embedding 和 storage 成本。

#### Hindsight

Hindsight 将长期记忆划分为 world、experience、observation、opinion 四类网络，并提供 retain、recall、reflect 三个核心操作。它强调事实、经验和 belief 的分离以及可追溯反思。

参考：[Hindsight paper](https://arxiv.org/abs/2512.12818)、[Hindsight GitHub](https://github.com/vectorize-io/hindsight)。

对 LHMSB 的价值：这是当前系统列表中比较明显的缺口，尤其适合测试“当前证据”和“agent 推断”混淆、retraction 后旧 belief 继续影响行为等问题。

### 3.3 研究型 memory systems

| System | 主要机制 | 建议 |
|---|---|---|
| [SimpleMem](https://arxiv.org/abs/2601.02553) | semantic structured compression、recursive consolidation、query-aware retrieval | 作为 efficiency/ROI research track |
| [ReasoningBank](https://arxiv.org/abs/2509.25140) | 从成功和失败 trajectory 中提炼可迁移策略和 guardrails | 作为 experience-memory track，不与普通 backend 直接混榜 |
| [A-MEM](https://arxiv.org/abs/2502.12110) | Zettelkasten 风格的动态索引、链接和 memory evolution | 作为 agentic/associative memory ablation |
| [MemoryOS](https://arxiv.org/abs/2506.06326) | short-term、mid-term、long-term 分层存储和更新 | 可作为 hierarchical memory 对照 |
| [MIRIX](https://arxiv.org/abs/2507.07957) | Core、Episodic、Semantic、Procedural、Resource、Knowledge Vault 六类 memory | 更适合 multimodal 或 shared-memory v2 |
| LightMem | short/mid/long-term 分层，SLM online processing 和 offline consolidation | 可作为低成本 memory 方向，暂不列入 v1 主榜单 |

研究型系统通常会改变 extraction prompt、reflection loop 或 agent policy，因此更适合单独报告，而不是和 Mem0/Chroma 这种 backend 直接混合排名。

## 4. 对 LHMSB 的系统选型建议

### 4.1 主榜单

建议主榜单扩展为：

| Priority | Condition | 说明 |
|---|---|---|
| Must | `no_memory` | counterfactual baseline |
| Must | `full_context` / `recent_context` | 长上下文和 recency 对照 |
| Must | `bm25` | 低成本 lexical floor |
| Must | `dense_vector` | 标准 dense retrieval |
| Must | `hybrid_bm25_vector` | 透明、低成本的强 baseline |
| Must | `mem0` | 生产级 hybrid memory |
| Must | `letta` | hierarchical/self-editing memory |
| Must | `graphiti` | temporal knowledge graph |
| High | `cognee` | graph/vector self-reorganizing pipeline |
| High | `hindsight` | provenance/faith-vs-belief memory |

现有仓库已经实现 `no_memory`、`chroma`、`mem0`、`letta`、`graphiti`、`cognee`，见 [`src/05-systems.md`](src/05-systems.md)。整体系统家族覆盖是合理的，但建议补充透明 baseline 和 Hindsight。

### 4.2 附加研究榜单

- `simplemem`：检验压缩和低 token 是否带来更高 Memory ROI；
- `reasoningbank`：检验失败经验是否能改善后续 software/research task；
- `a_mem`：检验 associative linking 和 memory evolution；
- `memoryos` 或 `lightmem`：检验层级 memory 和 offline consolidation。

### 4.3 校准条件

保留现有：

- `fake_perfect`：只返回当前有效事实；
- `fake_bad`：返回错误或已撤回事实。

建议再增加：

- `full_context_oracle`；
- `current_facts_only`；
- `stale_facts_only`。

这样可以区分 retrieval failure、stale-memory contamination 和 agent utilization failure。

## 5. 对当前设计的具体修正建议

### 5.1 Chroma 命名和 embedding

当前 [`src/lhmsb/adapters/chroma.py`](src/lhmsb/adapters/chroma.py) 使用 deterministic hashing bag-of-words embedding。它适合离线、可复现 smoke test，但不等价于标准 dense embedding。

建议：

- 将其标注为 `hash_vector` 或 `offline_vector`；
- 增加一个真正的本地 e5/BGE embedding 条件，名称为 `dense_vector`；
- 另外实现 `bm25` 和 `hybrid_bm25_vector`，避免把 lexical overlap 误称为 dense retrieval。

### 5.2 Capability 不应只描述 CRUD

现有 adapter 的 `supports_add/search/update/delete` 设计是必要的，但建议增加以下能力字段：

- `supports_temporal_validity`；
- `supports_provenance`；
- `supports_retraction`；
- `supports_reflection`；
- `supports_explicit_agent_tools`；
- `supports_internal_model_pinning`。

这样可以解释 Mem0 的 ADD-only、Graphiti 的 temporal invalidation、Letta 的 block overwrite 之间的差异。

### 5.3 增加规模 stress tier

当前 research episode 约为 3–6 sessions、15–40 facts，适合测 retraction 和 drift，但不覆盖 BEAM/LME-V2 的超长上下文规模。

建议增加独立的规模 tier，例如：

- 10K tokens；
- 100K tokens；
- 500K tokens。

通过插入不相关噪声，分别测量 retrieval degradation、capacity、latency 和 cost。该 tier 不必进入主 ROI，可作为 scalability appendix。

### 5.4 重新表述 novelty

不建议直接声称“第一个 cost-aware memory benchmark”。已有工作已经分别覆盖 TCO、accuracy-latency frontier 和 token efficiency，例如 [Cost and Accuracy of LTM](https://arxiv.org/abs/2601.07978) 和 [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2)。

更稳妥的贡献表述是：

> LHMSB 在统一的 counterfactual replay protocol 中，联合评估 memory 对跨 session task completion 的增益、goal/behavioral drift、retrieval quality 和 full-lifecycle cost，并通过 Memory ROI 将性能收益与生命周期成本关联起来。

## 6. 推荐的论文定位

可以将相关工作分为四段：

1. **Conversational memory benchmarks**：LoCoMo、LongMemEval、MemBench；
2. **Incremental and agent memory benchmarks**：MemoryAgentBench、BEAM、EverMemBench；
3. **Agentic task memory benchmarks**：AMA-Bench、MemoryArena、LongMemEval-V2；
4. **System-level cost and code-memory benchmarks**：Cost/Accuracy-LTM、LongMemCode。

LHMSB 的定位应强调：它不追求最长上下文或最高 QA recall，而是研究在 evolving procedural tasks 中，memory system 是否带来可验证的 task-level gain，以及该 gain 是否值得付出完整生命周期成本。

## 7. 参考资料

### Benchmarks

- [LoCoMo: Evaluating Very Long-Term Conversational Memory of LLM Agents](https://aclanthology.org/2024.acl-long.747/)
- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [MemBench](https://aclanthology.org/2025.findings-acl.989/)
- [MemoryAgentBench](https://arxiv.org/abs/2507.05257)
- [MemoryAgentBench GitHub](https://github.com/HUST-AI-HYZ/MemoryAgentBench)
- [BEAM: Beyond a Million Tokens](https://github.com/mohammadtavakoli78/BEAM)
- [EverMemBench](https://arxiv.org/abs/2602.01313)
- [AMA-Bench](https://arxiv.org/abs/2602.22769)
- [AMA-Bench GitHub](https://github.com/AMA-Bench/AMA-Bench)
- [MemoryArena](https://arxiv.org/abs/2602.16313)
- [LongMemEval-V2](https://arxiv.org/abs/2605.12493)
- [LongMemEval-V2 GitHub](https://github.com/xiaowu0162/LongMemEval-V2)
- [EMemBench](https://arxiv.org/abs/2601.16690)
- [Cost and Accuracy of Long-Term Memory](https://arxiv.org/abs/2601.07978)
- [LongMemCode](https://argosbrain.com/papers/longmemcode-benchmark)

### Memory systems

- [Mem0](https://arxiv.org/abs/2504.19413)
- [Mem0 evaluation docs](https://docs.mem0.ai/core-concepts/memory-evaluation)
- [Letta context hierarchy](https://docs.letta.com/guides/core-concepts/memory/context-hierarchy)
- [Graphiti overview](https://help.getzep.com/graphiti/getting-started/overview)
- [Hindsight](https://arxiv.org/abs/2512.12818)
- [SimpleMem](https://arxiv.org/abs/2601.02553)
- [ReasoningBank](https://arxiv.org/abs/2509.25140)
- [A-MEM](https://arxiv.org/abs/2502.12110)
- [MemoryOS](https://arxiv.org/abs/2506.06326)
- [MIRIX](https://arxiv.org/abs/2507.07957)
