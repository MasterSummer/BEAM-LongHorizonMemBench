# 2026 Agent Memory Systems Survey

Date: 2026-07-06

Scope: systems and projects published or substantially updated after March 2026, with a focus on how each memory system writes, stores, retrieves, updates, and spends cost. This report separates peer-reviewed/preprint systems from production/open-source projects because their evidence quality differs.

## 1. Selection Criteria

Include a system if it satisfies at least one condition:

- Published after 2026-03-01.
- Major project release or public technical write-up after 2026-03-01.
- Useful design pattern for LHMSB efficiency/deviation evaluation.

Each system is analyzed by:

| Dimension | Question |
|---|---|
| Write | What gets stored and when? |
| Store | What substrate stores memory? |
| Read | How does the agent retrieve memory? |
| Update | Can memory be revised, decayed, deleted, or consolidated? |
| Cost | Where does compute/token/latency cost come from? |
| LHMSB relevance | How should LHMSB evaluate it? |

## 2. Quick Taxonomy

Current 2026 systems cluster into six patterns:

| Pattern | Core idea | Example systems |
|---|---|---|
| Lightweight layered memory | Split memory into short/mid/long-term layers, use cheap models and offline consolidation | LightMem |
| Graph/associative memory | Memories form links; retrieval expands through associations | HeLa-Mem, Mnemoverse |
| Self-evolving retrieval | System modifies its own retrieval/configuration based on failure logs | EvolveMem, OmniMem |
| Credit-assigned episodic memory | Memories get value through downstream contribution chains | MemQ |
| Case-based adaptation | Store past cases and select them with bandit/exploration logic | CASCADE |
| Production hybrid memory | BM25 + vector + graph + lifecycle controls, often exposed through MCP/API | agentmemory, Dakera, Supermemory |

## 3. Research Systems

### 3.1 LightMem

Source: [Lightweight LLM Agent Memory with Small Language Models](https://arxiv.org/abs/2604.07798), submitted 2026-04-09, ACL 2026.

Core idea:

LightMem uses Small Language Models to reduce online memory cost. It separates memory into short-term memory, mid-term memory, and long-term memory. Online retrieval uses a fixed retrieval budget: coarse vector retrieval first, then semantic consistency reranking. Offline consolidation abstracts reusable interaction evidence into long-term memory.

How it works:

| Component | Mechanism |
|---|---|
| Write | Raw interaction evidence enters short-term or mid-term memory. |
| Store | Layered STM/MTM/LTM memory. |
| Read | Vector coarse retrieval plus semantic consistency reranking. |
| Update | Offline consolidation integrates evidence into LTM. |
| Cost | SLM calls, embeddings, reranking latency, offline consolidation. |

LHMSB relevance:

LightMem maps cleanly to the **efficiency** axis. It explicitly trades memory quality against online latency and model cost. LHMSB should test whether its offline consolidation improves cross-session task probes without causing stale-memory deviation.

Suggested LHMSB probes:

- Cross-session recall after consolidation.
- Update probes where an older MTM item conflicts with a newer LTM fact.
- Cost sensitivity: online retrieval latency vs task gain.

### 3.2 HeLa-Mem

Source: [HeLa-Mem: Hebbian Learning and Associative Memory for LLM Agents](https://arxiv.org/abs/2604.16839), submitted 2026-04-18, accepted to ACL 2026.

Core idea:

HeLa-Mem models memory as a dynamic graph. It has an episodic memory graph that strengthens through co-activation and a semantic memory store populated through Hebbian distillation. A reflective agent identifies dense memory hubs and distills them into reusable semantic knowledge.

How it works:

| Component | Mechanism |
|---|---|
| Write | Add episodic memories as graph nodes; co-activated memories strengthen edges. |
| Store | Dynamic episodic graph plus semantic store. |
| Read | Retrieval combines semantic similarity and learned associations. |
| Update | Reflective distillation turns dense graph hubs into semantic knowledge. |
| Cost | Graph maintenance, reflection/distillation calls, retrieval expansion. |

LHMSB relevance:

HeLa-Mem is useful for testing whether associative retrieval helps multi-hop or cross-session probes. It may also increase deviation risk if associations pull in obsolete memories.

Suggested LHMSB probes:

- Multi-fact research synthesis.
- Stale association probes where a formerly associated fact has been retracted.
- Deviation index by stale-fact citation.

### 3.3 EvolveMem

Source: [EvolveMem: Self-Evolving Memory Architecture via AutoResearch for LLM Agents](https://arxiv.org/abs/2605.13941), submitted 2026-05-13.

Core idea:

EvolveMem treats the memory system's retrieval configuration as a structured action space. It reads failure logs, diagnoses root causes, proposes configuration changes, and applies them with safeguards such as revert-on-regression and explore-on-stagnation.

How it works:

| Component | Mechanism |
|---|---|
| Write | Stores interaction/memory evidence similar to SimpleMem. |
| Store | Underlying SimpleMem-style memory plus tunable retrieval configuration. |
| Read | Retrieval behavior changes as configuration evolves. |
| Update | LLM-powered diagnosis module mutates retrieval settings. |
| Cost | Failure-log analysis, evolution rounds, benchmark reruns, LLM diagnosis. |

LHMSB relevance:

EvolveMem challenges static benchmark design. If the memory system evolves over evaluation, LHMSB must decide whether evolution cost counts. For this project, it should count as memory lifecycle cost.

Suggested LHMSB probes:

- Compare pre-evolution vs post-evolution ROI.
- Count diagnosis/evolution tokens in CostVector.
- Test transfer: tune on research, evaluate on software without retuning.

### 3.4 OmniMem / Omni-SimpleMem

Source: [OmniMem: Autoresearch-Guided Discovery of Lifelong Multimodal Agent Memory](https://arxiv.org/abs/2604.01007), submitted 2026-04-01.

Core idea:

OmniMem extends a text memory system into multimodal lifelong memory through an autonomous research loop. The system iteratively modifies architecture, retrieval, prompts, and data pipelines across text, image, audio, and video memory.

How it works:

| Component | Mechanism |
|---|---|
| Write | Ingests multimodal experiences. |
| Store | Multimodal memory stores, with text and non-text representations. |
| Read | Modality-aware retrieval and answer generation. |
| Update | Architecture/configuration discovered through autoresearch. |
| Cost | Multimodal encoders, LLM/VLM calls, autoresearch experiments. |

LHMSB relevance:

OmniMem is less central for v1 because LHMSB currently focuses on text/code tasks. It becomes important if a future family adds screenshots, figures, or audio notes.

Suggested LHMSB use:

- Keep as v2 candidate.
- Do not include in v1 main experiment unless adding multimodal family.

### 3.5 MemQ

Source: [MemQ: Integrating Q-Learning into Self-Evolving Memory Agents over Provenance DAGs](https://arxiv.org/abs/2605.08374), submitted 2026-05.

Core idea:

MemQ assigns value to memories through downstream contribution. It records a provenance DAG: which memories were retrieved when a new memory or action was produced. TD(lambda)-style credit propagates backward through the DAG so useful upstream memories gain higher Q-values.

How it works:

| Component | Mechanism |
|---|---|
| Write | Store episodic memories with provenance links. |
| Store | Memory DAG with Q-values. |
| Read | Retrieve memories by relevance and learned value. |
| Update | Propagate credit backward through provenance chains. |
| Cost | DAG maintenance, Q-value updates, retrieval scoring. |

LHMSB relevance:

MemQ is a strong match for LHMSB because it explicitly connects memory utility to task outcomes, not only retrieval relevance. It should be evaluated by task score and ROI, not only precision@k.

Suggested LHMSB probes:

- Multi-step software episodes where early memories enable later correct actions.
- Compare retrieval precision vs task gain.
- Check whether high-Q stale memories cause deviation after retraction.

### 3.6 CASCADE

Source: [CASCADE: Case-Based Continual Adaptation for Large Language Models During Deployment](https://arxiv.org/abs/2605.06702), submitted 2026-05-05.

Core idea:

CASCADE frames deployment-time learning as case-based continual adaptation. It stores past cases and uses contextual bandit logic to decide which cases to reuse, balancing exploration and exploitation.

How it works:

| Component | Mechanism |
|---|---|
| Write | Store deployment cases and outcomes. |
| Store | Explicit evolving episodic memory of cases. |
| Read | Select relevant cases using contextual bandit policy. |
| Update | Update case utility based on success/failure. |
| Cost | Case selection, policy updates, added context tokens. |

LHMSB relevance:

CASCADE fits the **efficiency** axis through deployment-time improvement. It also needs deviation testing because old successful cases may become wrong when the environment changes.

Suggested LHMSB probes:

- Reuse old software cases after API retraction.
- Measure whether case reuse helps stable tasks but hurts changed tasks.
- Count policy/case selection cost.

### 3.7 Mem-pi

Source: [Mem-pi: Adaptive Memory through Learning When and What to Generate](https://arxiv.org/abs/2605.21463), submitted 2026-05-20.

Core idea:

Mem-pi does not retrieve static memory entries. It uses a dedicated model to decide whether to generate context-specific guidance and what to generate. It can abstain when guidance would not help.

How it works:

| Component | Mechanism |
|---|---|
| Write | Learns/generates guidance policy rather than storing static entries. |
| Store | Dedicated memory/guidance model parameters. |
| Read | Generate guidance conditioned on current context. |
| Update | Trained with decision-content decoupled RL. |
| Cost | Additional model inference, RL training, generated guidance tokens. |

LHMSB relevance:

Mem-pi breaks the normal adapter assumption because memory is generated rather than retrieved. LHMSB can still evaluate it if the adapter exposes `search()` as "generate guidance" and records model cost.

Suggested LHMSB probes:

- Abstention: when should memory not be used?
- Deviation: does generated guidance preserve current constraints?
- Cost: extra model inference per probe.

### 3.8 WorldEvolver

Source: [Self-Evolving World Models for LLM Agent Planning](https://arxiv.org/abs/2606.30639), submitted 2026-06.

Core idea:

WorldEvolver uses memory to improve world-model planning. It has episodic memory for retrieved real action transitions, semantic memory for persistent heuristic rules, and selective foresight that filters low-confidence predictions.

How it works:

| Component | Mechanism |
|---|---|
| Write | Store action transitions and prediction-observation mismatches. |
| Store | Episodic transition memory plus semantic rule memory. |
| Read | Retrieve transitions to simulate future action consequences. |
| Update | Extract heuristic rules from mismatches. |
| Cost | Retrieval simulation, rule extraction, foresight filtering. |

LHMSB relevance:

WorldEvolver targets planning rather than factual memory. It is useful if LHMSB adds interactive worlds in v2. For v1 fixed-world tasks, it is a less direct fit.

### 3.9 HyphaeDB

Source: [HyphaeDB: A Living Knowledge Topology for Agent-First Memory](https://arxiv.org/html/2606.28781v1), March/June 2026.

Core idea:

HyphaeDB reinterprets HNSW not just as a vector search index but as a communication fabric for multi-agent knowledge propagation. Agents are nodes in vector space, knowledge spreads through gossip-like neighbor propagation with energy attenuation, and the system aims for contradiction detection and consensus formation.

How it works:

| Component | Mechanism |
|---|---|
| Write | Add knowledge nodes and memory diffs. |
| Store | HNSW-like graph topology over agents and knowledge. |
| Read | Retrieve through topology and propagated knowledge. |
| Update | Knowledge propagates and may be promoted through consensus. |
| Cost | Graph propagation, consensus/promotion, storage and retrieval. |

LHMSB relevance:

HyphaeDB is relevant for multi-agent memory. It is not necessary for v1 unless LHMSB adds multi-agent tasks. It gives useful design ideas for future deviation metrics: contradiction propagation and consensus can amplify stale memory.

## 4. Production and Open-Source Systems

### 4.1 agentmemory

Source: [agentmemory GitHub](https://github.com/rohitg00/agentmemory), active 2026.

Core idea:

agentmemory is a persistent memory server for coding agents. It captures agent activity through hooks, compresses observations, builds structured facts, indexes memories with BM25/vector/graph retrieval, and injects relevant context at session start.

How it works:

| Component | Mechanism |
|---|---|
| Write | Hooks capture prompts, tool use, file access, failures, and session summaries. |
| Store | Raw observations, episodic summaries, semantic facts, procedural workflows. |
| Read | Hybrid BM25 + vector + graph retrieval with RRF fusion. |
| Update | Versioning, supersession, decay, auto-forget, contradiction detection. |
| Cost | LLM compression, embedding, graph extraction, retrieval, context injection. |

LHMSB relevance:

This is a strong candidate for the software family because it targets coding agents. Its hook-based capture may not map directly onto the current LHMSB adapter interface, so a wrapper should record observations through `add_memory()` and call smart search through `search()`.

Suggested LHMSB probes:

- Deprecated API use.
- File-specific memory recall.
- Project convention changes.
- Cost of compression vs task gain.

### 4.2 Supermemory

Source: [Supermemory GitHub](https://github.com/supermemoryai/supermemory), active 2026.

Core idea:

Supermemory is a memory and context engine. It stores content, builds user profiles, extracts static and dynamic profile memory, and supports hybrid search across memories and documents.

How it works:

| Component | Mechanism |
|---|---|
| Write | `add()` stores text, conversations, URLs, HTML, documents. |
| Store | User/container-scoped memory and document indexes. |
| Read | `profile()` returns static profile, dynamic profile, and search results; hybrid search can combine RAG and memory. |
| Update | Settings configure extraction and chunking; document APIs manage sources. |
| Cost | Managed extraction, indexing, hybrid search, profile generation. |

LHMSB relevance:

Supermemory is useful as a managed production baseline. It may be hard to instrument internal costs unless the API exposes usage. For LHMSB, record visible API calls, latency, returned context tokens, and any provider usage metadata.

Suggested LHMSB probes:

- User/project preference memory.
- Dynamic profile update after contradiction.
- Compare profile memory vs raw retrieval.

### 4.3 Mnemoverse

Source: [Mnemoverse](https://mnemoverse.com/), active 2026.

Core idea:

Mnemoverse exposes persistent memory through API and MCP. It presents memory as atoms and associations, supports write/read/feedback/stats/delete tools, and emphasizes graph-like recall paths.

How it works:

| Component | Mechanism |
|---|---|
| Write | MCP tool `memory_write` or API call writes memory atoms. |
| Store | Memory graph of atoms and associations. |
| Read | Query activates associated concepts and traverses links. |
| Update | Feedback and delete tools; domain delete for broader cleanup. |
| Cost | API calls, graph traversal, storage, possible managed processing. |

LHMSB relevance:

Mnemoverse is useful for deviation because it supports delete and graph associations. It should be tested on retraction-heavy research episodes.

Suggested LHMSB probes:

- Does `memory_delete` remove influence of retracted facts?
- Does associative recall pull stale facts back into context?
- How expensive is graph traversal vs task gain?

### 4.4 Dakera

Source: [Dakera technical write-up](https://dakera.ai/blog/how-agent-memory-works), May 2026.

Core idea:

Dakera uses hybrid retrieval with HNSW vector search, BM25 full text, hybrid scoring, and importance decay. It emphasizes production failures of vector-only memory: recall precision, temporal relevance, importance weighting, and cross-session continuity.

How it works:

| Component | Mechanism |
|---|---|
| Write | Store agent memories with metadata and importance. |
| Store | HNSW vector index, BM25 full-text index, lifecycle states. |
| Read | Hybrid scoring/fusion, reranking, decay-aware retrieval. |
| Update | Importance decay, consolidation, archive lifecycle. |
| Cost | Embedding, HNSW search, BM25 search, reranking, consolidation. |

LHMSB relevance:

Dakera is an excellent baseline for the efficiency axis because it represents a strong production hybrid retrieval system. It also supports deviation tests through importance decay and archival behavior.

Suggested LHMSB probes:

- Exact API names and code identifiers where BM25 should beat vector-only search.
- Recency vs old but still valid facts.
- Stale fact retention after retraction.

## 5. What This Means for LHMSB

### 5.1 Systems to Prioritize

For the first full experiment, prioritize systems that cover distinct architecture families:

| Priority | System | Reason |
|---|---|---|
| Must | no_memory | Counterfactual baseline |
| Must | BM25 | Cheap lexical baseline |
| Must | Chroma/vector | Standard dense retrieval baseline |
| Must | hybrid BM25+vector | Strong cheap baseline |
| High | LightMem | Efficiency-focused layered memory |
| High | agentmemory or Dakera | Production hybrid lifecycle memory |
| High | HeLa-Mem or Mnemoverse | Graph/associative memory |
| Medium | EvolveMem | Self-evolving retrieval, but evaluation cost is complex |
| Medium | CASCADE/MemQ | Strong research ideas, may require custom adapter |
| Later | OmniMem/WorldEvolver/HyphaeDB | More relevant to multimodal/planning/multi-agent v2 |

### 5.2 Adapter Requirements

Each candidate adapter should expose:

```python
initialize(user_id, session_id=None)
reset(user_id)
add_memory(content, user_id, session_id=None, metadata=None)
search(query, user_id, session_id=None, top_k=10, **filters)
update_memory(memory_id, content=None, metadata=None)
delete_memory(memory_id)
```

For systems without native update/delete, the adapter should report unsupported capability rather than hiding it. Deviation experiments need to know whether stale memories persist because the system cannot delete, because it failed to retrieve current facts, or because the agent ignored them.

### 5.3 Cost Fields to Record

For these 2026 systems, LHMSB should record:

- write latency
- retrieval latency
- update/delete latency
- embedding calls/tokens
- internal LLM input/output tokens
- reflection/consolidation tokens
- evolution/diagnosis tokens
- storage bytes
- returned context tokens

For managed APIs where internal token counts are unavailable, record:

- API call count
- wall-clock latency
- returned token count
- reported usage metadata if provided
- mark internal cost as `unobserved` in diagnostics

### 5.4 Efficiency Questions

For each system, ask:

1. Does memory improve task score over no_memory?
2. Does memory improve cross-session utilization?
3. Does retrieval quality predict task score?
4. Does the gain justify memory-attributable cost?
5. Does a cheap hybrid baseline beat complex LLM-heavy memory?

### 5.5 Deviation Questions

For each system, ask:

1. Does it surface retracted facts?
2. Does it keep using superseded API/requirements?
3. Does graph/association retrieval pull stale neighbors into context?
4. Does consolidation preserve validity windows?
5. Does update/delete really remove behavioral influence?

## 6. Immediate Next Steps

1. Build a short candidate list for smoke and pilot:
   - no_memory
   - BM25
   - Chroma
   - hybrid BM25+vector
   - fake_perfect
   - fake_bad
   - one production memory: agentmemory, Dakera, Supermemory, or Mnemoverse

2. Add a survey table to the paper draft:
   - columns: system, date, write, store, read, update, cost visibility, LHMSB relevance.

3. Add adapter feasibility notes:
   - Does the system have Python API?
   - Does it have REST/MCP?
   - Can it run locally?
   - Can costs be observed?
   - Does it support delete/update?

4. Use research systems to shape metrics:
   - LightMem motivates online/offline cost separation.
   - HeLa-Mem motivates associative stale retrieval tests.
   - MemQ motivates task-outcome-based memory utility.
   - EvolveMem motivates counting evolution as lifecycle cost.
   - Mem-pi motivates abstention and generated-memory evaluation.

## 7. Sources

- LightMem: https://arxiv.org/abs/2604.07798
- HeLa-Mem: https://arxiv.org/abs/2604.16839
- EvolveMem: https://arxiv.org/abs/2605.13941
- OmniMem: https://arxiv.org/abs/2604.01007
- MemQ: https://arxiv.org/abs/2605.08374
- CASCADE: https://arxiv.org/abs/2605.06702
- Mem-pi: https://arxiv.org/abs/2605.21463
- WorldEvolver: https://arxiv.org/abs/2606.30639
- HyphaeDB: https://arxiv.org/html/2606.28781v1
- agentmemory: https://github.com/rohitg00/agentmemory
- Supermemory: https://github.com/supermemoryai/supermemory
- Mnemoverse: https://mnemoverse.com/
- Dakera: https://dakera.ai/blog/how-agent-memory-works
