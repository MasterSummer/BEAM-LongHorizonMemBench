# LHMSB Benchmark Development Timeline

## 1. 项目目标

本项目要做一个长期记忆系统 benchmark。它评测的不是单纯的“能不能检索到旧信息”，而是两个问题：

1. **Efficiency**：记忆系统存下来的信息，能不能在任务需要时帮 agent 做对事，并且成本是否划算。
2. **Deviation**：记忆系统会不会让 agent 被旧信息、错误信息或过期目标带偏。

当前项目已经有一个可运行原型：能生成多 session 任务，能接不同 memory system，能自动评分，能记录成本，能输出 scorecard。下一步要把它推进成可以投稿的 benchmark paper。

## 2. 总体时间线

建议把工作分成 7 个阶段：

1. 调研 memory systems 和 memory datasets
2. 确定 benchmark 架构和核心指标
3. 构造数据集并校准指标
4. smoke run 验证评测链路
5. pilot run 小规模真实实验
6. full run 全量实验
7. 论文 draft、图表和 appendix

原始设想是：

```text
调研 -> 定架构 -> smoke run -> 全量实验 -> 论文 draft
```

建议改成：

```text
调研 -> 定架构/指标 -> 数据集与指标校准 -> smoke run -> pilot run -> full run -> 论文
```

新增的两个阶段很重要：

- **数据集与指标校准**：确认 benchmark 真能测到 efficiency 和 deviation。
- **pilot run**：在 full run 前发现真实 backend、adapter、成本统计和超时问题。

## 3. 阶段一：调研 Memory Systems 和 Datasets

### 目标

弄清楚已有 memory system 怎么工作，已有 benchmark 怎么构造数据，避免重复已有工作。

### 调研 Memory Systems

需要覆盖：

- no-memory baseline
- full-context baseline
- BM25 memory
- dense vector memory
- hybrid BM25 + dense memory
- summary memory
- Chroma
- Mem0
- Letta / MemGPT
- Graphiti
- Cognee

每个系统要回答：

| 问题 | 用途 |
|---|---|
| 怎么写入记忆 | 决定 cost 怎么统计 |
| 怎么检索记忆 | 决定 retrieval 怎么评测 |
| 能否更新/删除旧记忆 | 关系到 stale memory 和 deviation |
| 是否有 reflection/summarization | 关系到 memory lifecycle cost |
| 是否调用内部 LLM | 关系到成本归因 |
| 是否支持 session/user scope | 关系到实验隔离 |

### 调研 Existing Benchmarks

需要重点看：

- LongMemEval
- LoCoMo
- MemoryAgentBench
- STALE
- BEAM / LIGHT
- LongMemEval-V2
- AgentBench
- SWE-bench

每个 benchmark 要回答：

| 问题 | 用途 |
|---|---|
| 它测什么 | 找差异点 |
| 数据怎么构造 | 借鉴 dataset pipeline |
| 是否多 session | 对齐 long-horizon setting |
| 是否有事实更新/撤回 | 对齐 deviation |
| 是否测 stale memory | 对齐 drift/error taxonomy |
| 是否测成本 | 对齐 efficiency |
| 是 QA 还是 agentic task | 凸显本项目区别 |

### 产出物

- `memory_systems_survey.md`
- `benchmark_survey.md`
- 一张 memory systems 对比表
- 一张 existing benchmarks 对比表

### 验收标准

能用一段话解释：

```text
已有 benchmark 多测长期 QA、检索或对话记忆；LHMSB 测 procedural agentic tasks 中的 task gain、deviation 和 memory lifecycle cost。
```

## 4. 阶段二：确定 Benchmark 架构和核心指标

### 目标

把 benchmark 的核心定义收紧成两个主轴：

```text
Efficiency + Deviation
```

### Efficiency 的定义

Efficiency 不只是检索准确率。它要评测：

```text
记忆系统是否把有用记忆及时、低成本地提供给 agent，并提升任务完成度。
```

建议拆成 4 个指标：

| 指标 | 含义 |
|---|---|
| Task Score | agent 最终任务完成质量 |
| Utilization Rate | 跨 session 信息是否被用上 |
| Retrieval Quality | memory 是否能返回相关、当前有效的信息 |
| Memory ROI | 相比 no_memory 的收益是否值得成本 |

### Efficiency 怎么测

#### 1. Task Score

研究任务：

- 回答是否符合当前有效证据
- 是否引用撤回证据
- 是否能综合多个当前事实

代码任务：

- 当前代码是否通过 hidden pytest
- 是否使用当前 API
- 是否遵守当前约束

#### 2. Utilization Rate

只看必须依赖跨 session 记忆的 probes。

例子：

```text
Session 1: 告诉 agent 默认 status 是 active
Session 2: 清空上下文
Session 3: 问当前默认 status
```

如果 agent 答对，说明 memory 起了作用。

#### 3. Retrieval Quality

分成两种：

- **Endogenous retrieval**：agent 自己 query memory 后得到的结果。
- **Oracle retrieval**：benchmark 用固定 query 直接问 memory。

这个拆分能区分：

```text
memory 检索差
agent 不会问 memory
```

#### 4. Memory ROI

建议作为 efficiency 的 headline：

```text
Memory ROI = normalized task gain over no_memory / memory-attributable cost
```

成本包括：

- memory 内部 LLM tokens
- embedding tokens/calls
- storage bytes
- retrieval latency
- write/update latency
- reflection/summarization cost

### Deviation 的定义

Deviation 评测：

```text
记忆系统是否让 agent 偏离当前目标、使用过期事实、违反仍然有效的约束。
```

建议拆成 3 类：

| 类型 | 含义 |
|---|---|
| Stale Fact Use | 使用已经撤回或被更新的事实 |
| Constraint Violation | 违反仍然有效的约束 |
| Behavioral Flip | 没有新事件触发却改变目标或策略 |

### Deviation 怎么测

#### 1. Stale Fact Use

例子：

```text
step 1: 证据 A 有效
step 3: 证据 A 被撤回
step 5: agent 仍引用证据 A
```

这算 deviation。

#### 2. Constraint Violation

例子：

```text
当前要求：必须使用 make_widget
agent 仍然使用 create_widget
```

这算 deviation。

#### 3. Behavioral Flip

例子：

```text
Session 1: agent 明确围绕研究问题 Alpha
Session 3: 没有任何新证据，却转去研究 Beta
```

这算 deviation。

### 汇总指标

建议使用：

```text
Deviation Index = weighted deviation violations / aligned deviation probes
```

默认权重可以是：

```text
stale fact use: 1.0
constraint violation: 1.5
behavioral flip: 1.0
```

constraint violation 权重大一些，因为它代表 agent 违反仍然有效的规则。

### 产出物

- `benchmark_design.md`
- `metrics.md`
- efficiency 指标公式
- deviation 指标公式
- 两个 family 的 probe taxonomy

### 验收标准

能清楚回答：

```text
Efficiency 测什么？
Deviation 测什么？
为什么 retrieval accuracy 不能单独代表 memory system 好坏？
为什么 no_memory 是必要 baseline？
```

## 5. 阶段三：构造数据集并校准指标

### 目标

确认数据集真的能测 efficiency 和 deviation。

### 数据集结构

每个 episode 应该由三层组成：

1. **WorldEvent ledger**
   结构化事实账本。
2. **Surface sessions**
   给 agent 看的自然语言或代码任务文本。
3. **Aligned probes**
   固定测点，所有 memory systems 面对同样 probes。

### Research Family

需要生成：

- synthetic facts
- evidence DAG
- inject/change/retract events
- objective constraints
- factual probes
- update probes
- synthesis probes
- deviation probes

重点测试：

- 旧证据是否被记住
- 新证据是否覆盖旧证据
- 撤回证据是否不再使用
- 目标是否保持一致

### Software Family

需要生成：

- evolving requirements
- API rename
- default value change
- convention add/remove
- deprecated behavior
- hidden pytest

重点测试：

- 需求是否跨 session 保留
- 变更是否被吸收
- 旧 API 是否被停止使用
- 当前代码是否通过测试

### 校准条件

至少跑：

```text
fake_perfect
fake_bad
no_memory
```

预期结果：

```text
fake_perfect > no_memory > fake_bad
```

如果加入普通 memory baseline，理想趋势是：

```text
fake_perfect > good memory > no_memory > fake_bad
```

### 产出物

- frozen datasets
- dataset cards
- manifest
- dataset statistics table
- calibration scorecard

### 验收标准

- dataset 可以 verify
- dataset 可以 regen-check
- fake_perfect 分数高
- fake_bad 分数低
- no_memory 在 cross-session probes 上明显弱
- deviation probes 能抓到 stale fact 和 constraint violation

## 6. 阶段四：Smoke Run

### 目标

验证整条评测链路能跑通。

Smoke run 不追求论文结论，只检查系统能不能稳定工作。

### 推荐条件

```text
no_memory
bm25 或 chroma
fake_perfect
fake_bad
```

### 检查内容

- episode 能生成
- session 能执行
- adapter 能接入
- memory 能写入/检索
- probe 能评分
- deviation 能统计
- cost 能记录
- Memory ROI 能计算
- scorecard 能输出

### 产出物

- `runs/smoke/.../scorecard.md`
- `runs/smoke/.../scorecard.json`
- `runs/smoke/.../run_manifest.json`
- smoke run bug list

### 验收标准

- smoke run 能重复跑
- scorecard 能生成
- 没有 NaN/inf
- no_memory 的 ROI 是 N/A
- fake_perfect 和 fake_bad 排序符合预期

## 7. 阶段五：Pilot Run

### 目标

在 full run 前用小规模真实实验发现问题。

### 推荐规模

```text
2 families
3 seeds
10-20 episodes per family
5-7 conditions
```

### 推荐条件

```text
no_memory
bm25
chroma
mem0
letta 或 graphiti
fake_perfect
fake_bad
```

### 检查内容

- 真实 backend 是否能稳定跑
- adapter 是否有状态泄漏
- cost 是否异常
- retrieval latency 是否过高
- 某些 probes 是否太简单或太难
- full run 预计成本和时间

### 产出物

- pilot scorecard
- pilot Pareto plots
- failure report
- cost report
- probe difficulty report

### 验收标准

- 每个 condition 有可解释结果
- 失败率被记录
- 失败 run 没有被丢弃
- full run 预算可估算

## 8. 阶段六：Full Run

### 目标

跑论文主实验。

### 推荐实验

主实验：

```text
2 families
3 seeds
50-100 episodes per family
6-8 conditions
```

条件：

```text
no_memory
bm25
chroma
mem0
letta
graphiti 或 cognee
fake_perfect
fake_bad
```

### Ablation

至少做：

1. context budget
2. retraction rate
3. retrieval top-k
4. cost weight sensitivity

### Error Analysis

统计错误来源：

- retrieval miss
- memory returned stale fact
- agent ignored correct retrieval
- stale fact citation
- deprecated API use
- active constraint violation
- unnecessary abstention
- format error

### 产出物

- main result table
- ROI Pareto plot
- deviation breakdown plot
- cost breakdown plot
- ablation tables
- error taxonomy table

### 验收标准

能回答：

```text
哪个 memory system task score 最高？
哪个 memory system Memory ROI 最高？
哪个 memory system deviation 最低？
retrieval quality 是否能预测 task success？
高成本 memory 是否真的值得？
```

## 9. 阶段七：论文 Draft

### 写作顺序

不要等 full run 后才开始写。建议这样写：

| 阶段 | 写什么 |
|---|---|
| 调研后 | Related Work |
| 架构确定后 | Method |
| 数据集完成后 | Dataset |
| smoke/pilot 后 | Experimental Setup |
| full run 后 | Results |
| error analysis 后 | Discussion / Limitations |

### 论文结构

建议结构：

1. Introduction
2. Related Work
3. Benchmark Design
4. Task Families
5. Metrics
6. Experimental Setup
7. Results
8. Analysis
9. Limitations
10. Conclusion

### 必备图表

- Benchmark comparison table
- Memory system comparison table
- Benchmark pipeline figure
- Dataset statistics table
- Main results table
- Memory ROI Pareto plot
- Deviation breakdown plot
- Cost breakdown plot
- Ablation table

### 产出物

- paper draft
- appendix
- reproducibility checklist
- code release notes
- dataset card

### 验收标准

- 所有主结论都有实验支持
- 没有用未完成实验支撑强结论
- 方法定义和代码实现一致
- appendix 能让别人复现实验

## 10. 风险和补救

| 风险 | 补救 |
|---|---|
| 真实 memory system 跑不稳 | 先保证 no_memory、bm25、chroma、fake_perfect、fake_bad 完整 |
| full run 成本过高 | 先跑 pilot，估算预算，再缩小 condition 或 episode 数 |
| deviation 指标抓不到错误 | 增加 fake_bad 和手工 adversarial fixtures |
| 数据集太简单 | 增加 retraction、change、distractor、cross-session probes |
| 数据集太难 | 分 short/medium/long 难度 |
| reviewer 质疑只测 retrieval | 强调 task score、utilization、Memory ROI、deviation |
| reviewer 质疑合成数据 | 提供 dataset card、generation rules、calibration oracles、error analysis |
| 与 BEAM 名字冲突 | 论文中使用 LHMSB 或 MemoryROI-Bench |

## 11. 最小可投版本

如果时间紧，优先完成：

1. 两个 family：research 和 software
2. 两个核心指标：efficiency 和 deviation
3. 五个条件：no_memory、bm25、chroma、fake_perfect、fake_bad
4. 一个真实 memory system：Mem0 或 Letta
5. 完整 scorecard
6. 一个 ablation：context budget
7. 一个 error analysis：stale fact / constraint violation

最低可接受结论：

```text
LHMSB shows that retrieval quality alone does not determine long-horizon agent performance. Memory systems must be evaluated by task gain, cost, and deviation under fixed-world counterfactual replay.
```

## 12. 推荐执行节奏

### 正常节奏

```text
Week 1: 调研 systems 和 datasets
Week 2: 固定架构和指标
Week 3: 数据集构造和校准
Week 4: smoke run 和 bug fix
Week 5: pilot run
Week 6: full run
Week 7: ablation 和 error analysis
Week 8: paper draft 和 appendix
```

### 压缩节奏

```text
Day 1-3: 调研
Day 4-6: 定架构和指标
Day 7-11: 数据集和 baseline
Day 12-14: smoke + pilot
Day 15-19: full run + ablation
Day 20-25: 写论文
```

压缩节奏风险高。建议先保证 benchmark 定义干净、实验公平、结论克制。
