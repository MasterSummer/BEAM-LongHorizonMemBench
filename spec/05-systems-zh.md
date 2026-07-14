# 05 — 被测系统与适配器接口

> **状态**：v1 规范。任务 5 实现此处定义的 `MemorySystemAdapter` ABC。
> 任务 12–16 按这些确切签名实现各系统适配器。任何与此签名的偏差都是规范违规。

---

## 1. MemorySystemAdapter — 规范接口

所有被测记忆系统（包括无记忆对照组）都封装在一个通用抽象接口之后。适配器是智能体 harness 读写记忆的**唯一路径**。LangGraph 的内置 checkpointer 和任何框架级缓存必须被禁用——所有持久化通过适配器流动。

### 1.1 必需方法（确切签名）

```python
class MemorySystemAdapter(ABC):

    @abstractmethod
    def initialize(self, *, user_id: str, session_id: str | None = None, **config) -> None:
        """为用户设置记忆后端。在第一个会话之前每个用户调用一次。
        session_id=None 表示尚不存在会话作用域。"""
        ...

    @abstractmethod
    def reset(self, *, user_id: str) -> None:
        """删除用户的所有记忆。在 episode 之间使用以确保干净状态。
        必须是幂等的。"""
        ...

    @abstractmethod
    def add_memory(self, content: str, *, user_id: str,
                   session_id: str | None = None,
                   metadata: dict | None = None) -> str:
        """将内容摄取到记忆系统中。返回一个 memory_id
        （此记忆条目的唯一、稳定的字符串标识符）。

        此操作触发的所有内部 LLM/嵌入调用必须
        放在 memory_scope() 中包装，以便 token 被计数。"""
        ...

    @abstractmethod
    def search(self, query: str, *, user_id: str,
               session_id: str | None = None,
               top_k: int = 10, **filters) -> SearchResult:
        """为查询检索相关记忆。

        返回一个 SearchResult，包含 MemoryEntry 对象列表
        和 total_count。结果应按相关性排序。

        此操作触发的内部 LLM/嵌入调用必须放在
        memory_scope() 中包装。"""
        ...

    @abstractmethod
    def update_memory(self, memory_id: str, *,
                      content: str | None = None,
                      metadata: dict | None = None) -> None:
        """更新现有记忆条目的内容和/或元数据。
        content=None 表示保留现有内容；metadata=None 表示保留
        现有元数据。至少提供一个。"""
        ...

    @abstractmethod
    def delete_memory(self, memory_id: str) -> None:
        """移除一个记忆条目。该条目在删除后不应再出现在
        搜索结果中。幂等——删除不存在的条目是无操作，非错误。

        注意：基准评分的是行为，而非实现。无论系统是通过
        移除、墓碑标记、边失效还是检索过滤来实现删除——只要
        被撤回/删除的事实不再影响搜索结果即可。"""
        ...
```

**返回类型**：

```python
@dataclass(frozen=True)
class MemoryEntry:
    memory_id: str
    content: str
    metadata: dict | None
    created_at: str       # ISO 8601 时间戳
    updated_at: str       # ISO 8601 时间戳
    score: float | None   # 来自搜索的相关性分数，直接检索时为 None

@dataclass(frozen=True)
class SearchResult:
    results: list[MemoryEntry]
    total_count: int      # 总匹配结果数（可能 > len(results)）
```

### 1.2 优雅的能力降级

并非所有记忆系统都支持每个操作。`Capabilities` 自省机制允许 harness 查询后端支持什么：

```python
@dataclass(frozen=True)
class Capabilities:
    supports_add: bool = True
    supports_search: bool = True
    supports_update: bool = True
    supports_delete: bool = True
    supports_reset: bool = True
    supports_sessions: bool = False
    supports_reflection: bool = False
    supports_forgetting: bool = False
```

适配器暴露 `get_capabilities() -> Capabilities`。当 harness 或智能体调用不支持的操作时，适配器必须抛出 `UnsupportedOperation`（一个记录日志但非致命的异常）——绝不崩溃，绝不静默忽略。

### 1.3 可选能力 Mixin

适配器可以实现这些 mixin 以暴露额外的记忆生命周期操作。基准在可用时使用它们；不强制要求。

```python
class ReflectionCapability(ABC):
    """支持整合/自重组的记忆系统。"""

    @abstractmethod
    def reflect(self, *, user_id: str, session_id: str | None = None) -> None:
        """触发反思/整合过程。此操作的内部 LLM token
        必须在 memory_scope() 下计数。"""
        ...

    @abstractmethod
    def summarize(self, *, user_id: str, session_id: str | None = None,
                  query: str | None = None) -> str:
        """生成存储记忆的摘要，可选地限定到某个查询。"""
        ...


class ForgettingCapability(ABC):
    """带有显式衰减/遗忘机制的记忆系统。"""

    @abstractmethod
    def apply_decay(self, *, user_id: str, **params) -> None:
        """应用遗忘/衰减步骤。可能降低相关性分数、
        归档旧记忆或物理删除低重要性条目。"""
        ...


class SessionCapability(ABC):
    """带有显式会话/线程分组的记忆系统。"""

    @abstractmethod
    def list_sessions(self, *, user_id: str) -> list[str]:
        """返回用户的所有会话 ID。"""
        ...

    @abstractmethod
    def get_session_memories(self, *, user_id: str,
                              session_id: str) -> list[MemoryEntry]:
        """返回限定在某个会话的所有记忆条目。"""
        ...

    @abstractmethod
    def promote_session(self, *, user_id: str, session_id: str) -> None:
        """将会话范围的记忆提升到全局/用户范围。"""
        ...
```

---

## 2. 被测系统

### 2.1 排行榜条件（6 个系统）

这六个条件出现在实际排行榜上（原生赛道为主要，受控赛道为次要，绝不混合）。

| # | 条件 | 系统 | 描述 | 关键 API |
|---|------|------|------|----------|
| 1 | `no_memory` | 无记忆对照组 | 在会话间不存储任何内容。`search()` 始终返回空。ROI 的反事实基线。 | `add`/`update`/`delete` 是无操作，返回有效 ID |
| 2 | `chroma` | ChromaDB | 纯向量存储基线。内存/离线。 | `collection.add/query/upsert/delete` |
| 3 | `mem0` | Mem0 | 混合语义 + BM25 + 实体记忆。`add` 上有内部 LLM。 | `Memory.add/search/update/delete` |
| 4 | `letta` | Letta / AI-Memory-SDK | 带有睡眠反思的智能体自编辑记忆区块。 | `add_messages/search/get_memory/delete_block` |
| 5 | `graphiti` | Graphiti (Zep) | 带有自动时间失效的时间知识图谱。 | `add_episode/search/remove_episode` |
| 6 | `cognee` | Cognee | 多阶段流水线（`cognify`）及自重组（`memify`）。基于文件的默认。 | `remember/recall/forget/improve` |

### 2.2 敏感性 / 校准条件（2 个假系统）

这些**不在**真实排行榜上。它们是用于验证指标敏感性的校准条件：`fake_perfect` 下的任务分数必须以明显差距超过 `fake_bad` 下的分数，否则指标有缺陷。

| # | 条件 | 行为 |
|---|------|------|
| F1 | `fake_perfect` | Oracle 记忆。为任何查询返回恰好相关的当前（未撤回）事实。使用 episode 的真实事实库。记忆系统理论上限。 |
| F2 | `fake_bad` | 对抗性记忆。返回看似合理但不正确或已撤回的事实。理论下界——任何真实记忆系统应超过它。 |

### 2.3 能力矩阵

| 条件 | 反思 | 遗忘 | 会话 |
|------|------|------|------|
| `no_memory` | — | — | — |
| `chroma` | — | — | — |
| `mem0` | add 时隐式 | — | — |
| `letta` | `reflect()`（睡眠时） | — | 通过 blocks |
| `graphiti` | — | 时间自动失效 | 通过 `group_id` |
| `cognee` | `memify()` / `improve()` | — | 通过 `session_id` |
| `fake_*` | — | — | — |

---

## 3. 赛道规则 — 原生 vs 受控

### 3.1 原生赛道（主要）

每个记忆系统按其发布状态测试：使用默认配置、自己的内部 LLM 模型、自己的嵌入器、自己的默认参数。这是主要排行榜，因为它代表*从业者实际部署的样子*。

**所有内部 LLM/嵌入成本都被仪表化并报告**，因此使用更昂贵内部模型的系统在其 ROI 中承担该成本。原生赛道并非"不公平"——它是"完全核算"。

### 3.2 受控赛道（次要）

当记忆系统支持配置其内部 LLM 和嵌入器时，它也用一个固定的共享模型进行测试（与智能体和受控赛道同侪使用的相同开放权重模型）。这隔离了记忆系统架构与模型选择。

**规则**：
- 受控赛道在所有支持模型配置的系统上使用相同的固定智能体模型。
- 不支持模型配置的系统（如 ChromaDB、无记忆）在两个赛道中等价存在（无内部 LLM 可固定）。
- 受控赛道结果在单独的表格/章节中报告，绝不与原生赛道结果合并在单一排行榜中。
- `RunConfig.track` 字段记录一个运行属于哪个赛道。

### 3.3 赛道比较

系统在原生态和受控之间的 ROI 差异量化了其性能有多少可归因于内部模型选择 vs. 其架构。受控赛道 ROI 显著下降的系统是依赖模型的；维持 ROI 的系统是架构稳健的。

---

## 4. 全生命周期成本仪表化

### 4.1 要求

每个适配器必须被包装，以便记忆系统内部消耗的所有 LLM 和嵌入 token 都被计数。这包括：

- **添加时处理**：Mem0 的提取 LLM、Cognee 的 `cognify` 流水线、Graphiti 在 `add_episode` 上的实体提取、Letta 的区块自编辑。
- **搜索时处理**：记忆系统在 `search()` 期间调用的任何查询重写、嵌入生成、重排序 LLM。
- **反思/整合**：Letta 的睡眠时、Cognee 的 `memify()`/`improve()`。
- **嵌入**：为向量存储嵌入的 token 和嵌入 API 调用消耗的 token。

这些 token 进入 `CostVector.mem_internal_in_tokens` 和 `mem_internal_out_tokens`（以及 `embedding_tokens` / `embedding_calls`），与智能体循环的 token（`agent_input_tokens` / `agent_output_tokens`）分开。

### 4.2 机制

成本仪表化层（任务 6）提供：

- `CostMeter`：带有作用域归属的线程安全累加器。
- `memory_scope()`：上下文管理器。在 `with meter.memory_scope():` 中进行的任何 LLM/嵌入调用都归属到记忆系统。
- `instrumented_llm(client)` 和 `instrumented_embedder(fn)`：自动计数 token 并尊重当前作用域的包装器。

适配器代码看起来像：

```python
def add_memory(self, content, *, user_id, session_id=None, metadata=None):
    with self.cost_meter.memory_scope():
        result = self._backend.add(content, user_id=user_id, ...)
    return result.memory_id
```

`memory_scope()` 确保后端在内部进行的任何 LLM 调用都被计为 `mem_internal_*`，而非智能体 token。

### 4.3 排除规则

以下成本不计入系统 CostVector：

- **数据集生成**：创建冻结 episode 的一次性成本。
- **判定器 token**：稀疏判定器的 LLM 调用。
- **表面渲染**：将结构化事件渲染为自然文本（冻结缓存，排除在 episode 成本之外）。
- **Harness 开销**：LangGraph 框架自身的 token 使用量（最小化、确定性、被跟踪但排除在系统比较之外）。

### 4.4 严格模式

当运行配置中 `strict_instrumentation=True` 时，在显式作用域（智能体或记忆）之外进行的任何 LLM 或嵌入调用都会抛出 `CostInstrumentationError`。这防止无声的未计数 token。在非严格模式下，未归入作用域的调用被归属到一个 catch-all `unscoped` 桶并标记警告。
