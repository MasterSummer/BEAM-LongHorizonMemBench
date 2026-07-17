from __future__ import annotations

from lhmsb.adapters.mem0_qualification import Mem0QualificationAdapter
from lhmsb.qualification.memory_runtime import LifecycleCapabilities, MemoryRuntime


class EmptyMem0Backend:
    def add(self, messages: list[dict[str, str]], **kwargs: object) -> object:
        del messages, kwargs
        return {"results": []}

    def search(self, query: str, **kwargs: object) -> object:
        del query, kwargs
        return {"results": []}

    def get_all(self, **kwargs: object) -> object:
        del kwargs
        return {"results": []}

    def history(self, memory_id: str) -> object:
        del memory_id
        return []


def test_mem0_adapter_implements_the_complete_generic_runtime_contract() -> None:
    adapter = Mem0QualificationAdapter(
        EmptyMem0Backend(),
        user_id="user",
        run_id="run",
    )

    assert isinstance(adapter, MemoryRuntime)
    assert adapter.capabilities == LifecycleCapabilities(
        add=True,
        update=True,
        delete=True,
        merge=False,
        links=False,
        history=True,
        resumable=True,
    )
    footprints = adapter.storage_footprints()
    assert tuple(item.component for item in footprints) == (
        "mem0_vector_store",
        "mem0_history_store",
    )
    assert all(item.bytes is None and item.unavailable_reason for item in footprints)
