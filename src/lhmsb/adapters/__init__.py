"""Memory-system adapter interface (canonical: ``spec/05-systems.md``).

Re-exports the adapter API so downstream code imports from one place::

    from lhmsb.adapters import MemorySystemAdapter, Capabilities, UnsupportedOperation

``ChromaAdapter`` is intentionally NOT re-exported here: it imports the optional
``chromadb`` dependency lazily, so importing this package must not require the
``chroma`` extra. Import it directly: ``from lhmsb.adapters.chroma import ChromaAdapter``.
"""

from lhmsb.adapters.base import (
    Capabilities,
    ForgettingCapability,
    MemorySystemAdapter,
    ReflectionCapability,
    SessionCapability,
    UnsupportedOperation,
)
from lhmsb.adapters.fakes import (
    FakeBadAdapter,
    FakePerfectAdapter,
    GroundTruthFact,
    WrongMemoryAdapter,
)
from lhmsb.adapters.no_memory import NoMemoryAdapter

__all__ = [
    "Capabilities",
    "FakeBadAdapter",
    "FakePerfectAdapter",
    "ForgettingCapability",
    "GroundTruthFact",
    "WrongMemoryAdapter",
    "MemorySystemAdapter",
    "NoMemoryAdapter",
    "ReflectionCapability",
    "SessionCapability",
    "UnsupportedOperation",
]
