"""Seeded randomness and deterministic ID generation.

POLICY — No Hidden Randomness:
  Every source of randomness in this benchmark MUST flow through `seeded_rng()`.
  No unseeded `random.random()`, `random.randint()`, or `random.choice()` calls
  anywhere in the codebase.  All random operations are traceable back to a seed
  stored in the run manifest or episode config, enabling exact reproduction.

  Deterministic ID generation uses `hashlib.sha256`, NOT Python's built-in
  `hash()` (which is salted per process and non-deterministic across runs).
"""

from __future__ import annotations

import hashlib
import random


def seeded_rng(seed: int) -> random.Random:
    """Return a `random.Random` instance seeded with `seed`.

    The returned RNG is a standard Python Mersenne Twister, suitable for
    reproducible shuffle, choice, and sampling.  For cryptographic hashing
    use `deterministic_id()` or `lhmsb.hashing`.
    """
    return random.Random(seed)


def deterministic_id(episode_id: str, seed: int, counter: int) -> str:
    """Produce a stable, cross-process unique ID.

    The ID is derived from (episode_id, seed, counter) via SHA-256,
    guaranteeing the same output for the same inputs regardless of
    Python process, platform, or invocation order.

    Returns a 32-character hex string (first 128 bits of the digest)
    suitable as a structured but readable identifier.
    """
    payload = f"{episode_id}|{seed}|{counter}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return digest[:32]
