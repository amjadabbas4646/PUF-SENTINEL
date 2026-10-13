"""Deterministic, hierarchical seeding.

Every random stream in the project is derived from one master seed and a tuple
of labels, for example ("E2", "device", 3, "challenges"). Two streams with
different labels are statistically independent (numpy SeedSequence spawning),
and the same labels always give the same stream on every platform.
"""
from __future__ import annotations

import zlib

import numpy as np

MASTER_SEED = 20261011


def _to_int(label) -> int:
    if isinstance(label, (int, np.integer)):
        return int(label) & 0xFFFFFFFF
    return zlib.crc32(str(label).encode("utf-8")) & 0xFFFFFFFF


def seed_sequence(*labels, master: int = MASTER_SEED) -> np.random.SeedSequence:
    return np.random.SeedSequence([master] + [_to_int(x) for x in labels])


def rng_for(*labels, master: int = MASTER_SEED) -> np.random.Generator:
    """Independent numpy Generator for the given label path."""
    return np.random.default_rng(seed_sequence(*labels, master=master))


def int_seed(*labels, master: int = MASTER_SEED) -> int:
    """31-bit integer seed for libraries that want an int (sklearn, torch)."""
    return int(seed_sequence(*labels, master=master).generate_state(1)[0] & 0x7FFFFFFF)
