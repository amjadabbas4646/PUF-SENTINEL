"""Keyed response masks.

Two masks are implemented.

hmac_mask (the revised design)
    M_K(c) = first m bits of HMAC-SHA256(K, label || 0x00 || device_id ||
             uint16(n) || uint16(m) || pack(c))
    K is a 256-bit uniformly random device key. Under the standard assumption
    that HMAC-SHA256 is a pseudorandom function (PRF) for a uniformly random
    key, the map c -> M_K(c) is indistinguishable from a random function to an
    adversary who does not know K. The label gives domain separation from any
    other use of K, the device identifier separates devices that might share a
    key by mistake, and the encoded lengths make the input encoding injective.
    The mask is deterministic: the same challenge always gets the same mask.

pbkdf2_mask_legacy (the original manuscript, kept only for comparison)
    M_K(c) = first m bits of PBKDF2-HMAC-SHA256(password=K, salt=pack(c),
             iterations=1000). PBKDF2 is a password-stretching function. Its
    iteration count exists to slow down guessing of low-entropy passwords and
    buys nothing when K already has 256 bits of entropy, while costing about
    2000 HMAC compression calls per challenge. It has no label and no device
    identifier, so it has no domain separation.

pack(c) is numpy.packbits of the challenge, most significant bit first, with
zero padding of the last byte.
"""
from __future__ import annotations

import hashlib
import hmac
import struct

import numpy as np

MASK_LABEL = b"PUF-SENTINEL/mask/v1"
KEY_BYTES = 32


def _bits_from_digests(digests: list[bytes], m: int) -> np.ndarray:
    raw = np.frombuffer(b"".join(digests), dtype=np.uint8).reshape(len(digests), -1)
    return np.unpackbits(raw, axis=1)[:, :m]


def hmac_mask(key: bytes, challenges: np.ndarray, m: int = 1,
              device_id: bytes = b"\x00" * 16) -> np.ndarray:
    """Return the (N, m) mask bits of the revised design. m must be 1..256."""
    if len(key) != KEY_BYTES:
        raise ValueError("device key must be 32 bytes")
    if len(device_id) != 16:
        raise ValueError("device_id must be 16 bytes")
    if not 1 <= m <= 256:
        raise ValueError("m must be between 1 and 256 bits")
    c = np.asarray(challenges, dtype=np.uint8)
    prefix = MASK_LABEL + b"\x00" + device_id + struct.pack(">HH", c.shape[1], m)
    packed = np.packbits(c, axis=1)
    digests = [hmac.digest(key, prefix + row.tobytes(), "sha256") for row in packed]
    return _bits_from_digests(digests, m)


def pbkdf2_mask_legacy(key: bytes, challenges: np.ndarray, m: int = 1,
                       iterations: int = 1000) -> np.ndarray:
    """Mask of the original manuscript (PBKDF2, challenge used as the salt)."""
    c = np.asarray(challenges, dtype=np.uint8)
    n_bytes = (m + 7) // 8
    packed = np.packbits(c, axis=1)
    digests = [hashlib.pbkdf2_hmac("sha256", key, row.tobytes(), iterations,
                                   dklen=n_bytes) for row in packed]
    return _bits_from_digests(digests, m)


class MaskedPUF:
    """response(c) = base(c) XOR M_K(c), one bit per challenge."""

    def __init__(self, base, key: bytes, device_id: bytes = b"\x00" * 16,
                 legacy_pbkdf2: bool = False):
        self.base, self.key, self.device_id = base, key, device_id
        self.legacy_pbkdf2 = legacy_pbkdf2
        self.n = base.n

    def mask(self, challenges: np.ndarray) -> np.ndarray:
        if self.legacy_pbkdf2:
            return pbkdf2_mask_legacy(self.key, challenges, 1)[:, 0]
        return hmac_mask(self.key, challenges, 1, self.device_id)[:, 0]

    def eval(self, challenges: np.ndarray, noise_rng=None) -> np.ndarray:
        return self.base.eval(challenges, noise_rng) ^ self.mask(challenges)

    def unmask(self, challenges: np.ndarray, responses: np.ndarray) -> np.ndarray:
        """What an adversary who knows K computes. Diagnostic only."""
        return np.asarray(responses, dtype=np.uint8) ^ self.mask(challenges)


def new_device_key(rng: np.random.Generator) -> bytes:
    """Simulation stand-in for a 256-bit device key. In the simulation the key
    comes from the seeded generator so that runs are reproducible. A real
    device must draw it from a cryptographic random source or reconstruct it
    with the fuzzy extractor in src/error_correction."""
    return rng.integers(0, 256, size=KEY_BYTES, dtype=np.uint8).tobytes()
