"""Code-offset secure sketch with a hash-based key derivation.

This is the construction of Juels and Wattenberg and of Dodis et al., used
here to reconstruct the device key K from a noisy weak-PUF reading (an SRAM
power-up pattern in the simulation).

Enrollment  Gen(w):
    for each block b = 1..B
        draw a uniformly random k-bit (or k-symbol) message s_b
        helper h_b = w_b XOR Encode(s_b)
    K = SHA-256(label || s_1 || ... || s_B)
    store the helper data h = (h_1, ..., h_B) in ordinary non-volatile memory

Reconstruction  Rep(w', h):
    for each block: decode (w'_b XOR h_b) to a codeword, extract s'_b
    K' = SHA-256(label || s'_1 || ... || s'_B)

Success means K' = K, i.e. exact recovery of every block. A block decodes
correctly exactly when the number of bit errors (BCH) or symbol errors (RS)
between w_b and w'_b is at most the correction radius t of the code.

What the helper data reveal: h_b is public. For an (n, k) binary linear code
it reveals at most n - k bits about w_b, so if w_b has n bits of min-entropy
the message s_b keeps k bits. If the PUF bits are biased or correlated the
loss can be larger; that case is not analysed here. The construction gives no
protection against an adversary who modifies the helper data.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .bch import BCH
from .rs import RS

KDF_LABEL = b"PUF-SENTINEL/key/v1"


@dataclass(frozen=True)
class CodeSpec:
    """One error-correction configuration, sized for a secret of >= 128 bits."""
    name: str
    family: str          # "BCH" or "RS"
    n: int               # block length in code symbols
    k: int               # block dimension in code symbols
    t: int               # guaranteed correction radius in code symbols
    bits_per_symbol: int
    blocks: int

    @property
    def puf_bits(self) -> int:          # weak-PUF cells consumed
        return self.blocks * self.n * self.bits_per_symbol

    @property
    def secret_bits(self) -> int:       # message bits before hashing
        return self.blocks * self.k * self.bits_per_symbol

    @property
    def helper_bits(self) -> int:       # code-offset helper data stored
        return self.puf_bits

    @property
    def leakage_bound_bits(self) -> int:  # upper bound on leakage about w
        return self.blocks * (self.n - self.k) * self.bits_per_symbol


def make_code(spec: CodeSpec):
    if spec.family == "BCH":
        code = BCH(7, spec.t)
    elif spec.family == "RS":
        code = RS(spec.n, spec.k)
    else:
        raise ValueError(spec.family)
    if (code.n, code.k, code.t) != (spec.n, spec.k, spec.t):
        raise AssertionError(
            f"{spec.name}: spec says ({spec.n},{spec.k},t={spec.t}) but the "
            f"constructed code is ({code.n},{code.k},t={code.t})")
    return code


def _to_symbols(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    if bits_per_symbol == 1:
        return bits.astype(np.uint8)
    return np.packbits(bits.reshape(-1, bits_per_symbol), axis=1)[:, 0]


class CodeOffsetExtractor:
    def __init__(self, spec: CodeSpec):
        self.spec = spec
        self.code = make_code(spec)

    def _blocks(self, w_bits: np.ndarray) -> np.ndarray:
        w_bits = np.asarray(w_bits, dtype=np.uint8)
        if w_bits.shape != (self.spec.puf_bits,):
            raise ValueError(f"need {self.spec.puf_bits} PUF bits")
        sym = _to_symbols(w_bits, self.spec.bits_per_symbol)
        return sym.reshape(self.spec.blocks, self.spec.n)

    @staticmethod
    def _kdf(messages: list[np.ndarray]) -> bytes:
        h = hashlib.sha256(KDF_LABEL)
        for s in messages:
            h.update(np.asarray(s, dtype=np.uint8).tobytes())
        return h.digest()

    def gen(self, w_bits: np.ndarray, rng: np.random.Generator):
        """Return (key, helper). helper has shape (blocks, n) in code symbols."""
        w = self._blocks(w_bits)
        hi = 2 if self.spec.bits_per_symbol == 1 else 256
        msgs, helper = [], []
        for b in range(self.spec.blocks):
            s = rng.integers(0, hi, size=self.spec.k, dtype=np.uint8)
            msgs.append(s)
            helper.append(w[b] ^ self.code.encode(s))
        return self._kdf(msgs), np.stack(helper)

    def rep(self, w_noisy_bits: np.ndarray, helper: np.ndarray):
        """Return (key or None, per-block decoder success flags)."""
        w = self._blocks(w_noisy_bits)
        msgs, flags = [], []
        for b in range(self.spec.blocks):
            cw, ok = self.code.decode(w[b] ^ helper[b])
            flags.append(ok)
            msgs.append(self.code.extract(cw))
        if not all(flags):
            return None, flags
        return self._kdf(msgs), flags

    def block_error_weights(self, error_bits: np.ndarray) -> np.ndarray:
        """Number of erroneous code symbols per block for a bit-error pattern."""
        e = np.asarray(error_bits, dtype=np.uint8)
        if self.spec.bits_per_symbol == 1:
            return e.reshape(self.spec.blocks, self.spec.n).sum(axis=1)
        sym_err = e.reshape(self.spec.blocks, self.spec.n,
                            self.spec.bits_per_symbol).any(axis=2)
        return sym_err.sum(axis=1)
