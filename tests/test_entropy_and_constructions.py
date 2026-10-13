import math

import numpy as np
import pytest

from common.seeding import rng_for
from error_correction.bch import BCH
from error_correction.entropy import (block_entropy_bounds, exact_block_entropy,
                                      key_entropy_bounds, naive_entropy_bound)
from puf_models.arbiter import XorArbiterPUF
from puf_models.constructions import (ObfuscatedPUF, challenge_linear, challenge_prf,
                                      challenge_xor, expand_challenges, gf2_rank,
                                      hash_output, random_invertible_gf2)


def _generator_rows(code: BCH) -> list[list[int]]:
    rows = []
    for i in range(code.k):
        msg = np.zeros(code.k, dtype=np.uint8)
        msg[i] = 1
        rows.append([int(b) for b in code.encode(msg)])
    return rows


@pytest.mark.parametrize("t", [1, 2, 3])
@pytest.mark.parametrize("p", [0.5, 0.45, 0.4, 0.3, 0.2])
def test_entropy_bounds_bracket_the_exact_value_for_length_15(t, p):
    code = BCH(4, t)
    exact = exact_block_entropy(_generator_rows(code), p)
    lo, hi = block_entropy_bounds(code.n, code.k, t, p)
    assert lo - 1e-9 <= exact <= hi + 1e-9
    assert exact <= code.k + 1e-9


def test_entropy_is_exact_for_a_perfect_code():
    code = BCH(4, 1)                          # Hamming (15, 11): every coset leader has weight <= 1
    for p in (0.5, 0.4, 0.25):
        lo, hi = block_entropy_bounds(code.n, code.k, 1, p)
        assert abs(lo - hi) < 1e-9
        assert abs(lo - exact_block_entropy(_generator_rows(code), p)) < 1e-9


def test_uniform_source_keeps_all_message_bits():
    for n, k, t in ((127, 78, 7), (127, 64, 10), (127, 36, 15), (384, 128, 16)):
        lo, hi = block_entropy_bounds(n, k, t, 0.5)
        assert abs(lo - k) < 1e-6 and abs(hi - k) < 1e-6


def test_entropy_decreases_with_bias_and_beats_the_naive_bound():
    prev = None
    for p in (0.5, 0.48, 0.45, 0.4, 0.35, 0.3):
        lo, hi = block_entropy_bounds(127, 64, 10, p)
        assert lo <= hi + 1e-9
        assert lo >= naive_entropy_bound(127, 64, p) - 1e-9
        if prev is not None:
            assert lo <= prev + 1e-9
        prev = lo


def test_key_entropy_adds_over_blocks():
    one = key_entropy_bounds(127, 36, 15, 1, 1, 0.45)
    four = key_entropy_bounds(127, 36, 15, 1, 4, 0.45)
    assert math.isclose(four["entropy_lower"], 4 * one["entropy_lower"])
    assert four["secret_bits"] == 144 and four["helper_redundancy_bits"] == 4 * 91


def test_gf2_rank_and_invertible_matrix():
    assert gf2_rank(np.eye(8, dtype=np.uint8)) == 8
    assert gf2_rank(np.ones((5, 5), dtype=np.uint8)) == 1
    a = random_invertible_gf2(64, rng_for("test", "gf2"))
    assert gf2_rank(a) == 64


def test_challenge_transforms_are_bijections_on_samples():
    rng = rng_for("test", "transforms")
    c = np.unique(rng.integers(0, 2, size=(4000, 64), dtype=np.uint8), axis=0)
    kappa = rng.integers(0, 2, size=64, dtype=np.uint8)
    a = random_invertible_gf2(64, rng)
    key = bytes(range(32))
    assert np.array_equal(challenge_xor(challenge_xor(c, kappa), kappa), c)
    for out in (challenge_xor(c, kappa), challenge_linear(c, a), challenge_prf(key, c)):
        assert out.shape == c.shape
        assert len(np.unique(out, axis=0)) == len(c)
    # linearity of class L over GF(2)
    assert np.array_equal(challenge_linear(c[:50] ^ c[50:100], a),
                          challenge_linear(c[:50], a) ^ challenge_linear(c[50:100], a))
    # the PRF transform depends on the key
    assert not np.array_equal(challenge_prf(key, c[:200]), challenge_prf(bytes(32), c[:200]))


def test_obfuscated_puf_matches_its_definition():
    rng = rng_for("test", "obf")
    base = XorArbiterPUF(64, 2, rng)
    c = rng.integers(0, 2, size=(500, 64), dtype=np.uint8)
    kappa = rng.integers(0, 2, size=64, dtype=np.uint8)
    x = ObfuscatedPUF(base, "X", kappa)
    assert np.array_equal(x.eval(c), base.eval(c ^ kappa))
    h = ObfuscatedPUF(base, "H", m=8)
    sub = expand_challenges(c, 8)
    assert sub.shape == (500, 8, 64)
    bits = base.eval(sub.reshape(-1, 64)).reshape(500, 8)
    assert np.array_equal(h.eval(c), hash_output(c, bits))
    assert np.array_equal(h.eval(c, internal=sub), h.eval(c))
    # the hashed output is balanced and changes when one PUF bit changes
    assert 0.4 < h.eval(c).mean() < 0.6
    flipped = bits.copy()
    flipped[:, 0] ^= 1
    assert 0.35 < np.mean(hash_output(c, flipped) != hash_output(c, bits)) < 0.65
