import hashlib
import hmac
import math
import struct

import numpy as np
import pytest

from common.seeding import rng_for
from puf_models.arbiter import (XorArbiterPUF, ZeroPUF, expected_single_chain_ber,
                                parity_features)
from puf_models.challenges import count_duplicates, unique_challenges
from puf_models.mask import (MASK_LABEL, MaskedPUF, hmac_mask, new_device_key,
                             pbkdf2_mask_legacy)


def test_parity_features_known_vector():
    # c = (1, 0, 1): d = (-1, +1, -1); phi = (d1 d2 d3, d2 d3, d3, 1)
    phi = parity_features(np.array([[1, 0, 1]], dtype=np.uint8))
    assert phi.tolist() == [[1.0, -1.0, -1.0, 1.0]]
    assert phi.shape == (1, 4)


def test_parity_features_all_zero_challenge_is_all_ones():
    assert np.all(parity_features(np.zeros((2, 64), dtype=np.uint8)) == 1.0)


def test_dimensions_and_dtype():
    rng = rng_for("test", "dims")
    puf = XorArbiterPUF(64, 2, rng)
    c = rng.integers(0, 2, size=(100, 64), dtype=np.uint8)
    r = puf.eval(c)
    assert r.shape == (100,) and r.dtype == np.uint8
    assert set(np.unique(r)) <= {0, 1}
    assert puf.weights.shape == (2, 65)


def test_deterministic_under_fixed_seed():
    c = rng_for("test", "c").integers(0, 2, size=(500, 64), dtype=np.uint8)
    a = XorArbiterPUF(64, 2, rng_for("test", "dev", 1)).eval(c)
    b = XorArbiterPUF(64, 2, rng_for("test", "dev", 1)).eval(c)
    other = XorArbiterPUF(64, 2, rng_for("test", "dev", 2)).eval(c)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, other)


def test_xor_is_xor_of_chains():
    rng = rng_for("test", "xor")
    puf = XorArbiterPUF(32, 3, rng)
    c = rng.integers(0, 2, size=(200, 32), dtype=np.uint8)
    bits = puf.chain_bits(c)
    assert np.array_equal(puf.eval(c), bits[:, 0] ^ bits[:, 1] ^ bits[:, 2])


def test_noise_free_eval_is_repeatable_and_noise_changes_bits():
    rng = rng_for("test", "noise")
    puf = XorArbiterPUF(64, 1, rng, noisiness=0.1)
    c = rng.integers(0, 2, size=(40000, 64), dtype=np.uint8)
    ideal = puf.eval(c)
    assert np.array_equal(ideal, puf.eval(c))
    noisy = puf.eval(c, rng_for("test", "noise", "eval"))
    ber = np.mean(noisy != ideal)
    assert ber == pytest.approx(expected_single_chain_ber(0.1), abs=0.004)
    assert expected_single_chain_ber(0.1) == pytest.approx(math.atan(0.1) / math.pi)


def test_hmac_mask_matches_reference_computation():
    key = bytes(range(32))
    dev = bytes(16)
    c = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 1]], dtype=np.uint8)   # 9 bits -> 2 bytes
    msg = MASK_LABEL + b"\x00" + dev + struct.pack(">HH", 9, 8) + bytes([0b10110001, 0b10000000])
    ref = hmac.new(key, msg, hashlib.sha256).digest()
    expect = np.unpackbits(np.frombuffer(ref, dtype=np.uint8))[:8]
    assert np.array_equal(hmac_mask(key, c, 8, dev)[0], expect)


def test_mask_is_deterministic_and_key_and_device_dependent():
    rng = rng_for("test", "mask")
    c = rng.integers(0, 2, size=(4000, 64), dtype=np.uint8)
    k1, k2 = new_device_key(rng), new_device_key(rng)
    m1 = hmac_mask(k1, c, 1)
    assert np.array_equal(m1, hmac_mask(k1, c, 1))
    assert 0.45 < np.mean(m1 != hmac_mask(k2, c, 1)) < 0.55
    assert 0.45 < np.mean(m1 != hmac_mask(k1, c, 1, device_id=b"\x01" * 16)) < 0.55
    assert 0.45 < m1.mean() < 0.55


def test_mask_rejects_bad_parameters():
    c = np.zeros((1, 64), dtype=np.uint8)
    with pytest.raises(ValueError):
        hmac_mask(b"short", c)
    with pytest.raises(ValueError):
        hmac_mask(bytes(32), c, m=257)


def test_legacy_pbkdf2_mask_matches_original_definition():
    key = bytes(range(32))
    c = np.array([[0, 1] * 32], dtype=np.uint8)
    ref = hashlib.pbkdf2_hmac("sha256", key, np.packbits(c[0]).tobytes(), 1000, dklen=8)
    expect = np.unpackbits(np.frombuffer(ref, dtype=np.uint8))
    assert np.array_equal(pbkdf2_mask_legacy(key, c, 64)[0], expect)


def test_known_key_unmasking_recovers_base_response_exactly():
    rng = rng_for("test", "unmask")
    base = XorArbiterPUF(64, 2, rng)
    masked = MaskedPUF(base, new_device_key(rng))
    c = rng.integers(0, 2, size=(3000, 64), dtype=np.uint8)
    r = masked.eval(c)
    assert np.array_equal(masked.unmask(c, r), base.eval(c))
    assert not np.array_equal(r, base.eval(c))


def test_mask_only_control_equals_mask():
    rng = rng_for("test", "zero")
    masked = MaskedPUF(ZeroPUF(64), new_device_key(rng))
    c = rng.integers(0, 2, size=(100, 64), dtype=np.uint8)
    assert np.array_equal(masked.eval(c), masked.mask(c))


def test_mask_preserves_hamming_distance_between_noisy_readings():
    rng = rng_for("test", "hd")
    base = XorArbiterPUF(64, 2, rng, noisiness=0.2)
    masked = MaskedPUF(base, new_device_key(rng))
    c = rng.integers(0, 2, size=(2000, 64), dtype=np.uint8)
    a = masked.eval(c, rng_for("test", "hd", "a"))
    ref = masked.eval(c)
    base_a = base.eval(c, rng_for("test", "hd", "a"))
    assert np.array_equal(a ^ ref, base_a ^ base.eval(c))


@pytest.mark.parametrize("dist", ["uniform", "markov"])
def test_unique_challenges_have_no_duplicates_and_splits_are_disjoint(dist):
    c = unique_challenges(5000, 16, rng_for("test", "uniq", dist), dist=dist, rho=0.7)
    assert c.shape == (5000, 16)
    assert count_duplicates(c) == 0
    train = {r.tobytes() for r in c[:4000]}
    test = {r.tobytes() for r in c[4000:]}
    assert not train & test


def test_markov_challenges_have_stated_adjacent_correlation():
    c = unique_challenges(40000, 64, rng_for("test", "markov"), dist="markov", rho=0.7)
    agree = np.mean(c[:, 1:] == c[:, :-1])
    assert agree == pytest.approx(0.85, abs=0.01)          # (1 + rho) / 2
    assert c.mean() == pytest.approx(0.5, abs=0.01)
