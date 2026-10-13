import numpy as np
import pytest

from common.seeding import rng_for
from error_correction.bch import BCH
from error_correction.error_models import analytic_failure_iid, sample_errors
from error_correction.fuzzy_extractor import CodeOffsetExtractor, CodeSpec
from error_correction.gf import GF
from error_correction.rs import RS


def test_gf_inverse_and_distributivity():
    for m in (7, 8):
        gf = GF(m)
        rng = rng_for("test", "gf", m)
        for _ in range(200):
            a, b, c = (int(x) for x in rng.integers(1, gf.q, size=3))
            assert gf.mul(a, gf.inv(a)) == 1
            assert gf.mul(a, b ^ c) == gf.mul(a, b) ^ gf.mul(a, c)
            assert gf.div(gf.mul(a, b), b) == a


@pytest.mark.parametrize("t,k", [(7, 78), (10, 64), (15, 36)])
def test_bch_127_dimensions_follow_from_construction(t, k):
    code = BCH(7, t)
    assert (code.n, code.k) == (127, k)


@pytest.mark.parametrize("t", [7, 10, 15])
def test_bch_codewords_have_zero_syndrome_and_respect_designed_distance(t):
    code = BCH(7, t)
    rng = rng_for("test", "bch", "cw", t)
    for _ in range(30):
        cw = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        assert not any(code.syndromes(cw))
        if cw.any():
            assert cw.sum() >= 2 * t + 1


@pytest.mark.parametrize("t", [7, 10, 15])
def test_bch_corrects_every_tested_pattern_up_to_t(t):
    code = BCH(7, t)
    rng = rng_for("test", "bch", "ok", t)
    for trial in range(120):
        msg = rng.integers(0, 2, size=code.k, dtype=np.uint8)
        cw = code.encode(msg)
        w = int(rng.integers(0, t + 1)) if trial % 3 else t
        noisy = cw.copy()
        noisy[rng.choice(code.n, size=w, replace=False)] ^= 1
        out, ok = code.decode(noisy)
        assert ok and np.array_equal(out, cw)
        assert np.array_equal(code.extract(out), msg)


@pytest.mark.parametrize("t", [7, 10, 15])
def test_bch_never_recovers_the_codeword_beyond_t(t):
    code = BCH(7, t)
    rng = rng_for("test", "bch", "fail", t)
    for _ in range(120):
        cw = code.encode(rng.integers(0, 2, size=code.k, dtype=np.uint8))
        w = int(rng.integers(t + 1, t + 6))
        noisy = cw.copy()
        noisy[rng.choice(code.n, size=w, replace=False)] ^= 1
        out, ok = code.decode(noisy)
        assert (not ok) or (not np.array_equal(out, cw))


@pytest.mark.parametrize("n,k,t", [(24, 16, 4), (32, 16, 8), (48, 16, 16)])
def test_rs_parameters_and_correction_up_to_t(n, k, t):
    code = RS(n, k)
    assert code.t == t and code.d_min == n - k + 1
    rng = rng_for("test", "rs", "ok", n)
    for trial in range(100):
        msg = rng.integers(0, 256, size=k, dtype=np.uint8)
        cw = code.encode(msg)
        assert not any(code.syndromes(cw))
        w = int(rng.integers(0, t + 1)) if trial % 3 else t
        noisy = cw.copy()
        pos = rng.choice(n, size=w, replace=False)
        noisy[pos] ^= rng.integers(1, 256, size=w, dtype=np.uint8)
        out, ok = code.decode(noisy)
        assert ok and np.array_equal(out, cw)
        assert np.array_equal(code.extract(out), msg)


@pytest.mark.parametrize("n,k,t", [(24, 16, 4), (32, 16, 8), (48, 16, 16)])
def test_rs_never_recovers_the_codeword_beyond_t(n, k, t):
    code = RS(n, k)
    rng = rng_for("test", "rs", "fail", n)
    for _ in range(100):
        cw = code.encode(rng.integers(0, 256, size=k, dtype=np.uint8))
        w = int(rng.integers(t + 1, min(n, t + 5) + 1))
        noisy = cw.copy()
        pos = rng.choice(n, size=w, replace=False)
        noisy[pos] ^= rng.integers(1, 256, size=w, dtype=np.uint8)
        out, ok = code.decode(noisy)
        assert (not ok) or (not np.array_equal(out, cw))


SPECS = [CodeSpec("BCH(127,64,t=10) x2", "BCH", 127, 64, 10, 1, 2),
         CodeSpec("RS(32,16,t=8)", "RS", 32, 16, 8, 8, 1)]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_fuzzy_extractor_roundtrip_and_failure(spec):
    fe = CodeOffsetExtractor(spec)
    rng = rng_for("test", "fe", spec.name)
    w = rng.integers(0, 2, size=spec.puf_bits, dtype=np.uint8)
    key, helper = fe.gen(w, rng)
    assert len(key) == 32 and helper.shape == (spec.blocks, spec.n)
    assert fe.rep(w, helper)[0] == key
    # exactly t symbol errors in the first block: must still succeed
    e = np.zeros(spec.puf_bits, dtype=np.uint8)
    e[np.arange(spec.t) * spec.bits_per_symbol] = 1
    assert fe.block_error_weights(e)[0] == spec.t
    assert fe.rep(w ^ e, helper)[0] == key
    # t + 1 symbol errors in the first block: key must not be recovered
    e[spec.t * spec.bits_per_symbol] = 1
    assert fe.rep(w ^ e, helper)[0] != key
    # a different device does not obtain the key from the same helper data
    w_other = rng.integers(0, 2, size=spec.puf_bits, dtype=np.uint8)
    assert fe.rep(w_other, helper)[0] != key


def test_spec_bookkeeping():
    s = CodeSpec("BCH(127,36,t=15) x4", "BCH", 127, 36, 15, 1, 4)
    assert (s.puf_bits, s.secret_bits, s.helper_bits, s.leakage_bound_bits) == (508, 144, 508, 364)
    r = CodeSpec("RS(48,16,t=16)", "RS", 48, 16, 16, 8, 1)
    assert (r.puf_bits, r.secret_bits, r.helper_bits) == (384, 128, 384)


def test_wrong_spec_is_rejected():
    with pytest.raises(AssertionError):
        CodeOffsetExtractor(CodeSpec("bad", "BCH", 127, 64, 15, 1, 1))


@pytest.mark.parametrize("model", ["iid", "heterogeneous", "common_mode"])
def test_error_models_have_the_requested_mean_ber(model):
    rng = rng_for("test", "errmodel", model)
    p = 0.08
    rate = np.mean([sample_errors(model, p, 508, rng).mean() for _ in range(4000)])
    assert rate == pytest.approx(p, abs=0.004)


def test_analytic_failure_matches_simulated_weights():
    rng = rng_for("test", "analytic")
    spec = SPECS[0]
    fe = CodeOffsetExtractor(spec)
    p, trials = 0.06, 20000
    fails = sum(bool((fe.block_error_weights(sample_errors("iid", p, spec.puf_bits, rng))
                      > spec.t).any()) for _ in range(trials))
    expect = analytic_failure_iid("BCH", 127, 10, 1, 2, p)
    assert fails / trials == pytest.approx(expect, abs=4 * np.sqrt(expect * (1 - expect) / trials))
