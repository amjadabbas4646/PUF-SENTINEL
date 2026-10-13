import math

import numpy as np
import pytest

from common.seeding import int_seed, rng_for
from common.stats import mean_t_interval, tost_one_sample, wilson_interval
from metrics.quality import (bit_aliasing, flip_rate_profile, mcv_min_entropy,
                             nist_frequency_p, nist_runs_p, nist_summary,
                             pairwise_hd, shannon_entropy_bit, uniformity,
                             uniqueness, uniqueness_device_bootstrap)
from puf_models.arbiter import XorArbiterPUF


def test_uniformity_uniqueness_aliasing_on_known_matrix():
    R = np.array([[0, 0, 1, 1],
                  [0, 1, 1, 0],
                  [1, 1, 1, 1]], dtype=np.uint8)
    assert uniformity(R).tolist() == [0.5, 0.5, 1.0]
    assert bit_aliasing(R).tolist() == pytest.approx([1 / 3, 2 / 3, 1.0, 2 / 3])
    assert sorted(pairwise_hd(R).tolist()) == [0.5, 0.5, 0.5]
    assert uniqueness(R) == 0.5


def test_uniqueness_bootstrap_brackets_point_estimate():
    rng = rng_for("test", "boot")
    R = rng.integers(0, 2, size=(12, 4000), dtype=np.uint8)
    lo, hi = uniqueness_device_bootstrap(R, rng, n_boot=500)
    assert lo < uniqueness(R) < hi and hi - lo < 0.02


def test_entropy_functions():
    assert shannon_entropy_bit(0.5) == 1.0
    assert shannon_entropy_bit(0.0) == 0.0
    assert shannon_entropy_bit(0.25) == pytest.approx(0.8112781, abs=1e-6)
    bits = np.array([1] * 750 + [0] * 250)
    p_u = 0.75 + 2.576 * math.sqrt(0.75 * 0.25 / 999)
    assert mcv_min_entropy(bits) == pytest.approx(-math.log2(p_u))


def test_nist_frequency_example_from_sp800_22():
    # SP 800-22 Rev. 1a section 2.1.8: epsilon = 1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000, P = 0.109599
    eps = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"
    bits = np.array([int(ch) for ch in eps])
    assert nist_frequency_p(bits) == pytest.approx(0.109599, abs=1e-6)


def test_nist_runs_example_from_sp800_22():
    # SP 800-22 Rev. 1a section 2.3.8, same 100-bit sequence, P = 0.500798
    eps = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"
    bits = np.array([int(ch) for ch in eps])
    assert nist_runs_p(bits) == pytest.approx(0.500798, abs=1e-6)


def test_nist_summary_on_ideal_and_biased_data():
    rng = rng_for("test", "nist")
    good = [nist_frequency_p(rng.integers(0, 2, size=4096)) for _ in range(300)]
    s = nist_summary(good)
    assert s["proportion_ok"] and s["pass_proportion"] > 0.96
    bad = [nist_frequency_p((rng.random(4096) < 0.54).astype(int)) for _ in range(300)]
    assert not nist_summary(bad)["proportion_ok"]


def test_flip_rate_of_last_stage_and_range():
    rng = rng_for("test", "flip")
    puf = XorArbiterPUF(64, 1, rng)
    c = rng.integers(0, 2, size=(4000, 64), dtype=np.uint8)
    prof = flip_rate_profile(puf, c)
    assert prof.shape == (64,)
    assert np.all((prof >= 0) & (prof <= 1))
    # flipping stage 1 changes a single feature, flipping stage 32 changes half
    assert prof[0] < prof[31]


def test_wilson_interval_known_value():
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert wilson_interval(0, 100)[0] == pytest.approx(0.0, abs=1e-12)


def test_tost_declares_equivalence_only_inside_margin():
    rng = rng_for("test", "tost")
    near = 0.5 + rng.normal(0, 0.004, size=10)
    far = 0.6 + rng.normal(0, 0.004, size=10)
    assert tost_one_sample(near, 0.5, 0.01)["p_tost"] < 0.05
    assert tost_one_sample(far, 0.5, 0.01)["p_tost"] > 0.95
    wide = 0.5 + rng.normal(0, 0.05, size=4)           # too noisy to conclude
    assert tost_one_sample(wide, 0.5, 0.01)["p_tost"] > 0.05


def test_mean_t_interval():
    r = mean_t_interval([1.0, 2.0, 3.0])
    assert r["mean"] == 2.0 and r["sd"] == 1.0
    assert r["lo"] == pytest.approx(2 - 4.3027 / math.sqrt(3), abs=1e-3)


def test_seeding_is_stable_and_label_dependent():
    assert int_seed("a", 1) == int_seed("a", 1)
    assert int_seed("a", 1) != int_seed("a", 2)
    assert rng_for("x").integers(0, 1 << 30) == rng_for("x").integers(0, 1 << 30)
