"""Checks of the stability likelihood against the simulation model."""
import math

import numpy as np
import pytest

from attacks.reliability import (exact_unstable_probability, fit_stability_model,
                                 sign_free_accuracy, unstable_indicator)
from common.seeding import rng_for
from puf_models.arbiter import XorArbiterPUF, parity_features
from puf_models.challenges import unique_challenges


def test_exact_probability_limits_and_known_values():
    # at Delta_0 = 0 every evaluation is a fair coin: 1 - 2 * (1/2)^T
    for T in (2, 3, 11):
        assert exact_unstable_probability(0.0, T) == pytest.approx(1 - 2.0 ** (1 - T))
    assert exact_unstable_probability(40.0, 11) == pytest.approx(0.0, abs=1e-12)
    # T = 2: both values seen with probability 2 q (1 - q)
    q = 0.5 * math.erfc(1.0 / math.sqrt(2))
    assert exact_unstable_probability(1.0, 2) == pytest.approx(2 * q * (1 - q))
    assert exact_unstable_probability(-1.0, 2) == pytest.approx(2 * q * (1 - q))


def test_single_query_flip_probability_matches_simulation():
    rng = rng_for("test", "stab", "flip")
    nu = 0.2
    puf = XorArbiterPUF(32, 1, rng, noisiness=nu)
    c = unique_challenges(60000, 32, rng_for("test", "stab", "c"))
    z = puf.delays(c)[:, 0] / (nu * math.sqrt(33))
    q = 0.5 * np.vectorize(math.erfc)(np.abs(z) / math.sqrt(2))
    flips = puf.eval(c, rng_for("test", "stab", "noise")) != puf.eval(c)
    # compare observed and predicted flip rates in bins of q
    for lo, hi in ((0.0, 0.01), (0.01, 0.1), (0.1, 0.3), (0.3, 0.5)):
        sel = (q >= lo) & (q < hi)
        assert flips[sel].mean() == pytest.approx(q[sel].mean(), abs=4 * math.sqrt(0.25 / sel.sum()) + 1e-3)


@pytest.mark.parametrize("T", [3, 11])
def test_exact_unstable_probability_matches_simulation(T):
    rng = rng_for("test", "stab", "dev", T)
    nu = 0.2
    puf = XorArbiterPUF(32, 1, rng, noisiness=nu)
    c = unique_challenges(60000, 32, rng_for("test", "stab", "c2", T))
    z = puf.delays(c)[:, 0] / (nu * math.sqrt(33))
    p = exact_unstable_probability(z, T)
    u = unstable_indicator(puf, c, T, rng_for("test", "stab", "noise2", T))
    assert u.mean() == pytest.approx(p.mean(), abs=0.006)
    for lo, hi in ((0.0, 0.05), (0.05, 0.5), (0.5, 1.01)):
        sel = (p >= lo) & (p < hi)
        assert u[sel].mean() == pytest.approx(p[sel].mean(), abs=4 * math.sqrt(0.25 / sel.sum()) + 2e-3)


@pytest.mark.parametrize("model", ["logistic", "exact"])
def test_both_stability_models_recover_the_arbiter(model):
    rng = rng_for("test", "stab", "fit")
    base = XorArbiterPUF(32, 1, rng, noisiness=0.2)
    c = unique_challenges(14000, 32, rng_for("test", "stab", "c3"))
    u = unstable_indicator(base, c[:12000], 11, rng_for("test", "stab", "noise3"))
    w_hat = fit_stability_model(parity_features(c[:12000]), u, seed=5, model=model, repeats=11)
    assert sign_free_accuracy(w_hat, parity_features(c[12000:]), base.eval(c[12000:])) > 0.9


def test_exact_model_recovers_the_weight_scale():
    """Only w / sigma is identifiable; the exact model should estimate its norm."""
    rng = rng_for("test", "stab", "scale")
    nu = 0.2
    base = XorArbiterPUF(32, 1, rng, noisiness=nu)
    c = unique_challenges(20000, 32, rng_for("test", "stab", "c4"))
    u = unstable_indicator(base, c, 11, rng_for("test", "stab", "noise4"))
    v_hat = fit_stability_model(parity_features(c), u, seed=7, model="exact", repeats=11)
    v_true = base.weights[0] / (nu * math.sqrt(33))
    assert np.linalg.norm(v_hat) == pytest.approx(np.linalg.norm(v_true), rel=0.15)
