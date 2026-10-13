"""Attack sanity tests. They use small PUFs so the suite stays fast."""
import numpy as np
import pytest

from attacks.models import ATTACKS, fit_predict
from attacks.reliability import (fit_stability_model, sign_free_accuracy,
                                 unstable_indicator)
from common.seeding import rng_for
from puf_models.arbiter import XorArbiterPUF, parity_features
from puf_models.challenges import unique_challenges
from puf_models.mask import MaskedPUF, new_device_key


def _data(puf, n_stage, n_train, n_val, n_test, label):
    c = unique_challenges(n_train + n_val + n_test, n_stage, rng_for("test", "atk", label))
    phi, y = parity_features(c), puf.eval(c)
    a, b = n_train, n_train + n_val
    return phi[:a], y[:a], phi[a:b], y[a:b], phi[b:], y[b:]


def test_logistic_regression_learns_single_arbiter():
    puf = XorArbiterPUF(32, 1, rng_for("test", "atk", "dev1"))
    Xtr, ytr, Xv, yv, Xte, yte = _data(puf, 32, 1600, 400, 2000, "lr")
    out = fit_predict("lr", Xtr, ytr, Xv, yv, Xte, seed=1)
    assert np.mean(out["test_pred"] == yte) > 0.95


def test_linear_model_fails_and_product_model_succeeds_on_2xor():
    puf = XorArbiterPUF(32, 2, rng_for("test", "atk", "dev2"))
    Xtr, ytr, Xv, yv, Xte, yte = _data(puf, 32, 4000, 1000, 2000, "xor")
    lin = fit_predict("lr", Xtr, ytr, Xv, yv, Xte, seed=1)
    prod = fit_predict("lr_xor", Xtr, ytr, Xv, yv, Xte, seed=1, k_model=2)
    assert np.mean(lin["test_pred"] == yte) < 0.62
    assert np.mean(prod["test_pred"] == yte) > 0.93


def test_masked_response_is_not_learned_but_unmasked_labels_are():
    rng = rng_for("test", "atk", "dev3")
    base = XorArbiterPUF(32, 1, rng)
    masked = MaskedPUF(base, new_device_key(rng))
    c = unique_challenges(6000, 32, rng_for("test", "atk", "maskc"))
    phi, y = parity_features(c), masked.eval(c)
    out = fit_predict("lr", phi[:3200], y[:3200], phi[3200:4000], y[3200:4000], phi[4000:], seed=1)
    assert abs(np.mean(out["test_pred"] == y[4000:]) - 0.5) < 0.05
    y_or = masked.unmask(c, y)
    out = fit_predict("lr", phi[:3200], y_or[:3200], phi[3200:4000], y_or[3200:4000], phi[4000:], seed=1)
    assert np.mean(out["test_pred"] == y_or[4000:]) > 0.95


@pytest.mark.parametrize("name", ATTACKS)
def test_every_attack_runs_and_is_deterministic(name):
    puf = XorArbiterPUF(16, 1, rng_for("test", "atk", "dev4"))
    Xtr, ytr, Xv, yv, Xte, yte = _data(puf, 16, 400, 100, 300, "all")
    a = fit_predict(name, Xtr, ytr, Xv, yv, Xte, seed=3, k_model=1)
    b = fit_predict(name, Xtr, ytr, Xv, yv, Xte, seed=3, k_model=1)
    assert a["test_pred"].shape == (300,)
    assert np.array_equal(a["test_pred"], b["test_pred"])
    assert np.mean(a["test_pred"] == yte) > 0.6


def test_stability_of_masked_bit_equals_stability_of_base_bit():
    rng = rng_for("test", "rel", "dev")
    base = XorArbiterPUF(32, 1, rng, noisiness=0.2)
    masked = MaskedPUF(base, new_device_key(rng))
    c = unique_challenges(3000, 32, rng_for("test", "rel", "c"))
    u_masked = unstable_indicator(masked, c, 7, rng_for("test", "rel", "noise"))
    u_base = unstable_indicator(base, c, 7, rng_for("test", "rel", "noise"))
    assert np.array_equal(u_masked, u_base)
    assert 0.05 < u_base.mean() < 0.6


def test_reliability_attack_recovers_masked_arbiter_up_to_sign():
    rng = rng_for("test", "rel", "dev2")
    base = XorArbiterPUF(32, 1, rng, noisiness=0.2)
    masked = MaskedPUF(base, new_device_key(rng))
    c = unique_challenges(14000, 32, rng_for("test", "rel", "c2"))
    u = unstable_indicator(masked, c[:12000], 11, rng_for("test", "rel", "noise2"))
    w_hat = fit_stability_model(parity_features(c[:12000]), u, seed=5)
    acc = sign_free_accuracy(w_hat, parity_features(c[12000:]), base.eval(c[12000:]))
    assert acc > 0.9
