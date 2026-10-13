"""Stability-informed model recovery for a (masked) single arbiter chain.

Idea (Delvaux and Verbauwhede 2013; Becker 2015): a challenge whose noise-free
delay difference is small gives an unstable response bit. The adversary
queries each challenge T times and records only whether both bit values were
observed. A deterministic mask M_K(c) is the same in every repetition, so it
cancels: the stability of the masked bit equals the stability of the unmasked
bit. The stability indicator therefore carries information about the weight
vector that the mask does not hide.

Physical likelihood under the simulation model
----------------------------------------------
Let Delta_0(c) = <w, phi(c)> be the noise-free delay difference and let every
evaluation add independent noise eta ~ N(0, sigma^2). One evaluation differs
from the noise-free bit with probability
    q(c) = Phi(-|Delta_0(c)| / sigma).
With T independent evaluations the probability of observing both bit values is
    P[unstable | c] = 1 - (1 - q(c))^T - q(c)^T.                       (exact)
Only the ratio v = w / sigma is identifiable, and only up to sign.

Two models are implemented.

exact      the likelihood above with the parameter v, fitted by maximum
           likelihood (binary cross-entropy on the exact probability). T is
           known to the adversary. From a random start the optimiser often
           stalls: with small |v| the model predicts "unstable" everywhere and
           the gradient carries little directional information. The exact
           model is therefore also run from a warm start, the direction found
           by the logistic surrogate with its length set by a line search on
           the exact likelihood. Both variants are reported.
logistic   the surrogate  P[unstable | c] = sigmoid(a - <u, phi(c)>^2).  It has
           the right qualitative shape (maximal at Delta_0 = 0, decreasing in
           |Delta_0|) but it is not the physical likelihood: its tail is
           exp(-z^2) while the exact tail is about 2 T Phi(-|z|), and its
           value at zero is a free parameter while the exact value is
           1 - 2^(1-T). It was the model of the first revision and is kept for
           comparison.

Becker fits w with CMA-ES by maximising the correlation between |<w, phi>| and
the measured reliability. Both models here are gradient-based variants of that
idea.

The result is scored against the hidden, unmasked, noise-free arbiter bit,
taking the better of the two signs. That score is an oracle diagnostic: it
shows how much of the arbiter model leaks through the mask. It is not the
adversary's accuracy at predicting masked responses, which stays at chance
without the key.
"""
from __future__ import annotations

import math

import numpy as np
import torch

RELIABILITY_CONFIG = {"repeats": 11, "lr": 2e-2, "batch": 256, "max_epochs": 150,
                      "patience": 15, "restarts": 5, "val_fraction": 0.2}
STABILITY_MODELS = ("logistic", "exact_cold", "exact_warm")
_EPS = 1e-6


def exact_unstable_probability(z, repeats: int):
    """P[both bit values seen in T evaluations] for z = Delta_0 / sigma.
    Works on numpy arrays and floats."""
    z = np.abs(np.asarray(z, dtype=float))
    q = 0.5 * np.vectorize(math.erfc)(z / math.sqrt(2.0))
    return 1.0 - (1.0 - q) ** repeats - q ** repeats


class _LogisticStability(torch.nn.Module):
    """Surrogate: returns the logit a - <u, phi>^2."""

    def __init__(self, n_feat: int, gen: torch.Generator, repeats: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n_feat, generator=gen) / math.sqrt(n_feat))
        self.a = torch.nn.Parameter(torch.zeros(1))

    def logit(self, x):
        return self.a - (x @ self.w) ** 2

    def prob(self, x):
        return torch.sigmoid(self.logit(x))


class _ExactStability(torch.nn.Module):
    """Exact likelihood: 1 - (1 - q)^T - q^T with q = Phi(-|<v, phi>|)."""

    def __init__(self, n_feat: int, gen: torch.Generator, repeats: int):
        super().__init__()
        # |<v, phi>| starts at order one, i.e. in the region where the
        # likelihood has usable gradients
        self.w = torch.nn.Parameter(torch.randn(n_feat, generator=gen) / math.sqrt(n_feat))
        self.repeats = repeats

    def prob(self, x):
        z = torch.abs(x @ self.w)
        q = 0.5 * torch.special.erfc(z / math.sqrt(2.0))
        return 1.0 - (1.0 - q) ** self.repeats - q ** self.repeats


_MODEL_CLASS = {"logistic": _LogisticStability, "exact": _ExactStability}


def unstable_indicator(puf, challenges: np.ndarray, repeats: int,
                       noise_rng: np.random.Generator) -> np.ndarray:
    """1 where the T repeated evaluations of `puf` were not all equal."""
    first = puf.eval(challenges, noise_rng)
    unstable = np.zeros(len(first), dtype=bool)
    for _ in range(repeats - 1):
        unstable |= puf.eval(challenges, noise_rng) != first
    return unstable.astype(np.uint8)


def fit_stability_model(phi: np.ndarray, unstable: np.ndarray, seed: int,
                        cfg: dict = RELIABILITY_CONFIG, model: str = "logistic",
                        repeats: int | None = None, init: np.ndarray | None = None,
                        return_info: bool = False):
    """Fit a stability model and return the recovered weight vector. The
    restart with the lowest validation loss is kept; validation uses only
    stability observations, which the adversary has.

    init: optional direction for a warm start of the exact model. Its length
    is set by a line search on the exact training likelihood and a single
    refinement run follows.
    return_info: also return {"val_loss": held-out binary cross-entropy}."""
    repeats = cfg["repeats"] if repeats is None else repeats
    n = phi.shape[0]
    n_val = max(1, int(cfg["val_fraction"] * n))
    X = torch.from_numpy(np.ascontiguousarray(phi, dtype=np.float32))
    y = torch.from_numpy(unstable.astype(np.float32))
    Xtr, ytr, Xval, yval = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    def loss_fn(net, xb, yb):
        if hasattr(net, "logit"):          # numerically stable form for the surrogate
            return torch.nn.functional.binary_cross_entropy_with_logits(net.logit(xb), yb)
        p = torch.clamp(net.prob(xb), _EPS, 1.0 - _EPS)
        return torch.nn.functional.binary_cross_entropy(p, yb)

    warm = None
    if init is not None:
        if model != "exact":
            raise ValueError("a warm start is defined for the exact model only")
        direction = torch.from_numpy(np.asarray(init, dtype=np.float32))
        direction = direction / torch.linalg.norm(direction)
        probe = _ExactStability(phi.shape[1], torch.Generator().manual_seed(seed), repeats)
        best_s, best_l = None, float("inf")
        with torch.no_grad():
            for s in np.logspace(-0.5, 2.5, 49):
                probe.w.copy_(direction * float(s))
                l = float(loss_fn(probe, Xtr, ytr))
                if l < best_l:
                    best_s, best_l = float(s), l
        warm = direction * best_s

    best_loss, best_w = float("inf"), None
    for r in range(1 if warm is not None else cfg["restarts"]):
        gen = torch.Generator().manual_seed(seed + 104729 * r)
        net = _MODEL_CLASS[model](phi.shape[1], gen, repeats)
        if warm is not None:
            with torch.no_grad():
                net.w.copy_(warm)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        run_best, run_state, bad = float("inf"), None, 0
        m = Xtr.shape[0]
        batch = max(16, min(cfg["batch"], m))
        for _ in range(cfg["max_epochs"]):
            perm = torch.randperm(m, generator=gen)
            for s in range(0, m, batch):
                idx = perm[s:s + batch]
                opt.zero_grad()
                loss = loss_fn(net, Xtr[idx], ytr[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                vloss = float(loss_fn(net, Xval, yval))
            if vloss < run_best - 1e-5:
                run_best, bad = vloss, 0
                run_state = net.w.detach().clone().numpy()
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        if run_state is not None and run_best < best_loss:
            best_loss, best_w = run_best, run_state
    if return_info:
        return best_w, {"val_loss": best_loss}
    return best_w


def sign_free_accuracy(w_hat: np.ndarray, phi_test: np.ndarray,
                       hidden_bits: np.ndarray) -> float:
    pred = (phi_test.astype(np.float64) @ w_hat > 0).astype(np.uint8)
    acc = float(np.mean(pred == hidden_bits))
    return max(acc, 1.0 - acc)
