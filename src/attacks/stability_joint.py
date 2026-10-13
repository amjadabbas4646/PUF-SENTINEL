"""Stability attacks on constructions with several arbiter evaluations per
response: the k-XOR arbiter PUF behind a keyed mask and the keyless
hashed-output class H of puf_models/constructions.py.

The adversary of type A2 queries each of N challenges T times and records only
the indicator z(c) that the T answers were not all equal. A deterministic
function of the challenge that is XORed onto the response, such as the keyed
mask, is the same in every repetition and cancels in z.

Joint surrogate for the k-XOR arbiter PUF
-----------------------------------------
The XOR output is unstable when an odd number of chains flip, which for small
flip probabilities is close to "some chain is unstable". With the single-chain
surrogate s_j(c) = sigmoid(a_j - <u_j, phi(c)>^2) of attacks/reliability.py,

    P[z(c) = 1]  ~  1 - prod_j (1 - s_j(c)).

All k weight vectors are fitted together by minimising the binary
cross-entropy with Adam. Becker (CHES 2015) recovers the chains one at a time
with an evolution strategy that maximises a correlation; the joint gradient
fit is a different optimiser for the same side information. The model is a
surrogate, not the physical likelihood (the same caveats as for one chain
apply, and simultaneous flips of two chains are ignored).

Score: the recovered vectors define the bit [prod_j <u_j, phi(c)> > 0]. It is
compared with the hidden noise-free XOR bit, taking the better of the two
global signs, because stability does not fix the sign of any u_j.

Surrogate for class H (m PUF bits hashed into one response bit)
---------------------------------------------------------------
The hashed bit can change only when at least one of the m PUF bits is
unstable, and then it changes with a probability below one, so

    P[z(c) = 1]  ~  g * (1 - prod_{i=1..m} (1 - s(c_i))),   0 < g < 1,

with one shared weight vector u for the m public sub-challenges c_i and a free
factor g. Because H has no secret besides the PUF, a recovered u gives a
complete model: the adversary evaluates the public pipeline with the model
bits. The global sign of u is chosen on the adversary's own CRPs.
"""
from __future__ import annotations

import math

import numpy as np
import torch

JOINT_CONFIG = {"lr": 2e-2, "batch": 256, "max_epochs": 150, "patience": 15,
                "restarts": 8, "val_fraction": 0.2}
_EPS = 1e-6

torch.set_num_threads(1)


class _JointXor(torch.nn.Module):
    def __init__(self, n_feat: int, k: int, gen: torch.Generator):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(k, n_feat, generator=gen) / math.sqrt(n_feat))
        self.a = torch.nn.Parameter(torch.zeros(k))

    def prob(self, x):                               # x: (N, n_feat)
        s = torch.sigmoid(self.a - (x @ self.w.T) ** 2)
        return 1.0 - torch.prod(1.0 - s, dim=1)


class _HashedOutput(torch.nn.Module):
    def __init__(self, n_feat: int, k: int, gen: torch.Generator):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(n_feat, generator=gen) / math.sqrt(n_feat))
        self.a = torch.nn.Parameter(torch.zeros(1))
        self.g = torch.nn.Parameter(torch.full((1,), 2.0))

    def prob(self, x):                               # x: (N, m, n_feat)
        s = torch.sigmoid(self.a - (x @ self.w) ** 2)
        return torch.sigmoid(self.g) * (1.0 - torch.prod(1.0 - s, dim=1))


def _fit(net_class, phi: np.ndarray, unstable: np.ndarray, k: int, seed: int, cfg: dict):
    n = phi.shape[0]
    n_val = max(1, int(cfg["val_fraction"] * n))
    X = torch.from_numpy(np.ascontiguousarray(phi, dtype=np.float32))
    y = torch.from_numpy(unstable.astype(np.float32))
    Xtr, ytr, Xval, yval = X[n_val:], y[n_val:], X[:n_val], y[:n_val]

    def loss_fn(net, xb, yb):
        return torch.nn.functional.binary_cross_entropy(
            torch.clamp(net.prob(xb), _EPS, 1.0 - _EPS), yb)

    best_loss, best_w = float("inf"), None
    losses, epochs = [], 0
    for r in range(cfg["restarts"]):
        gen = torch.Generator().manual_seed(seed + 104729 * r)
        net = net_class(phi.shape[-1], k, gen)
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        run_best, run_state, bad = float("inf"), None, 0
        m = Xtr.shape[0]
        batch = max(16, min(cfg["batch"], m))
        for _ in range(cfg["max_epochs"]):
            epochs += 1
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
        losses.append(run_best)
        if run_state is not None and run_best < best_loss:
            best_loss, best_w = run_best, run_state
    return best_w, {"val_loss": best_loss, "restart_val_losses": losses, "epochs": epochs,
                    "restarts": cfg["restarts"]}


def fit_xor_stability(phi: np.ndarray, unstable: np.ndarray, k: int, seed: int,
                      cfg: dict = JOINT_CONFIG):
    """Fit k chains to the stability indicator of a k-XOR arbiter PUF.
    phi: (N, n+1) parity features. Returns ((k, n+1) weights, info)."""
    return _fit(_JointXor, phi, unstable, k, seed, cfg)


def fit_hashed_stability(phi_sub: np.ndarray, unstable: np.ndarray, seed: int,
                         cfg: dict = JOINT_CONFIG):
    """Fit one chain to the stability indicator of class H.
    phi_sub: (N, m, n+1) parity features of the public sub-challenges.
    Returns ((n+1,) weights, info)."""
    return _fit(_HashedOutput, phi_sub, unstable, 1, seed, cfg)


def xor_model_bits(weights: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Bit [prod_j <u_j, phi> > 0] of a recovered k-chain model."""
    return (np.prod(phi.astype(np.float64) @ np.atleast_2d(weights).T, axis=1) > 0).astype(np.uint8)


def sign_free_agreement(pred_bits: np.ndarray, hidden_bits: np.ndarray) -> float:
    acc = float(np.mean(pred_bits == hidden_bits))
    return max(acc, 1.0 - acc)


def matched_cosines(weights: np.ndarray, true_weights: np.ndarray) -> list[float]:
    """For every true chain, the largest absolute cosine with a recovered chain."""
    w = np.atleast_2d(weights)
    t = np.atleast_2d(true_weights)
    cos = np.abs(t @ w.T) / (np.linalg.norm(t, axis=1)[:, None] * np.linalg.norm(w, axis=1)[None, :])
    return [float(x) for x in cos.max(axis=1)]
