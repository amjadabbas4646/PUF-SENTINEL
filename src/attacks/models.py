"""Modelling attacks with one common interface.

Every attack receives the adversary's CRPs already split into a training part
and a validation part (80 % and 20 % of the budget), as parity features. The
validation part is used for early stopping, for choosing among restarts and,
in the experiment driver, for choosing the "best" attack. The test set is never
seen here: `fit_predict` returns predictions for it and nothing is tuned on it.

Attacks
-------
lr        logistic regression on parity features (scikit-learn, L-BFGS)
svm       linear support vector machine on parity features
rf        random forest
gb        histogram gradient boosting (scikit-learn)
mlp       multilayer perceptron, three tanh hidden layers of 64 units, Adam,
          early stopping on validation loss
lr_xor    product-of-linear-models logistic regression of Ruehrmair et al.,
          the attack that matches the algebraic structure of a k-XOR arbiter
          PUF. It is a surrogate classifier, not the response model: the score
          g(c) = prod_j <theta_j, phi(c)> is trained with the logistic loss on
          the label r and the prediction is r = 1 when g(c) > 0. For noise-free
          responses r = 1 exactly when (-1)^(k+1) prod_j <w_j, phi(c)> > 0, so
          a perfect surrogate exists with theta_1 = (-1)^(k+1) w_1 and
          theta_j = w_j otherwise. The theta_j are identified only up to
          positive rescaling with product one, a permutation and an even
          number of sign changes. Several random restarts, the restart with
          the best validation accuracy is kept.

Besides accuracy, fit_predict reports how the optimiser behaved: "epochs"
(passes over the training data, summed over restarts, for the gradient
attacks), "iterations" (solver iterations for the scikit-learn models) and
"restarts" (restarts actually used by the product-model attack).

Hyperparameters are fixed in ATTACK_CONFIG and were not tuned per construction.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ATTACK_CONFIG = {
    "lr": {"C": 10.0, "max_iter": 2000},
    "svm": {"C": 1.0, "max_iter": 5000},
    "rf": {"n_estimators": 100},
    "gb": {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31},
    "mlp": {"hidden": [64, 64, 64], "lr": 3e-3, "batch": 256, "max_epochs": 150,
            "patience": 12},
    "lr_xor": {"lr": 2e-2, "batch": 256, "max_epochs": 100, "patience": 10,
               "restarts": 5},
}
ATTACKS = tuple(ATTACK_CONFIG)

torch.set_num_threads(1)


def _train_torch(model: torch.nn.Module, Xtr, ytr, Xval, yval, seed: int,
                 lr: float, batch: int, max_epochs: int, patience: int,
                 info: dict | None = None):
    """Minibatch Adam with early stopping on validation loss. Returns the best
    validation accuracy; the model is left holding the best weights. If `info`
    is given, the number of epochs run and whether the epoch limit was hit are
    added to it."""
    gen = torch.Generator().manual_seed(seed)
    Xtr_t = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    ytr_t = torch.from_numpy(np.ascontiguousarray(ytr, dtype=np.float32))
    Xval_t = torch.from_numpy(np.ascontiguousarray(Xval, dtype=np.float32))
    yval_t = torch.from_numpy(np.ascontiguousarray(yval, dtype=np.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    n = Xtr_t.shape[0]
    batch = max(16, min(batch, n))
    best_loss, best_state, best_acc, bad = float("inf"), None, 0.5, 0
    epochs_run, stopped_early = 0, False
    for _ in range(max_epochs):
        epochs_run += 1
        model.train()
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]).squeeze(-1), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xval_t).squeeze(-1)
            vloss = float(loss_fn(out, yval_t))
            vacc = float(((out > 0).float() == yval_t).float().mean())
        if vloss < best_loss - 1e-5:
            best_loss, best_acc, bad = vloss, vacc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                stopped_early = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    if info is not None:
        info["epochs"] = info.get("epochs", 0) + epochs_run
        info["hit_epoch_limit"] = info.get("hit_epoch_limit", 0) + (not stopped_early)
    return best_acc


def _predict_torch(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)))
    return (out.squeeze(-1) > 0).numpy().astype(np.uint8)


class _ProductLR(torch.nn.Module):
    def __init__(self, n_feat: int, k: int, gen: torch.Generator):
        super().__init__()
        w = torch.randn(k, n_feat, generator=gen) / np.sqrt(n_feat)
        self.w = torch.nn.Parameter(w)

    def forward(self, x):
        return torch.prod(x @ self.w.T, dim=1, keepdim=True)


def _mlp(n_feat: int, hidden: list[int], seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    layers, d = [], n_feat
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.Tanh()]
        d = h
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers)


def product_lr_single_restart(Xtr, ytr, Xval, yval, Xte, seed: int, k_model: int):
    """One restart of the product-model attack with the standard settings.
    Used by the restart study (E5d) to separate convergence failure from
    resistance. Returns validation accuracy and test predictions."""
    cfg = ATTACK_CONFIG["lr_xor"]
    gen = torch.Generator().manual_seed(seed)
    model = _ProductLR(Xtr.shape[1], max(1, k_model), gen)
    _train_torch(model, Xtr, ytr, Xval, yval, seed, cfg["lr"], cfg["batch"],
                 cfg["max_epochs"], cfg["patience"])
    vp = _predict_torch(model, Xval)
    return {"val_acc": float(np.mean(vp == yval)), "test_pred": _predict_torch(model, Xte)}


def fit_predict(name: str, Xtr, ytr, Xval, yval, Xte, seed: int, k_model: int = 1):
    """Train attack `name` and return a dict with validation accuracy, test
    predictions and wall-clock training time in seconds."""
    cfg = ATTACK_CONFIG[name]
    t0 = time.perf_counter()
    info = {"epochs": 0, "iterations": 0, "restarts": 1, "hit_epoch_limit": 0}
    single_class = np.unique(ytr).size < 2
    if single_class:                      # degenerate tiny training sets
        const = int(ytr[0])
        val_pred = np.full(len(yval), const, dtype=np.uint8)
        te_pred = np.full(len(Xte), const, dtype=np.uint8)
    elif name in ("lr", "svm", "rf", "gb"):
        if name == "lr":
            clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"])
        elif name == "svm":
            clf = LinearSVC(C=cfg["C"], max_iter=cfg["max_iter"], random_state=seed)
        elif name == "rf":
            clf = RandomForestClassifier(n_estimators=cfg["n_estimators"], n_jobs=1,
                                         random_state=seed)
        else:
            clf = HistGradientBoostingClassifier(
                max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                max_leaf_nodes=cfg["max_leaf_nodes"], early_stopping=False,
                random_state=seed)
        clf.fit(Xtr, ytr)
        if name == "rf":
            info["iterations"] = cfg["n_estimators"]
        else:
            info["iterations"] = int(np.max(np.atleast_1d(clf.n_iter_)))
        val_pred = clf.predict(Xval).astype(np.uint8)
        te_pred = clf.predict(Xte).astype(np.uint8)
    elif name == "mlp":
        model = _mlp(Xtr.shape[1], cfg["hidden"], seed)
        _train_torch(model, Xtr, ytr, Xval, yval, seed, cfg["lr"], cfg["batch"],
                     cfg["max_epochs"], cfg["patience"], info)
        val_pred, te_pred = _predict_torch(model, Xval), _predict_torch(model, Xte)
    elif name == "lr_xor":
        best = (-1.0, None, None)
        for r in range(cfg["restarts"]):
            gen = torch.Generator().manual_seed(seed + 7919 * r)
            model = _ProductLR(Xtr.shape[1], max(1, k_model), gen)
            _train_torch(model, Xtr, ytr, Xval, yval, seed + r, cfg["lr"],
                         cfg["batch"], cfg["max_epochs"], cfg["patience"], info)
            info["restarts"] = r + 1
            vp = _predict_torch(model, Xval)
            acc = float(np.mean(vp == yval))
            if acc > best[0]:
                best = (acc, vp, _predict_torch(model, Xte))
            if acc > 0.98:                # converged, further restarts add nothing
                break
        _, val_pred, te_pred = best
    else:
        raise ValueError(f"unknown attack {name!r}")
    return {"val_acc": float(np.mean(val_pred == yval)), "test_pred": te_pred,
            "seconds": time.perf_counter() - t0, **info}
