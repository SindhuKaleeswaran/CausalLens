"""Optional IHDP estimators. Install the experiments extra to fit these models."""

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .base import BaseCausalEstimator


class HonestCausalForest(BaseCausalEstimator):
    """Cross-fitted residualization followed by EconML's honest generalized random forest."""

    def __init__(self, n_estimators=128, random_state=42):
        super().__init__()
        self.n_estimators = n_estimators
        self.random_state = random_state

    @property
    def assumptions(self):
        return {
            "identification": "Conditional exchangeability, consistency, overlap and no interference.",
            "forest": "Honest splits, cross-fitted outcome and treatment nuisances, local moment estimation.",
        }

    def fit(self, X, T, Y):
        from econml.dml import CausalForestDML

        x, t, y = self._validate_inputs(X, T, Y)
        if min(np.bincount(t.astype(int))) < 3:
            raise ValueError(
                "Three-fold cross-fitting requires three observations per arm"
            )
        self.forest_ = CausalForestDML(
            model_y=RidgeCV(),
            model_t=LogisticRegression(max_iter=1000),
            discrete_treatment=True,
            categories=[0, 1],
            honest=True,
            inference=False,
            n_estimators=self.n_estimators,
            criterion="het",
            min_samples_leaf=5,
            max_samples=0.45,
            n_jobs=1,
            random_state=self.random_state,
            cv=StratifiedKFold(3, shuffle=True, random_state=self.random_state),
        ).fit(y, t.astype(int), X=x)
        self.n_features_in_ = x.shape[1]
        self.training_sample_size_ = len(x)
        self.training_treated_count_ = int(t.sum())
        self.training_control_count_ = len(x) - int(t.sum())
        self.fitted_ = True
        self.training_ate_ = self.estimate_ate(x)
        return self

    def estimate_ite(self, X):
        self._check_is_fitted()
        x, _, _ = self._validate_inputs(X)
        return self.forest_.effect(x, T0=0, T1=1)

    def predict_potential_outcomes(self, X):
        raise NotImplementedError(
            "This forest identifies the conditional contrast. Outcome levels require a separate model."
        )


def linear_mmd_squared(control, treated):
    """Squared MMD for the linear kernel, with equal arm priors."""
    return (treated.mean(dim=0) - control.mean(dim=0)).square().sum()


class CFRNetMMD(BaseCausalEstimator):
    """CFR with a normalized shared representation, separate response heads and linear MMD."""

    def __init__(
        self,
        width=32,
        epochs=100,
        learning_rate=0.005,
        balance_weight=1.0,
        random_state=42,
    ):
        super().__init__()
        self.width = width
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.balance_weight = balance_weight
        self.random_state = random_state

    @property
    def assumptions(self):
        return {
            "identification": "Conditional exchangeability, consistency, overlap and no interference.",
            "representation": "Factual prediction and representation balance jointly determine the fitted contrast.",
        }

    def fit(self, X, T, Y):
        import torch
        from torch import nn

        x, t, y = self._validate_inputs(X, T, Y)
        if (
            self.epochs < 1
            or self.width < 1
            or self.balance_weight < 0
            or self.learning_rate <= 0
        ):
            raise ValueError("Invalid CFRNet architecture or optimization parameters")
        self.x_scaler_ = StandardScaler().fit(x)
        self.y_scaler_ = StandardScaler().fit(y[:, None])
        features = torch.tensor(self.x_scaler_.transform(x), dtype=torch.float32)
        target = torch.tensor(
            self.y_scaler_.transform(y[:, None]).ravel(), dtype=torch.float32
        )
        treated = torch.tensor(t == 1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)

            def head():
                return nn.Sequential(
                    nn.Linear(self.width, self.width),
                    nn.ELU(),
                    nn.Linear(self.width, 1),
                )

            self.network_ = nn.ModuleDict(
                {
                    "representation": nn.Sequential(
                        nn.Linear(x.shape[1], self.width),
                        nn.ELU(),
                        nn.Linear(self.width, self.width),
                    ),
                    "control": head(),
                    "treated": head(),
                }
            )
            optimizer = torch.optim.Adam(
                self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4
            )
            for _ in range(self.epochs):
                optimizer.zero_grad()
                representation = nn.functional.normalize(
                    self.network_["representation"](features), dim=1
                )
                z0, z1 = representation[~treated], representation[treated]
                err0 = self.network_["control"](z0).flatten() - target[~treated]
                err1 = self.network_["treated"](z1).flatten() - target[treated]
                factual = (err0.square().mean() + err1.square().mean()) / 2
                penalty = linear_mmd_squared(z0, z1)
                loss = factual + self.balance_weight * penalty
                loss.backward()
                optimizer.step()
        self.last_factual_loss_ = float(factual.detach())
        self.last_mmd_squared_ = float(penalty.detach())
        self.training_sample_size_ = len(x)
        self.training_treated_count_ = int(t.sum())
        self.training_control_count_ = len(x) - int(t.sum())
        self.n_features_in_ = x.shape[1]
        self.fitted_ = True
        self.training_ate_ = self.estimate_ate(x)
        return self

    def predict_potential_outcomes(self, X: Any):
        import torch

        self._check_is_fitted()
        x, _, _ = self._validate_inputs(X)
        features = torch.tensor(self.x_scaler_.transform(x), dtype=torch.float32)
        with torch.no_grad():
            z = torch.nn.functional.normalize(
                self.network_["representation"](features), dim=1
            )
            return tuple(
                self.y_scaler_.inverse_transform(self.network_[arm](z).numpy())
                .ravel()
                .astype(float)
                for arm in ("control", "treated")
            )

    def estimate_ite(self, X):
        y0, y1 = self.predict_potential_outcomes(X)
        return y1 - y0


def get_experimental_models():
    return {
        "Causal Forest (honest DML)": HonestCausalForest(),
        "CFRNet (linear MMD)": CFRNetMMD(),
    }
