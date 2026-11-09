"""Optional scientific regression tests for the separate eight-estimator experiment."""

import numpy as np
import pytest
from sklearn.base import clone

from causallens.models.experimental import (
    CFRNetMMD,
    HonestCausalForest,
    linear_mmd_squared,
)


def sample():
    rng = np.random.default_rng(901)
    x = rng.normal(size=(240, 4))
    t = rng.binomial(1, 0.5, size=len(x))
    y = x[:, 1] + (2 + x[:, 0]) * t + rng.normal(scale=0.05, size=len(x))
    return x, t, y


def test_honest_forest_uses_orthogonalized_generalized_forest():
    from importlib.util import find_spec

    if find_spec("econml") is None:
        pytest.skip("Install experiments extra to exercise EconML")
    from econml.dml import CausalForestDML

    x, t, y = sample()
    model = HonestCausalForest(n_estimators=32).fit(x, t, y)
    assert isinstance(model.forest_, CausalForestDML)
    assert model.forest_.honest and model.forest_.criterion == "het"
    pred = model.estimate_ite(x)
    np.testing.assert_array_equal(pred, clone(model).fit(x, t, y).estimate_ite(x))
    assert pred.mean() == pytest.approx(2, abs=0.4)
    assert np.corrcoef(pred, x[:, 0])[0, 1] > 0.5
    with pytest.raises(NotImplementedError):
        model.predict_potential_outcomes(x)


def test_cfr_has_active_balance_gradient_and_two_response_heads():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    z0 = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    z1 = torch.tensor([[2.0, 2.0], [2.0, 2.0]], requires_grad=True)
    loss = linear_mmd_squared(z0, z1)
    assert loss.item() == 5
    loss.backward()
    np.testing.assert_allclose(z1.grad.numpy(), [[1.0, 2.0], [1.0, 2.0]])
    x, t, y = sample()
    model = CFRNetMMD().fit(x, t, y)
    y0, y1 = model.predict_potential_outcomes(x)
    effect = model.estimate_ite(x)
    np.testing.assert_allclose(y1 - y0, effect)
    np.testing.assert_array_equal(effect, clone(model).fit(x, t, y).estimate_ite(x))
    assert np.sqrt(np.mean((effect - 2 - x[:, 0]) ** 2)) < 0.55
    ablated = clone(model).set_params(balance_weight=0).fit(x, t, y)
    assert not np.allclose(effect, ablated.estimate_ite(x))
