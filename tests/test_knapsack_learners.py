import itertools

import numpy as np
import pytest

from dia.knapsack import allocate_budget
from dia.learners import GPTS, SlotTS, SlotUCB


def brute_force(values):
    n_sub, n_levels = values.shape
    best = -np.inf
    for combo in itertools.product(range(n_levels), repeat=n_sub):
        if sum(combo) <= n_levels - 1:
            best = max(best, sum(values[s, g] for s, g in enumerate(combo)))
    return best


def test_knapsack_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(40):
        values = np.cumsum(rng.random((3, 6)), axis=1)
        values[:, 0] = 0
        levels, total = allocate_budget(values)
        assert levels.sum() <= values.shape[1] - 1
        assert total == pytest.approx(sum(values[s, g] for s, g in enumerate(levels)))
        assert total == pytest.approx(brute_force(values))


def test_knapsack_handles_non_monotone_values():
    values = np.array([[0, 5, 1, 1], [0, 0, 0, 9]], dtype=float)
    levels, total = allocate_budget(values)
    assert total == 9 and levels.tolist() == [0, 3]


def test_ucb_is_optimistic_for_unseen_slots():
    learner = SlotUCB(4)
    learner.update(np.array([100.0, 0, 0, 0]), np.array([10.0, 0, 0, 0]))
    index = learner.index(np.random.default_rng(0))
    assert index[0] > 0.1
    assert np.all(index[1:] == 1.0)


def test_ts_concentrates_with_data():
    learner = SlotTS(2)
    learner.update(np.array([5000.0, 5000.0]), np.array([500.0, 1500.0]))
    samples = np.array([learner.index(np.random.default_rng(s)) for s in range(50)])
    assert abs(samples[:, 0].mean() - 0.1) < 0.01
    assert abs(samples[:, 1].mean() - 0.3) < 0.01


def test_gpts_fits_observations_and_anchors_zero():
    arms = np.linspace(0, 1, 11)
    gp = GPTS(arms, scale=10.0)
    truth = 10 * np.sqrt(arms)
    for _ in range(20):
        gp.update(np.arange(1, 11), truth[1:])
    assert gp.posterior_mean[0] == pytest.approx(0.0, abs=1e-3)
    np.testing.assert_allclose(gp.posterior_mean[1:], truth[1:], atol=0.3)
    assert gp.std[5] < 0.1


def test_gpts_prior_is_wide_without_data():
    gp = GPTS(np.linspace(0, 1, 5), scale=1.0, anchor_zero=False)
    assert np.allclose(gp.std, 1.0)
    assert gp.sample(np.random.default_rng(0)).shape == (5,)
