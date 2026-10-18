import itertools

import numpy as np
import pytest

from dia.matching import (
    UNASSIGNED,
    assign,
    batch_assign,
    batch_slot_of,
    batch_vcg_first_row,
    batch_welfare,
    hungarian,
    vcg_payments,
    welfare,
)


@pytest.mark.parametrize("shape", [(4, 4), (5, 3), (7, 4), (6, 1)])
def test_hungarian_is_optimal(shape):
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = rng.random(shape)
        assert welfare(w, hungarian(w)) == pytest.approx(welfare(w, assign(w)))


def test_hungarian_returns_a_valid_assignment():
    w = np.random.default_rng(1).random((7, 4))
    slots = hungarian(w)
    used = slots[slots != UNASSIGNED]
    assert sorted(used.tolist()) == [0, 1, 2, 3]


def test_hungarian_rejects_more_slots_than_ads():
    with pytest.raises(ValueError):
        hungarian(np.ones((2, 3)))


def test_batch_assign_matches_scipy():
    w = np.random.default_rng(2).random((300, 7, 4))
    choice = batch_assign(w)
    expected = [welfare(w[i], assign(w[i])) for i in range(len(w))]
    np.testing.assert_allclose(batch_welfare(w, choice), expected)


def test_vcg_hand_example():
    # 3 ads, 2 slots, separable values for easy checking
    q = np.array([[0.3, 0.2], [0.3, 0.2], [0.3, 0.2]])
    bids = np.array([3.0, 2.0, 1.0])
    w = q * bids[:, None]
    p = vcg_payments(w)
    # ad 0 takes slot 0, ad 1 slot 1, ad 2 nothing
    # p0 = SW(-0) - (SW - v0) = (0.6 + 0.2) - (0.9 + 0.4 - 0.9) = 0.4
    # p1 = SW(-1) - (SW - v1) = (0.9 + 0.2) - (1.3 - 0.4) = 0.2
    np.testing.assert_allclose(p, [0.4, 0.2, 0.0])


def test_vcg_is_individually_rational():
    rng = np.random.default_rng(3)
    for _ in range(100):
        w = rng.random((7, 4))
        slots = assign(w)
        p = vcg_payments(w, slots)
        for i, j in enumerate(slots):
            value = w[i, j] if j != UNASSIGNED else 0.0
            assert -1e-12 <= p[i] <= value + 1e-12


def test_batched_vcg_matches_reference():
    w = np.random.default_rng(4).random((200, 7, 4))
    choice = batch_assign(w)
    expected = [vcg_payments(w[i])[0] for i in range(len(w))]
    np.testing.assert_allclose(batch_vcg_first_row(w, choice), expected)
    slots = batch_slot_of(0, choice, 7, 4)
    assert np.all((slots >= 0) | (slots == UNASSIGNED))


def test_brute_force_agrees_on_small_case():
    w = np.random.default_rng(5).random((4, 2))
    best = max(w[a, 0] + w[b, 1] for a, b in itertools.permutations(range(4), 2))
    assert welfare(w, hungarian(w)) == pytest.approx(best)
