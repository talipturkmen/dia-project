"""Ad-to-slot matching (Hungarian algorithm) and VCG payments.

The publisher allocates ads to slots by solving a maximum-weight bipartite
matching, where assigning ad ``i`` to slot ``j`` is worth ``q[i, j] * b[i]``.
"""

from __future__ import annotations

import itertools
from functools import cache

import numpy as np
from scipy.optimize import linear_sum_assignment

UNASSIGNED = -1


def hungarian(weights: np.ndarray) -> np.ndarray:
    """Maximum-weight assignment of rows (ads) to columns (slots).

    Shortest augmenting path version of the Hungarian algorithm with dual
    potentials, O(n^2 m). Works for rectangular matrices with at least as many
    rows as columns: every column (slot) is filled, and surplus rows stay
    unassigned.

    Returns ``slot_of_row``: the column of each row, or ``UNASSIGNED``.
    """
    weights = np.asarray(weights, dtype=float)
    n_rows, n_cols = weights.shape
    if n_cols > n_rows:
        raise ValueError("need at least as many ads (rows) as slots (columns)")

    # Minimise cost = -weight. Columns are the "left" side being matched here,
    # so iterate over slots and match each to some ad.
    cost = -weights.T  # (n_cols, n_rows)
    n, m = n_cols, n_rows
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    match = np.zeros(m + 1, dtype=int)  # match[ad + 1] = slot + 1, 0 = free
    way = np.zeros(m + 1, dtype=int)

    for slot in range(1, n + 1):
        match[0] = slot
        ad0 = 0
        min_v = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[ad0] = True
            slot0 = match[ad0]
            delta = np.inf
            ad1 = 0
            for ad in range(1, m + 1):
                if used[ad]:
                    continue
                cur = cost[slot0 - 1, ad - 1] - u[slot0] - v[ad]
                if cur < min_v[ad]:
                    min_v[ad] = cur
                    way[ad] = ad0
                if min_v[ad] < delta:
                    delta = min_v[ad]
                    ad1 = ad
            for ad in range(m + 1):
                if used[ad]:
                    u[match[ad]] += delta
                    v[ad] -= delta
                else:
                    min_v[ad] -= delta
            ad0 = ad1
            if match[ad0] == 0:
                break
        while True:
            ad1 = way[ad0]
            match[ad0] = match[ad1]
            ad0 = ad1
            if ad0 == 0:
                break

    slot_of_row = np.full(n_rows, UNASSIGNED)
    for ad in range(1, m + 1):
        if match[ad] != 0:
            slot_of_row[ad - 1] = match[ad] - 1
    return slot_of_row


def assign(weights: np.ndarray, backend: str = "scipy") -> np.ndarray:
    """Maximum-weight assignment; returns the slot of each ad or ``UNASSIGNED``.

    ``backend="scipy"`` uses ``scipy.optimize.linear_sum_assignment`` (fast, used by
    the experiments); ``backend="hungarian"`` uses this module's implementation.
    Both return an optimal matching; the tests check they agree.
    """
    if backend == "hungarian":
        return hungarian(weights)
    if backend != "scipy":
        raise ValueError(f"unknown backend {backend!r}")
    rows, cols = linear_sum_assignment(weights, maximize=True)
    slot_of_row = np.full(weights.shape[0], UNASSIGNED)
    slot_of_row[rows] = cols
    return slot_of_row


def welfare(weights: np.ndarray, slot_of_row: np.ndarray) -> float:
    """Total value of an assignment under ``weights``."""
    rows = np.flatnonzero(slot_of_row != UNASSIGNED)
    return float(weights[rows, slot_of_row[rows]].sum())


def vcg_payments(weights: np.ndarray, slot_of_row: np.ndarray | None = None) -> np.ndarray:
    """Expected VCG payment of every ad.

    ``p_i = SW*(without i) - (SW*(all ads) - value of i in the best allocation)``.
    Unassigned ads pay 0. Divide by the ad's click probability in its slot to get
    the pay-per-click price.
    """
    if slot_of_row is None:
        slot_of_row = assign(weights)
    best = welfare(weights, slot_of_row)
    payments = np.zeros(weights.shape[0])
    for i in np.flatnonzero(slot_of_row != UNASSIGNED):
        others = np.delete(weights, i, axis=0)
        best_without_i = welfare(others, assign(others))
        value_i = weights[i, slot_of_row[i]]
        payments[i] = best_without_i - (best - value_i)
    return np.maximum(payments, 0.0)


# --- Batched exact matching ---------------------------------------------------
#
# The experiments solve hundreds of thousands of small matchings (7 ads x 4
# slots). With so few ads, enumerating every injective ad->slot map (840 of
# them) and taking the best is exact and vectorises across auctions.


@cache
def injections(n_rows: int, n_cols: int) -> np.ndarray:
    """All ways to fill ``n_cols`` slots with distinct rows: array (P, n_cols) of row ids."""
    return np.array(list(itertools.permutations(range(n_rows), n_cols)), dtype=int)


def batch_assign(weights: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Best injection index for each auction. ``weights`` has shape (N, rows, slots)."""
    n, n_rows, n_cols = weights.shape
    perms = injections(n_rows, n_cols)
    cols = np.arange(n_cols)
    best = np.empty(n, dtype=int)
    for start in range(0, n, chunk):
        w = weights[start : start + chunk]
        totals = w[:, perms, cols].sum(axis=-1)  # (chunk, P)
        best[start : start + chunk] = np.argmax(totals, axis=1)
    return best


def batch_welfare(weights: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """Value of each auction's chosen injection under ``weights`` (N, rows, slots)."""
    n, n_rows, n_cols = weights.shape
    rows = injections(n_rows, n_cols)[choice]  # (N, slots)
    return weights[np.arange(n)[:, None], rows, np.arange(n_cols)].sum(axis=-1)


def batch_slot_of(row: int, choice: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Slot given to ``row`` in each auction, or ``UNASSIGNED``."""
    rows = injections(n_rows, n_cols)[choice]  # (N, slots)
    hit = rows == row
    return np.where(hit.any(axis=1), hit.argmax(axis=1), UNASSIGNED)


def batch_vcg_first_row(weights: np.ndarray, choice: np.ndarray) -> np.ndarray:
    """Expected VCG payment of row 0 (our ad) in each auction; 0 when not shown."""
    n, n_rows, n_cols = weights.shape
    best = batch_welfare(weights, choice)
    others = weights[:, 1:, :]
    best_without = batch_welfare(others, batch_assign(others))
    slot = batch_slot_of(0, choice, n_rows, n_cols)
    shown = slot != UNASSIGNED
    own_value = np.where(shown, weights[np.arange(n), 0, np.maximum(slot, 0)], 0.0)
    return np.where(shown, np.maximum(best_without - (best - own_value), 0.0), 0.0)
