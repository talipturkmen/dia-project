"""Multiple-choice knapsack for splitting a daily budget across subcampaigns."""

from __future__ import annotations

import numpy as np


def allocate_budget(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Pick one budget level per subcampaign to maximise total value.

    ``values[s, g]`` is the value of giving subcampaign ``s`` the budget level
    ``g`` (``g`` units of the budget grid step). Levels across subcampaigns must
    sum to at most ``values.shape[1] - 1`` (the whole daily budget).

    Dynamic programme over subcampaigns, O(S * G^2). Returns the chosen level per
    subcampaign and the optimal total value.
    """
    values = np.asarray(values, dtype=float)
    n_sub, n_levels = values.shape
    best = np.zeros(n_levels)  # best[g]: best value of the first s subcampaigns using exactly <= g levels
    choice = np.zeros((n_sub, n_levels), dtype=int)

    for s in range(n_sub):
        new_best = np.full(n_levels, -np.inf)
        for g in range(n_levels):
            # give subcampaign s level k, the previous ones at most g - k
            candidates = values[s, : g + 1] + best[g::-1]
            k = int(np.argmax(candidates))
            new_best[g] = candidates[k]
            choice[s, g] = k
        best = new_best

    levels = np.zeros(n_sub, dtype=int)
    g = int(np.argmax(best))
    total = float(best[g])
    for s in range(n_sub - 1, -1, -1):
        levels[s] = choice[s, g]
        g -= levels[s]
    return levels, total
