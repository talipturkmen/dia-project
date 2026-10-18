"""Context generation for the publisher's matching problem.

Every week the publisher splits its users on the two binary features with a
greedy tree. A split of context ``X`` into ``X1``/``X2`` is accepted when the
lower bound of the value after the split beats the lower bound before it:

    p̲(X1) * μ̲(X1) + p̲(X2) * μ̲(X2) >= μ̲(X)

``p̲`` is a Hoeffding lower bound on the probability of the child context
(given the parent), and ``μ̲`` is the welfare our ad adds to the publisher's
matching when its click probabilities are set to their lower confidence bounds
(estimated by Monte Carlo over the competitors' bid distribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .matching import assign, welfare
from .scenario import FEATURE_COMBOS, FEATURES


def hoeffding_lower(p: float, n: float, delta: float) -> float:
    """Hoeffding lower bound on a probability estimated from ``n`` samples."""
    return max(p - np.sqrt(-np.log(delta) / (2 * max(n, 1))), 0.0)


def click_lower(mean: np.ndarray, n: np.ndarray, z: float) -> np.ndarray:
    """Normal-approximation lower bound on click probabilities.

    Much tighter than Hoeffding for the small probabilities of ad clicks.
    """
    std = np.sqrt(mean * (1 - mean) / np.maximum(n, 1))
    return np.where(n > 0, np.maximum(mean - z * std, 0.0), 0.0)


class WelfareOracle:
    """Expected welfare that our ad adds to the matching, for a vector of click probabilities."""

    def __init__(self, q_competitors: np.ndarray, bid_samples: np.ndarray, our_bid: float):
        self.q_competitors = q_competitors
        self.bid_samples = bid_samples  # (M, n_competitors)
        self.our_bid = our_bid
        self.baseline = np.mean([self._best(None, b) for b in bid_samples])

    def _best(self, q_ours: np.ndarray | None, bids: np.ndarray) -> float:
        weights = self.q_competitors * bids[:, None]
        if q_ours is not None:
            weights = np.vstack([q_ours * self.our_bid, weights])
        return welfare(weights, assign(weights))

    def added_value(self, q_ours: np.ndarray) -> float:
        total = np.mean([self._best(q_ours, b) for b in self.bid_samples])
        return max(total - self.baseline, 0.0)


@dataclass(frozen=True)
class Context:
    combos: tuple[int, ...]

    def label(self) -> str:
        """Readable description, e.g. 'older · desktop' or 'all users'."""
        if len(self.combos) == len(FEATURE_COMBOS):
            return "all users"
        parts = []
        for f, (_, names) in enumerate(FEATURES):
            values = {FEATURE_COMBOS[c][f] for c in self.combos}
            if len(values) == 1:
                parts.append(names[values.pop()])
        return " · ".join(parts)


def generate_contexts(
    impressions: np.ndarray,
    clicks: np.ndarray,
    users: np.ndarray,
    oracle: WelfareOracle,
    delta: float = 0.05,
    z: float = 1.645,
) -> list[Context]:
    """Greedy context tree from per-feature-combination data.

    ``impressions[c, j]`` and ``clicks[c, j]`` are our ad's counts in slot ``j``
    for users with feature combination ``c``; ``users[c]`` counts all users seen.
    """

    def value(combos: tuple[int, ...]) -> float:
        idx = list(combos)
        n = impressions[idx].sum(axis=0)
        k = clicks[idx].sum(axis=0)
        mean = np.divide(k, n, out=np.zeros_like(k, dtype=float), where=n > 0)
        return oracle.added_value(click_lower(mean, n, z))

    def split(combos: tuple[int, ...], free_features: tuple[int, ...]) -> list[Context]:
        parent_value = value(combos)
        n_parent = users[list(combos)].sum()
        best_gain, best_children = 0.0, None
        for f in free_features:
            children = [tuple(c for c in combos if FEATURE_COMBOS[c][f] == bit) for bit in (0, 1)]
            if any(len(child) == 0 for child in children):
                continue
            split_value = 0.0
            for child in children:
                p = users[list(child)].sum() / max(n_parent, 1)
                split_value += hoeffding_lower(p, n_parent, delta) * value(child)
            gain = split_value - parent_value
            if gain > best_gain:
                best_gain, best_children = gain, (f, children)
        if best_children is None:
            return [Context(combos)]
        f, children = best_children
        remaining = tuple(x for x in free_features if x != f)
        return [ctx for child in children for ctx in split(child, remaining)]

    return split(tuple(range(len(FEATURE_COMBOS))), tuple(range(len(FEATURES))))
