"""Bandit learners.

* ``SlotUCB`` / ``SlotTS``: the publisher's estimates of our ad's click
  probability in each slot. Plugged into the matching they give a
  combinatorial bandit (CUCB / combinatorial Thompson sampling): the super-arm
  is the whole ad-to-slot assignment and only our ad's slot gives feedback.
* ``GPTS``: Gaussian-process Thompson sampling over a 1-D grid of arms, used by
  the advertiser to learn clicks as a function of the daily budget.
"""

from __future__ import annotations

import numpy as np


class SlotLearner:
    """Click/impression counts of our ad per slot, with a per-day index."""

    def __init__(self, n_slots: int):
        self.impressions = np.zeros(n_slots)
        self.clicks = np.zeros(n_slots)

    def update(self, impressions: np.ndarray, clicks: np.ndarray) -> None:
        self.impressions += impressions
        self.clicks += clicks

    @property
    def mean(self) -> np.ndarray:
        return np.divide(self.clicks, self.impressions, out=np.zeros_like(self.clicks), where=self.impressions > 0)

    def index(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


class SlotUCB(SlotLearner):
    """UCB1 upper confidence bound on each slot's click probability."""

    def index(self, rng: np.random.Generator) -> np.ndarray:
        total = max(self.impressions.sum(), 1.0)
        bonus = np.sqrt(2 * np.log(total) / np.maximum(self.impressions, 1e-9))
        return np.where(self.impressions > 0, np.minimum(self.mean + bonus, 1.0), 1.0)


class SlotTS(SlotLearner):
    """Thompson sample from a Beta(1 + clicks, 1 + misses) posterior per slot."""

    def index(self, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(1 + self.clicks, 1 + self.impressions - self.clicks)


class GPTS:
    """Gaussian-process Thompson sampling on a fixed grid of arms in [0, 1].

    Squared-exponential kernel with fixed hyperparameters. Rewards are divided
    by ``scale`` so the prior (zero mean, unit variance) is meaningful. A
    noiseless pseudo-observation of 0 at arm 0 encodes "no budget, no clicks".

    Observations are kept as a running mean per arm (noise variance shrinks as
    ``noise**2 / count``), so the GP never grows beyond the number of arms.
    """

    def __init__(
        self,
        arms: np.ndarray,
        scale: float,
        length_scale: float = 0.3,
        noise: float = 0.1,
        anchor_zero: bool = True,
    ):
        self.arms = np.asarray(arms, dtype=float)
        self.scale = scale
        self.length_scale = length_scale
        self.noise = noise
        self.anchor_zero = anchor_zero and np.isclose(self.arms[0], 0.0)
        self.sums = np.zeros(len(self.arms))
        self.counts = np.zeros(len(self.arms))
        self._refresh()

    def _kernel(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((a[:, None] - b[None, :]) / self.length_scale) ** 2)

    def _refresh(self) -> None:
        seen = self.counts > 0
        if self.anchor_zero:
            seen[0] = True
        if not seen.any():
            self.mean = np.zeros(len(self.arms))
            self.std = np.ones(len(self.arms))
            return
        x = self.arms[seen]
        y = np.divide(self.sums, self.counts, out=np.zeros_like(self.sums), where=self.counts > 0)[seen]
        noise_var = np.where(self.counts > 0, self.noise**2 / np.maximum(self.counts, 1), 1e-12)[seen]
        if self.anchor_zero:
            y[0], noise_var[0] = 0.0, 1e-12
        k_xx = self._kernel(x, x) + np.diag(noise_var)
        k_ax = self._kernel(self.arms, x)
        chol = np.linalg.cholesky(k_xx + 1e-9 * np.eye(len(x)))
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y))
        self.mean = k_ax @ alpha
        v = np.linalg.solve(chol, k_ax.T)
        self.std = np.sqrt(np.maximum(1.0 - np.sum(v**2, axis=0), 1e-12))

    def update(self, arm_idx: int | np.ndarray, reward: float | np.ndarray) -> None:
        """Add observations (arm indices, rewards) and refit."""
        np.add.at(self.sums, arm_idx, np.asarray(reward, dtype=float) / self.scale)
        np.add.at(self.counts, arm_idx, 1)
        self._refresh()

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """One Thompson sample of the (non-negative) reward at every arm."""
        return np.maximum(rng.normal(self.mean, self.std), 0.0) * self.scale

    @property
    def posterior_mean(self) -> np.ndarray:
        return np.maximum(self.mean, 0.0) * self.scale
