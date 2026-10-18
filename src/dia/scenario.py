"""The advertising scenario: publishers, slots, user classes, and stochastic advertisers.

Our advertiser sponsors one product with one subcampaign per publisher (website).
Each publisher shows ``N_SLOTS`` ads per page view and runs one auction per
interested user. Users carry two binary features; the four feature
combinations map to three user classes with different click behaviour.

Click probabilities ``q[i, j]`` (ad ``i`` shown in slot ``j``) are deliberately
*not* factorised into an ad term times a slot term: every ad has its own
slot-decay profile, and our ad's profile also depends on the user class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_SLOTS = 4
N_COMPETITORS = 6

FEATURES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("age", ("young", "older")),
    ("device", ("mobile", "desktop")),
)

# Feature combinations, indexed 0..3, as (age, device) bits.
FEATURE_COMBOS: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

# Hidden user class of each feature combination.
CLASS_OF_COMBO: tuple[int, ...] = (0, 0, 1, 2)
CLASS_NAMES: tuple[str, ...] = ("young", "older · mobile", "older · desktop")
N_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True)
class Publisher:
    """One website, and therefore one subcampaign of our advertiser."""

    name: str
    users_mean: np.ndarray  # (4,) mean daily interested users per feature combination
    q_ours: np.ndarray  # (N_CLASSES, N_SLOTS) our ad's click probability per class and slot
    q_competitors: np.ndarray  # (N_COMPETITORS, N_SLOTS) known to everyone
    bid_mean: np.ndarray  # (N_COMPETITORS,) stochastic advertisers' bid distribution
    bid_std: np.ndarray  # (N_COMPETITORS,)
    value_per_click: float  # our advertiser's value of a click (truthful bid)

    @property
    def class_weights(self) -> np.ndarray:
        """Share of each user class among this publisher's daily users."""
        per_class = np.zeros(N_CLASSES)
        for combo, mean in enumerate(self.users_mean):
            per_class[CLASS_OF_COMBO[combo]] += mean
        return per_class / per_class.sum()

    @property
    def q_ours_aggregate(self) -> np.ndarray:
        """Our ad's click probability per slot, averaged over the user mix."""
        return self.class_weights @ self.q_ours

    def q_ours_for_combos(self, combos: list[int] | tuple[int, ...]) -> np.ndarray:
        """Our ad's click probability per slot for users from a set of feature combinations."""
        weights = np.array([self.users_mean[c] for c in combos], dtype=float)
        rows = np.array([self.q_ours[CLASS_OF_COMBO[c]] for c in combos])
        return (weights / weights.sum()) @ rows

    def sample_users(self, rng: np.random.Generator) -> np.ndarray:
        """Daily interested users per feature combination."""
        return rng.poisson(self.users_mean)

    def sample_competitor_bids(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        """Bids of the stochastic advertisers for one auction (or ``size`` auctions)."""
        shape = (N_COMPETITORS,) if size is None else (size, N_COMPETITORS)
        return np.maximum(0.01, rng.normal(self.bid_mean, self.bid_std, size=shape))


@dataclass(frozen=True)
class Scenario:
    publishers: tuple[Publisher, ...] = field(default_factory=tuple)

    @property
    def n_publishers(self) -> int:
        return len(self.publishers)


def _decay(top: float, ratios: list[float]) -> np.ndarray:
    """Click probabilities for slots 0..N_SLOTS-1 from a top-slot value and per-slot ratios."""
    return top * np.array(ratios)


def default_scenario(seed: int = 2020) -> Scenario:
    """The scenario used for every experiment in this repository.

    Our ad is strongest with young users and weakest with older desktop users,
    and the size of that gap varies by publisher. Competitors are drawn once
    from a fixed seed so the scenario is identical across runs.
    """
    rng = np.random.default_rng(seed)
    specs = [
        # name, users per (young·mobile, young·desktop, older·mobile, older·desktop), affinity, value per click
        ("News", [55, 35, 30, 50], [1.00, 0.55, 0.25], 1.0),
        ("Tech blog", [70, 45, 15, 25], [1.15, 0.70, 0.30], 1.1),
        ("Sports", [45, 25, 40, 40], [0.90, 0.80, 0.35], 0.9),
        ("Lifestyle", [60, 20, 45, 30], [1.05, 0.45, 0.60], 1.0),
    ]
    # Slot-decay profile of our ad for each class: young users rarely scroll,
    # older desktop users read the whole page.
    class_profiles = [
        [1.00, 0.63, 0.50, 0.44],
        [1.00, 0.85, 0.70, 0.55],
        [1.00, 0.97, 0.93, 0.90],
    ]
    base_top = 0.158

    publishers = []
    for name, users, affinity, value in specs:
        q_ours = np.array([_decay(base_top * a, profile) for a, profile in zip(affinity, class_profiles, strict=True)])
        tops = rng.uniform(0.052, 0.136, size=N_COMPETITORS)
        q_comp = np.empty((N_COMPETITORS, N_SLOTS))
        for i, top in enumerate(tops):
            ratios = np.sort(rng.uniform(0.277, 0.753, size=N_SLOTS - 1))[::-1]
            q_comp[i] = top * np.concatenate(([1.0], ratios))
        publishers.append(
            Publisher(
                name=name,
                users_mean=np.array(users, dtype=float),
                q_ours=q_ours,
                q_competitors=q_comp,
                bid_mean=rng.uniform(0.6, 1.4, size=N_COMPETITORS),
                bid_std=np.full(N_COMPETITORS, 0.32),
                value_per_click=value,
            )
        )
    return Scenario(publishers=tuple(publishers))
