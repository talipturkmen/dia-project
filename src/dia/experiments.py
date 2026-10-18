"""Experiments for points 3-7 of the assignment.

Each ``run_*`` function simulates one seed and returns plain arrays; the CLI
runs several seeds in parallel and plots mean curves with 95% confidence bands.

Conventions
-----------
* Row 0 of every weight matrix is our ad; rows 1.. are the stochastic advertisers.
* Learners update once per day (batched feedback); every auction of a day uses
  the same estimates, while competitor bids are redrawn per auction.
* Regret is the expected welfare (or value) lost against a clairvoyant, computed
  from true click probabilities so it does not carry click-sampling noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .context import Context, WelfareOracle, generate_contexts
from .knapsack import allocate_budget
from .learners import GPTS, SlotTS, SlotUCB
from .matching import UNASSIGNED, batch_assign, batch_slot_of, batch_vcg_first_row, batch_welfare
from .scenario import CLASS_OF_COMBO, FEATURE_COMBOS, N_SLOTS, Publisher, Scenario, default_scenario

LEARNERS = {"UCB1": SlotUCB, "TS": SlotTS}


# --- shared helpers -----------------------------------------------------------


def _day_auctions(pub: Publisher, rng: np.random.Generator):
    """One day of users for a publisher: feature combo per auction, competitor weights."""
    users = pub.sample_users(rng)
    combos = np.repeat(np.arange(len(users)), users)
    rng.shuffle(combos)
    bids = pub.sample_competitor_bids(rng, size=len(combos))
    competitor_weights = pub.q_competitors[None, :, :] * bids[:, :, None]  # (N, comps, slots)
    return users, combos, competitor_weights


def _with_our_row(our_row: np.ndarray, competitor_weights: np.ndarray) -> np.ndarray:
    return np.concatenate([our_row[:, None, :], competitor_weights], axis=1)


def _learner_index(cls, impressions, clicks, rng):
    learner = cls(N_SLOTS)
    learner.update(impressions, clicks)
    return learner.index(rng)


# --- points 3 and 4: the publisher learns our ad's click probabilities -------


@dataclass
class PublisherResult:
    regret: np.ndarray  # (days,) expected welfare regret summed over publishers
    n_contexts: list[list[int]] = field(default_factory=list)  # per week, per publisher
    final_contexts: list[list[str]] = field(default_factory=list)
    final_estimates: np.ndarray | None = None  # (publishers, slots) aggregate estimate


def run_publisher_learning(
    seed: int,
    days: int = 365,
    learner: str = "UCB1",
    contextual: bool = False,
    per_class_benchmark: bool = False,
    scenario: Scenario | None = None,
    week: int = 7,
) -> PublisherResult:
    """Point 3 (``contextual=False``) and point 4 (``contextual=True``).

    With ``per_class_benchmark`` the clairvoyant knows every user's class, which
    is the right benchmark for context generation. Otherwise the clairvoyant
    knows the click probabilities averaged over the user mix (point 3).
    """
    scenario = scenario or default_scenario()
    rng = np.random.default_rng(seed)
    cls = LEARNERS[learner]
    n_combos = len(FEATURE_COMBOS)

    state = []
    for pub in scenario.publishers:
        oracle = None
        if contextual:
            oracle = WelfareOracle(pub.q_competitors, pub.sample_competitor_bids(rng, size=150), pub.value_per_click)
        state.append(
            {
                "impressions": np.zeros((n_combos, N_SLOTS)),
                "clicks": np.zeros((n_combos, N_SLOTS)),
                "users": np.zeros(n_combos),
                "contexts": [Context(tuple(range(n_combos)))],
                "oracle": oracle,
            }
        )

    regret = np.zeros(days)
    n_contexts: list[list[int]] = []
    for day in range(days):
        for pub, st in zip(scenario.publishers, state, strict=True):
            users, combos, comp_w = _day_auctions(pub, rng)
            st["users"] += users

            # today's estimate of our click probabilities for each feature combination
            estimate_of_combo = np.empty((n_combos, N_SLOTS))
            for ctx in st["contexts"]:
                idx = list(ctx.combos)
                estimate_of_combo[idx] = _learner_index(
                    cls, st["impressions"][idx].sum(0), st["clicks"][idx].sum(0), rng
                )

            v = pub.value_per_click
            chosen = batch_assign(_with_our_row(estimate_of_combo[combos] * v, comp_w))

            true_q_of_user = pub.q_ours[np.array(CLASS_OF_COMBO)[combos]]
            eval_q = (
                true_q_of_user if per_class_benchmark else np.broadcast_to(pub.q_ours_aggregate, true_q_of_user.shape)
            )
            eval_w = _with_our_row(eval_q * v, comp_w)
            best = batch_welfare(eval_w, batch_assign(eval_w))
            regret[day] += float((best - batch_welfare(eval_w, chosen)).sum())

            # feedback: clicks on our ad, sampled from each user's true class
            slot = batch_slot_of(0, chosen, 1 + comp_w.shape[1], N_SLOTS)
            shown = slot != UNASSIGNED
            clicked = rng.random(len(combos)) < true_q_of_user[np.arange(len(combos)), np.maximum(slot, 0)]
            np.add.at(st["impressions"], (combos[shown], slot[shown]), 1)
            np.add.at(st["clicks"], (combos[shown], slot[shown]), clicked[shown])

        if contextual and (day + 1) % week == 0:
            for st in state:
                st["contexts"] = generate_contexts(st["impressions"], st["clicks"], st["users"], st["oracle"])
            n_contexts.append([len(st["contexts"]) for st in state])

    estimates = np.array(
        [
            np.divide(
                st["clicks"].sum(0), st["impressions"].sum(0), out=np.zeros(N_SLOTS), where=st["impressions"].sum(0) > 0
            )
            for st in state
        ]
    )
    return PublisherResult(
        regret=regret,
        n_contexts=n_contexts,
        final_contexts=[[c.label() for c in st["contexts"]] for st in state],
        final_estimates=estimates,
    )


# --- point 5: VCG payments ----------------------------------------------------


def vcg_curves(bids: np.ndarray, auctions: int = 20000, seed: int = 0, scenario: Scenario | None = None):
    """Expected clicks per auction and pay-per-click of our ad as a function of its bid.

    The publisher is clairvoyant and uses our ad's click probabilities averaged
    over the user mix; clicks happen with each user's true class probability.
    """
    scenario = scenario or default_scenario()
    rng = np.random.default_rng(seed)
    clicks = np.zeros((scenario.n_publishers, len(bids)))
    ppc = np.zeros_like(clicks)
    for p, pub in enumerate(scenario.publishers):
        combos = rng.choice(len(FEATURE_COMBOS), size=auctions, p=pub.users_mean / pub.users_mean.sum())
        comp_w = pub.q_competitors[None] * pub.sample_competitor_bids(rng, size=auctions)[:, :, None]
        true_q = pub.q_ours[np.array(CLASS_OF_COMBO)[combos]]
        for k, bid in enumerate(bids):
            our_row = np.broadcast_to(pub.q_ours_aggregate * bid, (auctions, N_SLOTS))
            w = _with_our_row(our_row, comp_w)
            chosen = batch_assign(w)
            slot = batch_slot_of(0, chosen, w.shape[1], N_SLOTS)
            shown = slot != UNASSIGNED
            expected_clicks = np.where(shown, true_q[np.arange(auctions), np.maximum(slot, 0)], 0.0)
            payment = batch_vcg_first_row(w, chosen)
            clicks[p, k] = expected_clicks.mean()
            # charged per click: expected payment divided by the click probability the publisher uses
            per_click = np.where(shown, payment / pub.q_ours_aggregate[np.maximum(slot, 0)], 0.0)
            ppc[p, k] = (per_click * expected_clicks).sum() / max(expected_clicks.sum(), 1e-12)
    return clicks, ppc


# --- points 6 and 7: the advertiser learns bids and budgets -------------------


def simulate_campaign_day(
    pub: Publisher,
    bid: float,
    rng: np.random.Generator,
    publisher_q: np.ndarray | None = None,
):
    """One day of a subcampaign with an unlimited budget.

    Returns the cumulative spend after each of our clicks (so the clicks bought
    with budget ``d`` are ``(cum_spend <= d).sum()``), plus per-auction data the
    caller needs for publisher regret. ``publisher_q`` is the publisher's
    estimate of our click probabilities (clairvoyant when ``None``).
    """
    _, combos, comp_w = _day_auctions(pub, rng)
    q_used = pub.q_ours_aggregate if publisher_q is None else publisher_q
    w = _with_our_row(np.broadcast_to(q_used * bid, (len(combos), N_SLOTS)), comp_w)
    chosen = batch_assign(w)
    slot = batch_slot_of(0, chosen, w.shape[1], N_SLOTS)
    shown = slot != UNASSIGNED
    true_q = pub.q_ours[np.array(CLASS_OF_COMBO)[combos]]
    clicked = shown & (rng.random(len(combos)) < true_q[np.arange(len(combos)), np.maximum(slot, 0)])
    payment = batch_vcg_first_row(w, chosen)
    price = np.where(shown, payment / np.maximum(q_used[np.maximum(slot, 0)], 1e-9), 0.0)
    cum_spend = np.cumsum(price[clicked])
    return cum_spend, {"combos": combos, "comp_w": comp_w, "chosen": chosen, "slot": slot, "clicked": clicked}


@dataclass
class CampaignSetup:
    bids: np.ndarray
    budget_levels: np.ndarray  # daily budget grid for one subcampaign
    total_budget: float
    expected_clicks: np.ndarray  # (publishers, bids, levels) with a clairvoyant publisher
    optimum: float  # clairvoyant daily value
    optimal_levels: np.ndarray
    optimal_bids: np.ndarray


def campaign_setup(
    scenario: Scenario | None = None,
    bids: np.ndarray | None = None,
    n_levels: int = 21,
    budget_share: float = 0.45,
    days: int = 200,
    seed: int = 12345,
) -> CampaignSetup:
    """Clairvoyant tables for points 6-7, estimated by Monte Carlo.

    The total daily budget is ``budget_share`` of what the advertiser would
    spend at the highest bid with no budget limit, so the constraint binds.
    """
    scenario = scenario or default_scenario()
    bids = np.array([0.4, 0.6, 0.8, 1.0, 1.2, 1.4]) if bids is None else bids
    rng = np.random.default_rng(seed)

    spends = {}
    max_spend = 0.0
    for p, pub in enumerate(scenario.publishers):
        for k, bid in enumerate(bids):
            runs = [simulate_campaign_day(pub, bid, rng)[0] for _ in range(days)]
            spends[p, k] = runs
            if k == len(bids) - 1:
                max_spend += np.mean([r[-1] if len(r) else 0.0 for r in runs])

    total_budget = round(budget_share * max_spend)
    levels = np.linspace(0, total_budget, n_levels)
    expected = np.zeros((scenario.n_publishers, len(bids), n_levels))
    for (p, k), runs in spends.items():
        for g, d in enumerate(levels):
            expected[p, k, g] = np.mean([(r <= d + 1e-12).sum() for r in runs])

    values = np.array([pub.value_per_click for pub in scenario.publishers])
    best_bid = expected.argmax(axis=1)  # (publishers, levels)
    value_table = values[:, None] * expected.max(axis=1)
    opt_levels, optimum = allocate_budget(value_table)
    opt_bids = bids[best_bid[np.arange(len(opt_levels)), opt_levels]]
    return CampaignSetup(bids, levels, float(total_budget), expected, optimum, opt_levels, opt_bids)


@dataclass
class AdvertiserResult:
    regret: np.ndarray  # (days,) advertiser value regret
    budget_share: np.ndarray  # (days, publishers) chosen budget levels
    chosen_bids: np.ndarray  # (days, publishers)
    publisher_regret: np.ndarray | None = None  # (days,) for point 7


def run_advertiser_learning(
    seed: int,
    setup: CampaignSetup,
    days: int = 180,
    publisher_learns: bool = False,
    publisher_learner: str = "TS",
    scenario: Scenario | None = None,
) -> AdvertiserResult:
    """Point 6 (clairvoyant publisher) and point 7 (``publisher_learns=True``).

    For each subcampaign and bid the advertiser keeps a GP over the budget grid
    (clicks as a function of budget). Every day it Thompson-samples every GP,
    takes the best bid at each budget level, and splits the total budget with
    the multiple-choice knapsack.

    In point 7 the publisher simultaneously learns our click probabilities
    (aggregated, as in point 3; Thompson sampling by default) and its welfare
    regret is tracked too.
    """
    scenario = scenario or default_scenario()
    rng = np.random.default_rng(seed)
    n_pub = scenario.n_publishers
    arms = setup.budget_levels / setup.total_budget  # GP input in [0, 1]
    values = np.array([pub.value_per_click for pub in scenario.publishers])
    click_scale = setup.expected_clicks.max()
    gps = [[GPTS(arms, scale=click_scale) for _ in setup.bids] for _ in range(n_pub)]
    publisher = [LEARNERS[publisher_learner](N_SLOTS) for _ in range(n_pub)]

    regret = np.zeros(days)
    pub_regret = np.zeros(days)
    levels_log = np.zeros((days, n_pub), dtype=int)
    bids_log = np.zeros((days, n_pub))

    for day in range(days):
        samples = np.array([[gp.sample(rng) for gp in row] for row in gps])  # (pub, bids, levels)
        best_bid = samples.argmax(axis=1)
        levels, _ = allocate_budget(values[:, None] * samples.max(axis=1))

        realised = 0.0
        for p, pub in enumerate(scenario.publishers):
            k = best_bid[p, levels[p]]
            budget = setup.budget_levels[levels[p]]
            q_est = publisher[p].index(rng) if publisher_learns else None
            cum_spend, info = simulate_campaign_day(pub, setup.bids[k], rng, publisher_q=q_est)
            n_clicks = int((cum_spend <= budget + 1e-12).sum())
            # The spend timeline also reveals the clicks any smaller budget would
            # have bought, so every level up to today's is observed.
            observed = np.arange(levels[p] + 1)
            gps[p][k].update(observed, [(cum_spend <= setup.budget_levels[g] + 1e-12).sum() for g in observed])
            levels_log[day, p] = levels[p]
            bids_log[day, p] = setup.bids[k]

            if publisher_learns:
                realised += values[p] * n_clicks
                # our ad leaves the auctions once its budget runs out
                n = len(info["combos"])
                in_budget = np.ones(n, dtype=bool)
                if n_clicks < len(cum_spend):
                    last_click_auction = np.flatnonzero(info["clicked"])[n_clicks]
                    in_budget[last_click_auction:] = False
                slot, clicked = info["slot"], info["clicked"]
                shown = (slot != UNASSIGNED) & in_budget
                publisher[p].update(
                    np.bincount(slot[shown], minlength=N_SLOTS).astype(float),
                    np.bincount(slot[shown & clicked], minlength=N_SLOTS).astype(float),
                )
                pub_regret[day] += _publisher_regret(pub, setup.bids[k], info, in_budget)
            else:
                realised += values[p] * setup.expected_clicks[p, k, levels[p]]

        regret[day] = setup.optimum - realised

    return AdvertiserResult(
        regret=regret,
        budget_share=levels_log,
        chosen_bids=bids_log,
        publisher_regret=pub_regret if publisher_learns else None,
    )


def _publisher_regret(pub: Publisher, bid: float, info: dict, in_budget: np.ndarray) -> float:
    """Expected welfare the publisher lost today versus knowing our click probabilities."""
    comp_w = info["comp_w"]
    q_true = np.where(in_budget[:, None], pub.q_ours_aggregate[None, :] * bid, 0.0)
    w = _with_our_row(q_true, comp_w)
    best = batch_welfare(w, batch_assign(w))
    got = batch_welfare(w, info["chosen"])
    # once out of budget our ad is not in the auction; the publisher's choice for
    # those auctions is re-solved without it, so it loses nothing there
    return float((best - got)[in_budget].sum())
