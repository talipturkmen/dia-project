"""Run the experiments and write figures plus a JSON summary.

python -m dia run                 # everything, 10 seeds (a few minutes)
python -m dia run --steps 3 5     # selected points of the assignment
python -m dia run --quick         # short smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from . import plotting
from .experiments import (
    campaign_setup,
    run_advertiser_learning,
    run_publisher_learning,
    vcg_curves,
)
from .scenario import CLASS_NAMES, default_scenario


def _parallel(fn, seeds, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, seeds))


def step3(args, summary):
    curves = {}
    for learner in ("UCB1", "TS"):
        runs = _parallel(partial(run_publisher_learning, days=args.days, learner=learner), args.seed_list, args.workers)
        curves[f"{learner}"] = np.array([r.regret for r in runs])
        summary["step3"][learner] = _final(curves[learner])
    plotting.regret_figure(
        curves,
        "Point 3 · Publisher learns our ad's click probabilities",
        "cumulative welfare regret",
        args.figures / "step3_matching_regret.png",
        note=f"Mean and 95% CI over {len(args.seed_list)} seeds. Clairvoyant knows q averaged over the user mix.",
    )


def step4(args, summary):
    no_ctx = _parallel(
        partial(run_publisher_learning, days=args.days, learner="TS", per_class_benchmark=True),
        args.seed_list,
        args.workers,
    )
    ctx = _parallel(
        partial(run_publisher_learning, days=args.days, learner="TS", contextual=True, per_class_benchmark=True),
        args.seed_list,
        args.workers,
    )
    curves = {
        "no contexts": np.array([r.regret for r in no_ctx]),
        "weekly context generation": np.array([r.regret for r in ctx]),
    }
    summary["step4"] = {k: _final(v) for k, v in curves.items()}
    names = [p.name for p in default_scenario().publishers]
    summary["step4"]["final_contexts_seed0"] = dict(zip(names, ctx[0].final_contexts, strict=True))
    plotting.regret_figure(
        curves,
        "Point 4 · Context generation (Thompson sampling)",
        "cumulative welfare regret",
        args.figures / "step4_context_regret.png",
        note=f"Mean and 95% CI over {len(args.seed_list)} seeds. Clairvoyant knows every user's class.",
    )
    plotting.contexts_figure(np.array([r.n_contexts for r in ctx]), names, args.figures / "step4_contexts.png")


def step5(args, summary):
    bids = np.round(np.arange(0.2, 1.81, 0.1), 2)
    clicks, ppc = vcg_curves(bids, auctions=4000 if args.quick else 20000)
    names = [p.name for p in default_scenario().publishers]
    plotting.vcg_figure(bids, clicks, ppc, names, args.figures / "step5_vcg.png")
    summary["step5"] = {
        name: {
            "clicks_per_auction_at_bid_1": float(clicks[p, list(bids).index(1.0)]),
            "price_per_click_at_bid_1": float(ppc[p, list(bids).index(1.0)]),
        }
        for p, name in enumerate(names)
    }


def steps67(args, summary, steps):
    setup = campaign_setup(days=60 if args.quick else 300)
    names = [p.name for p in default_scenario().publishers]
    summary["campaign"] = {
        "total_daily_budget": setup.total_budget,
        "clairvoyant_daily_value": setup.optimum,
        "clairvoyant_budgets": dict(zip(names, setup.budget_levels[setup.optimal_levels].tolist(), strict=True)),
        "clairvoyant_bids": dict(zip(names, setup.optimal_bids.tolist(), strict=True)),
    }
    days = args.days_campaign
    if 6 in steps:
        runs = _parallel(partial(run_advertiser_learning, setup=setup, days=days), args.seed_list, args.workers)
        regret = np.array([r.regret for r in runs])
        summary["step6"] = _final(regret) | {"clairvoyant_total": setup.optimum * days}
        plotting.regret_figure(
            {"GP-TS + knapsack": regret},
            "Point 6 · Advertiser learns bids and budgets (clairvoyant publisher)",
            "cumulative value regret",
            args.figures / "step6_advertiser_regret.png",
            note=f"Mean and 95% CI over {len(args.seed_list)} seeds. "
            f"Clairvoyant earns {setup.optimum:.1f} value/day with budget {setup.total_budget:.0f}/day.",
        )
        plotting.budget_figure(
            np.array([r.budget_share for r in runs]),
            setup.budget_levels,
            setup.optimal_levels,
            names,
            args.figures / "step6_budgets.png",
        )
    if 7 in steps:
        runs = _parallel(
            partial(run_advertiser_learning, setup=setup, days=days, publisher_learns=True),
            args.seed_list,
            args.workers,
        )
        pub = np.array([r.publisher_regret for r in runs])
        adv = np.array([r.regret for r in runs])
        summary["step7"] = {"publisher": _final(pub), "advertiser": _final(adv)}
        plotting.two_regrets_figure(
            ("Publisher (welfare)", pub),
            ("Advertiser (value)", adv),
            "Point 7 · Publisher and advertiser learning at the same time",
            args.figures / "step7_joint_regret.png",
        )


def _final(runs: np.ndarray) -> dict:
    cum = np.cumsum(runs, axis=1)[:, -1]
    tail = runs[:, -max(runs.shape[1] // 10, 1) :].mean(axis=1)
    return {
        "cumulative_regret_mean": float(cum.mean()),
        "cumulative_regret_ci95": float(1.96 * cum.std(ddof=1) / np.sqrt(len(cum))) if len(cum) > 1 else 0.0,
        "daily_regret_last_10pct": float(tail.mean()),
        "daily_regret_first_day": float(runs[:, 0].mean()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="dia", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run experiments and make figures")
    run.add_argument("--steps", type=int, nargs="+", default=[3, 4, 5, 6, 7], choices=[3, 4, 5, 6, 7])
    run.add_argument("--seeds", type=int, default=10)
    run.add_argument("--days", type=int, default=365, help="horizon for points 3-4")
    run.add_argument("--days-campaign", type=int, default=180, help="horizon for points 6-7")
    run.add_argument("--workers", type=int, default=None)
    run.add_argument("--figures", type=Path, default=Path("figures"))
    run.add_argument("--results", type=Path, default=Path("results"))
    run.add_argument("--quick", action="store_true", help="short smoke run (2 seeds, few days)")
    args = parser.parse_args(argv)

    if args.quick:
        args.seeds, args.days, args.days_campaign = 2, 28, 20
    args.seed_list = list(range(args.seeds))
    args.figures.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)

    scenario = default_scenario()
    plotting.scenario_figure(scenario, CLASS_NAMES, args.figures / "scenario.png")
    summary: dict = {"seeds": args.seeds, "days": args.days, "days_campaign": args.days_campaign, "step3": {}}
    for step in sorted(set(args.steps)):
        if step in (6, 7):
            continue
        start = time.time()
        {3: step3, 4: step4, 5: step5}[step](args, summary)
        print(f"point {step} done in {time.time() - start:.1f}s")
    if {6, 7} & set(args.steps):
        start = time.time()
        steps67(args, summary, set(args.steps))
        print(f"points 6-7 done in {time.time() - start:.1f}s")

    out = args.results / "summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out} and figures in {args.figures}/")


if __name__ == "__main__":
    main()
