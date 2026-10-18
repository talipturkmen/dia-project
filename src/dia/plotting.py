"""Figures for the README. One style for all charts: thin lines, recessive axes,
direct labels at line ends, mean curves with 95% confidence bands over seeds."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": TEXT_2,
        "lines.linewidth": 2,
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
    }
)


def _mean_ci(runs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    runs = np.asarray(runs, dtype=float)
    mean = runs.mean(axis=0)
    half = 1.96 * runs.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
    return mean, half


def _band(ax, x, runs, color, label, end_label=True):
    mean, half = _mean_ci(runs)
    ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, mean, color=color, label=label, solid_capstyle="round")
    if end_label:
        ax.annotate(
            f"{label}  {mean[-1]:,.0f}",
            xy=(x[-1], mean[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT_2,
        )


def _finish(fig, path: Path, note: str | None = None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=TEXT_2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def scenario_figure(scenario, class_names, path: Path):
    n = scenario.n_publishers
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.6), sharey=True)
    vmax = max(p.q_ours.max() for p in scenario.publishers)
    for ax, pub in zip(axes, scenario.publishers, strict=True):
        ax.grid(False)
        im = ax.imshow(pub.q_ours, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), q in np.ndenumerate(pub.q_ours):
            ax.text(j, i, f"{q:.2f}", ha="center", va="center", fontsize=7, color="white" if q > 0.55 * vmax else TEXT)
        ax.set_title(pub.name)
        ax.set_xticks(range(pub.q_ours.shape[1]), [f"slot {j + 1}" for j in range(pub.q_ours.shape[1])])
        ax.set_yticks(range(len(class_names)), class_names)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=7, colors=TEXT_2)
    cbar.outline.set_visible(False)
    cbar.set_label("click probability of our ad", fontsize=8, color=TEXT_2)
    fig.suptitle(
        "Our ad's true click probability by user class and slot",
        x=0.01,
        y=1.08,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def regret_figure(curves: dict[str, np.ndarray], title: str, ylabel: str, path: Path, note: str | None = None):
    """Cumulative regret curves (each value: runs x days of per-day regret)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for color, (label, runs) in zip(SERIES, curves.items(), strict=False):
        cum = np.cumsum(runs, axis=1)
        _band(ax, np.arange(1, cum.shape[1] + 1), cum, color, label)
    ax.set_title(title)
    ax.set_xlabel("day")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.margins(x=0.18)
    _finish(fig, path, note)


def contexts_figure(n_contexts: np.ndarray, publisher_names, path: Path):
    """n_contexts: runs x weeks x publishers."""
    fig, ax = plt.subplots(figsize=(7, 3.0))
    weeks = np.arange(1, n_contexts.shape[1] + 1)
    for p, (color, name) in enumerate(zip(SERIES, publisher_names, strict=False)):
        mean = n_contexts[:, :, p].mean(axis=0)
        ax.plot(weeks, mean, color=color, label=name)
    ax.set_title("Contexts in use (mean over seeds)")
    ax.set_xlabel("week")
    ax.set_ylabel("number of contexts")
    ax.set_ylim(0.8, 4.2)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_xlim(1, weeks[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=4)
    _finish(fig, path)


def vcg_figure(bids, clicks, ppc, publisher_names, path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for p, (color, name) in enumerate(zip(SERIES, publisher_names, strict=False)):
        ax1.plot(bids, clicks[p], color=color, label=name, marker="o", markersize=4)
        shown = clicks[p] > 0
        ax2.plot(bids[shown], ppc[p][shown], color=color, label=name, marker="o", markersize=4)
    ax2.plot(bids, bids, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)), label="bid (upper bound)")
    ax1.set_title("Expected clicks per auction")
    ax2.set_title("VCG price per click")
    for ax in (ax1, ax2):
        ax.set_xlabel("our bid")
        ax.set_ylim(bottom=0)
    ax1.set_ylabel("clicks / auction")
    ax2.set_ylabel("price per click")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")
    _finish(fig, path)


def budget_figure(levels: np.ndarray, budget_grid, optimal_levels, publisher_names, path: Path):
    """levels: runs x days x publishers (chosen budget level indices)."""
    fig, ax = plt.subplots(figsize=(7, 3.4))
    budget = np.asarray(budget_grid)[levels]  # runs x days x publishers
    days = np.arange(1, budget.shape[1] + 1)
    window = 7
    kernel = np.ones(window) / window
    for p, (color, name) in enumerate(zip(SERIES, publisher_names, strict=False)):
        mean = budget[:, :, p].mean(axis=0)
        smooth = np.convolve(mean, kernel, mode="valid")
        ax.plot(days[window - 1 :], smooth, color=color, label=name)
        opt = budget_grid[optimal_levels[p]]
        ax.axhline(opt, color=color, linewidth=1, linestyle=(0, (3, 3)), alpha=0.9)
        ax.annotate(
            f"{name}  (optimum {opt:.1f})",
            xy=(days[-1], smooth[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=TEXT_2,
        )
    ax.set_title("Daily budget per subcampaign (7-day mean over seeds; dashed: clairvoyant)")
    ax.set_xlabel("day")
    ax.set_ylabel("budget")
    ax.set_ylim(bottom=0)
    ax.set_xlim(1, days[-1] * 1.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=4)
    _finish(fig, path)


def two_regrets_figure(left: tuple[str, np.ndarray], right: tuple[str, np.ndarray], title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, color, (label, runs) in zip(axes, SERIES[:2], (left, right), strict=True):
        cum = np.cumsum(runs, axis=1)
        _band(ax, np.arange(1, cum.shape[1] + 1), cum, color, label, end_label=False)
        mean = cum.mean(axis=0)
        ax.set_title(f"{label}: {mean[-1]:,.0f} after {cum.shape[1]} days")
        ax.set_xlabel("day")
        ax.set_ylabel("cumulative regret")
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_xlim(left=0)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=TEXT)
    _finish(fig, path)
