import json

import numpy as np

from dia.cli import main
from dia.context import WelfareOracle, generate_contexts
from dia.experiments import campaign_setup, run_advertiser_learning, run_publisher_learning
from dia.scenario import CLASS_OF_COMBO, N_SLOTS, default_scenario


def _oracle():
    pub = default_scenario().publishers[0]
    rng = np.random.default_rng(0)
    return WelfareOracle(pub.q_competitors, pub.sample_competitor_bids(rng, size=100), 1.0)


def _data(q_by_class, n=20000):
    impressions = np.full((4, N_SLOTS), float(n))
    clicks = np.array([q_by_class[CLASS_OF_COMBO[c]] for c in range(4)]) * n
    return impressions, clicks, np.full(4, float(n))


def test_no_split_when_classes_behave_the_same():
    q = np.array([[0.1, 0.08, 0.06, 0.05]] * 3)
    contexts = generate_contexts(*_data(q), _oracle())
    assert [c.label() for c in contexts] == ["all users"]


def test_splits_on_age_when_age_matters():
    q = np.array([[0.30, 0.10, 0.05, 0.02], [0.02, 0.02, 0.02, 0.02], [0.02, 0.02, 0.02, 0.02]])
    labels = [c.label() for c in generate_contexts(*_data(q), _oracle())]
    assert "young" in labels


def test_publisher_learning_regret_is_non_negative_and_shrinks():
    result = run_publisher_learning(seed=0, days=60, learner="TS")
    assert np.all(result.regret >= -1e-9)
    assert result.regret[-10:].mean() < result.regret[:10].mean()


def test_context_generation_runs():
    result = run_publisher_learning(seed=1, days=21, learner="TS", contextual=True, per_class_benchmark=True)
    assert len(result.n_contexts) == 3
    assert all(1 <= n <= 4 for week in result.n_contexts for n in week)


def test_campaign_setup_and_advertiser_learning():
    setup = campaign_setup(days=20)
    assert setup.optimal_levels.sum() <= len(setup.budget_levels) - 1
    assert setup.optimum > 0
    result = run_advertiser_learning(seed=0, setup=setup, days=10, publisher_learns=True)
    assert result.regret.shape == (10,)
    assert result.publisher_regret is not None and np.all(result.publisher_regret >= -1e-9)


def test_cli_quick_run(tmp_path):
    main(["run", "--quick", "--workers", "1", "--figures", str(tmp_path / "fig"), "--results", str(tmp_path / "res")])
    summary = json.loads((tmp_path / "res" / "summary.json").read_text())
    assert {"step3", "step4", "step5", "step6", "step7"} <= summary.keys()
    assert len(list((tmp_path / "fig").glob("*.png"))) == 8
