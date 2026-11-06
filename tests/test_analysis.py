import numpy as np
import pytest

from robot_arm_il.analysis import auc, coverage, outcome_counts, seeds_of, train_layouts, wilson
from robot_arm_il.dataset import DemoSet


def test_wilson_interval_contains_the_rate_and_stays_in_range():
    lo, hi = wilson(300, 300)
    assert hi == 1.0 and 0.98 < lo < 1.0
    lo, hi = wilson(0, 300)
    assert lo == 0.0 and hi < 0.02
    lo, hi = wilson(150, 300)
    assert lo < 0.5 < hi


def test_auc_perfect_inverse_and_tied():
    s = np.array([1.0, 2.0, 3.0, 4.0])
    pos = np.array([False, False, True, True])
    assert auc(s, pos) == 1.0
    assert auc(-s, pos) == 0.0
    assert auc(np.zeros(4), pos) == 0.5
    assert np.isnan(auc(s, np.zeros(4, bool)))


def test_outcome_counts_uses_success_before_failed_at():
    rows = [
        {"success": True, "failed_at": None},
        {"success": False, "failed_at": "lifted"},
        {"success": False, "failed_at": "lifted"},
    ]
    counts = outcome_counts(rows)
    assert counts["success"] == 1 and counts["lifted"] == 2 and counts["reached"] == 0


def test_coverage_finds_failures_far_from_training_layouts():
    layouts = np.array([[0.0, 0.0, 1.0, 1.0]])
    rows = [
        {"cube_xy": [0.01, 0.0], "goal_xy": [1.0, 1.0], "success": True},
        {"cube_xy": [0.5, 0.0], "goal_xy": [1.0, 1.0], "success": False},
    ]
    res = coverage(rows, layouts)
    assert res["auc_cube"] == 1.0 and res["auc_goal"] == 0.5
    assert res["median_cube_cm"]["failure"] == pytest.approx(50.0)


def test_train_layouts_reads_first_frames_of_the_selected_episodes():
    n_ep, length = 20, 3
    obs = np.zeros((n_ep * length, 13), np.float32)
    obs[:, 6] = np.repeat(np.arange(n_ep), length)  # cube x = episode id
    demos = DemoSet(
        obs,
        np.zeros((n_ep * length, 6), np.float32),
        np.arange(1, n_ep + 1) * length,
        np.arange(n_ep),
    )
    assert train_layouts(demos, 0, 0).shape == (16, 4)
    picked = train_layouts(demos, 5, 1)
    assert picked.shape == (5, 4) and len(set(picked[:, 0])) == 5
    assert (picked[:-1, 0] < picked[1:, 0]).all()


def test_seeds_of_collects_only_existing_repeats():
    reports = {"a": {"success_rate": 0.5}, "a_s2": {"success_rate": 0.7}}
    assert seeds_of(reports, "a") == [0.5, 0.7]
    reports["a_s1_newest"] = {"success_rate": 0.1}
    assert seeds_of(reports, "a", "_newest") == [0.1]
