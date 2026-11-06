import numpy as np
import pytest

pytest.importorskip("torch")

from robot_arm_il.dataset import DemoSet  # noqa: E402
from robot_arm_il.stall import copy_baseline, is_frozen  # noqa: E402


def test_is_frozen_looks_only_at_the_last_window():
    moving = np.linspace(0, 1, 50)[:, None] * np.ones(5)
    assert not is_frozen(moving)
    settled = np.concatenate([moving, np.ones((20, 5))])
    assert is_frozen(settled)
    assert not is_frozen(np.ones((5, 5)))


def test_copy_baseline_grows_with_the_horizon_on_a_ramp():
    t = np.arange(10, dtype=np.float32)[:, None]
    obs = np.zeros((10, 13), np.float32)
    obs[:, :5] = t
    act = np.zeros((10, 6), np.float32)
    act[:, :5] = t  # action equals the current joints
    demos = DemoSet(obs, act, np.array([10]), np.array([0]))
    res = copy_baseline(demos, (0, 2), np.ones(5, np.float32))
    assert res["0"] == 0.0 and res["2"] == pytest.approx(2.0)
