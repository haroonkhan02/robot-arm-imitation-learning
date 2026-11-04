import numpy as np
import pytest

from robot_arm_il.config import ACTION_DIM, STATE_DIM
from robot_arm_il.env import PickPlaceEnv
from robot_arm_il.expert import run_episode


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


def test_expert_solves_the_task_on_several_seeds(env):
    results = [run_episode(env, seed)[2] for seed in range(5)]
    assert all(results)


def test_recorded_shapes_and_gripper_range(env):
    obs, act, ok = run_episode(env, 3)
    assert ok
    assert obs.shape == (len(act), STATE_DIM) and act.shape[1] == ACTION_DIM
    assert set(np.unique(act[:, -1])) <= {0.0, 1.0}
    assert act[0, -1] == 1.0 and act[-1, -1] == 1.0  # opens before grasp and after release


def test_executed_noise_leaves_recorded_actions_close_to_clean_ones(env):
    _, clean, _ = run_episode(env, 4)
    _, noisy, _ = run_episode(env, 4, exec_noise=0.02)
    n = min(len(clean), len(noisy))
    assert np.abs(clean[:n, :5] - noisy[:n, :5]).max() < 0.5
    assert not np.allclose(clean[:n], noisy[:n])


def test_collecting_frames_matches_episode_length(env, can_render):
    frames: list = []
    obs, _, _ = run_episode(env, 2, frames=frames)
    assert len(frames) == len(obs)
