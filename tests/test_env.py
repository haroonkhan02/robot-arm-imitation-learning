import numpy as np
import pytest

from robot_arm_il.config import ACTION_DIM, STATE_DIM, EnvConfig
from robot_arm_il.env import PickPlaceEnv, quat_yaw


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


def test_reset_is_deterministic_per_seed(env):
    a = env.reset(seed=7)
    b = env.reset(seed=7)
    c = env.reset(seed=8)
    assert a.shape == (STATE_DIM,)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_layout_respects_ranges_and_min_distance(env):
    cfg = env.cfg
    rng = np.random.default_rng(0)
    for _ in range(200):
        cube, goal, yaw = env.sample_layout(rng)
        assert cfg.cube_x[0] <= cube[0] <= cfg.cube_x[1]
        assert cfg.goal_y[0] <= goal[1] <= cfg.goal_y[1]
        assert np.linalg.norm(cube - goal) >= cfg.min_cube_goal_dist
        assert -np.pi <= yaw <= np.pi


def test_step_checks_action_shape(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(ACTION_DIM + 1))


def test_idle_arm_does_not_succeed_and_cube_rests_on_table(env):
    obs = env.reset(seed=1)
    hold = np.concatenate([obs[:5], [1.0]])
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(hold)
    assert not terminated and reward == 0.0
    assert info["cube_goal_dist"] >= env.cfg.min_cube_goal_dist - 0.01
    assert abs(env.cube_pos[2] - 0.02) < 0.005


def test_truncates_at_max_steps():
    env = PickPlaceEnv(EnvConfig(max_steps=3))
    obs = env.reset(seed=0)
    hold = np.concatenate([obs[:5], [1.0]])
    flags = [env.step(hold)[3] for _ in range(3)]
    assert flags == [False, False, True]


def test_joint_targets_are_clipped_to_limits(env):
    env.reset(seed=0)
    env.step(np.array([50.0, 50.0, 50.0, 50.0, 50.0, 1.0]))
    hi = env.model.jnt_range[:5, 1]
    assert np.all(env.data.ctrl[:5] <= hi + 1e-9)


def test_render_returns_image(env, can_render):
    env.reset(seed=0)
    img = env.render("front")
    assert img.shape == (*env.cfg.image_size, 3) and img.dtype == np.uint8


def test_quat_yaw():
    assert quat_yaw(np.array([1.0, 0, 0, 0])) == pytest.approx(0.0)
    half = np.pi / 4
    assert quat_yaw(np.array([np.cos(half), 0, 0, np.sin(half)])) == pytest.approx(np.pi / 2)
