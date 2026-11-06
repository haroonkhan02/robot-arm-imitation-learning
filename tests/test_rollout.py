import numpy as np
import pytest

pytest.importorskip("torch")

from robot_arm_il.env import PickPlaceEnv  # noqa: E402
from robot_arm_il.expert import ScriptedExpert  # noqa: E402
from robot_arm_il.policies import MLPPolicy  # noqa: E402
from robot_arm_il.policy_eval import STAGES, PolicyRunner, StageTracker, rollout  # noqa: E402
from robot_arm_il.train import save_checkpoint  # noqa: E402


def test_expert_episode_passes_every_stage_in_order():
    env = PickPlaceEnv()
    env.reset(seed=3)
    expert = ScriptedExpert()
    expert.reset(env)
    tracker = StageTracker(env)
    for step in range(1, env.cfg.max_steps + 1):
        _, _, terminated, _, _ = env.step(expert.act(env))
        tracker.update(env, step)
        if terminated:
            break
    assert terminated and tracker.failed_at(True) is None
    firsts = [tracker.first[s] for s in STAGES]
    assert firsts == sorted(firsts)
    assert not tracker.knocked and not tracker.dropped


def test_idle_arm_fails_at_the_first_stage():
    env = PickPlaceEnv()
    env.reset(seed=3)
    tracker = StageTracker(env)
    tracker.update(env, 1)
    assert tracker.failed_at(False) == "reached"


@pytest.fixture
def ckpt(tmp_path):
    model = MLPPolicy(history=3)
    stats = {
        "obs": {"mean": np.zeros(13), "std": np.ones(13)},
        "act": {"mean": np.zeros(6), "std": np.ones(6)},
    }
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, "bc", {"history": 3}, stats, {})
    return str(path)


def test_runner_history_repeats_first_frame_then_slides(ckpt):
    runner = PolicyRunner(ckpt, device=__import__("torch").device("cpu"))
    a, b = np.zeros(13, np.float32), np.ones(13, np.float32)
    runner.predict(a)
    assert len(runner.frames) == 3 and all((f == a).all() for f in runner.frames)
    runner.predict(b)
    assert (runner.frames[0] == a).all() and (runner.frames[2] == b).all()
    runner.reset()
    assert len(runner.frames) == 0


def test_exec_noise_is_seeded_and_changes_the_rollout(ckpt):
    env = PickPlaceEnv()
    runner = PolicyRunner(ckpt, device=__import__("torch").device("cpu"))
    plain = rollout(env, runner, 7)
    noisy = rollout(env, runner, 7, exec_noise=0.05)
    again = rollout(env, runner, 7, exec_noise=0.05)
    assert noisy == again
    assert noisy != plain
