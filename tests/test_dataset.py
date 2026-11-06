import json

import numpy as np

from robot_arm_il.collect import main as collect_main
from robot_arm_il.dataset import (
    DemoSet,
    build_demoset,
    chunk_targets,
    load_demoset,
    normalisation_stats,
    save_demoset,
    split_episodes,
    stack_history,
)


def _episodes():
    rng = np.random.default_rng(0)
    return [(rng.normal(size=(n, 13)), rng.normal(size=(n, 6))) for n in (5, 7, 4)]


def test_roundtrip_and_episode_access(tmp_path):
    demos = build_demoset(_episodes(), [10, 11, 12])
    save_demoset(tmp_path / "d.npz", demos)
    back = load_demoset(tmp_path / "d.npz")
    assert back.n_episodes == 3 and list(back.episode_ends) == [5, 12, 16]
    o, a = back.episode(1)
    assert o.shape == (7, 13) and a.shape == (7, 6)
    assert np.allclose(o, demos.observations[5:12])


def test_subset_keeps_boundaries_and_seeds():
    demos = build_demoset(_episodes(), [10, 11, 12])
    sub = demos.subset(np.array([2, 0]))
    assert list(sub.episode_ends) == [4, 9] and list(sub.seeds) == [12, 10]
    assert np.allclose(sub.episode(0)[0], demos.episode(2)[0])


def test_splits_are_disjoint_complete_and_seeded():
    s = split_episodes(50, 0.1, 0.2, seed=1)
    ids = np.concatenate([s["train"], s["val"], s["test"]])
    assert sorted(ids) == list(range(50))
    assert len(s["test"]) == 10 and len(s["val"]) == 5
    assert all(np.array_equal(s[k], split_episodes(50, 0.1, 0.2, seed=1)[k]) for k in s)


def test_normalisation_stats_guard_constant_columns():
    x = np.stack([np.arange(10.0), np.ones(10)], axis=1)
    st = normalisation_stats(x)
    assert st["std"][1] > 0 and np.isclose(st["mean"][0], 4.5)


def test_collect_cli_writes_dataset_and_summary(tmp_path):
    out = tmp_path / "demos.npz"
    collect_main(["--episodes", "3", "--seed-start", "0", "--out", str(out)])
    demos = load_demoset(out)
    summary = json.loads(out.with_suffix(".json").read_text())
    assert demos.n_episodes == summary["episodes_kept"] == 3
    assert summary["frames"] == len(demos.observations)


def test_chunk_targets_follow_the_episode_and_mask_the_tail():
    demos = build_demoset(_episodes(), [10, 11, 12])  # lengths 5, 7, 4
    actions, mask = chunk_targets(demos, 3)
    assert actions.shape == (16, 3, 6) and mask.shape == (16, 3)
    assert np.allclose(actions[0, 1], demos.actions[1])
    # last frame of episode 0: only the first slot is real, nothing from episode 1 leaks in
    assert list(mask[4]) == [True, False, False]
    assert np.allclose(actions[4, 1], demos.actions[4])
    assert mask[:3].all() and list(mask[5 + 5]) == [True, True, False]


def test_stack_history_is_oldest_first_and_stops_at_the_episode_start():
    obs = np.arange(5, dtype=np.float32)[:, None]  # episodes: frames 0-1 and 2-4
    demos = DemoSet(obs, np.zeros((5, 6), np.float32), np.array([2, 5]), np.array([0, 1]))
    stacked = stack_history(demos, obs, 3)
    assert stacked.tolist() == [[0, 0, 0], [0, 0, 1], [2, 2, 2], [2, 2, 3], [2, 3, 4]]
