"""Episode storage: one .npz with flat arrays plus the index where each episode ends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DemoSet:
    observations: np.ndarray  # (N, state_dim) float32
    actions: np.ndarray  # (N, action_dim) float32
    episode_ends: np.ndarray  # (E,) exclusive end index of each episode
    seeds: np.ndarray  # (E,) environment seed of each episode

    @property
    def n_episodes(self) -> int:
        return len(self.episode_ends)

    def episode(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        start = 0 if i == 0 else int(self.episode_ends[i - 1])
        end = int(self.episode_ends[i])
        return self.observations[start:end], self.actions[start:end]

    def subset(self, episode_ids: np.ndarray) -> DemoSet:
        obs, act, ends, seeds = [], [], [], []
        total = 0
        for i in episode_ids:
            o, a = self.episode(int(i))
            obs.append(o)
            act.append(a)
            total += len(o)
            ends.append(total)
            seeds.append(self.seeds[int(i)])
        return DemoSet(np.concatenate(obs), np.concatenate(act), np.array(ends), np.array(seeds))


def build_demoset(episodes: list[tuple[np.ndarray, np.ndarray]], seeds: list[int]) -> DemoSet:
    lengths = np.array([len(o) for o, _ in episodes])
    return DemoSet(
        observations=np.concatenate([o for o, _ in episodes]).astype(np.float32),
        actions=np.concatenate([a for _, a in episodes]).astype(np.float32),
        episode_ends=np.cumsum(lengths),
        seeds=np.array(seeds),
    )


def save_demoset(path: str | Path, demos: DemoSet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=demos.observations,
        actions=demos.actions,
        episode_ends=demos.episode_ends,
        seeds=demos.seeds,
    )


def load_demoset(path: str | Path) -> DemoSet:
    with np.load(path) as f:
        return DemoSet(f["observations"], f["actions"], f["episode_ends"], f["seeds"])


def split_episodes(
    n_episodes: int, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 0
) -> dict[str, np.ndarray]:
    """Random split by *episode*, never by frame, so no trajectory leaks across splits."""
    order = np.random.default_rng(seed).permutation(n_episodes)
    n_test = max(1, round(n_episodes * test_frac))
    n_val = max(1, round(n_episodes * val_frac))
    return {
        "test": np.sort(order[:n_test]),
        "val": np.sort(order[n_test : n_test + n_val]),
        "train": np.sort(order[n_test + n_val :]),
    }


def normalisation_stats(x: np.ndarray, eps: float = 1e-6) -> dict[str, np.ndarray]:
    return {"mean": x.mean(axis=0), "std": np.maximum(x.std(axis=0), eps)}


def chunk_targets(demos: DemoSet, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """For every frame, the next `chunk` actions and a mask that is False past the episode end.

    Padding repeats the last action (the expert holds still at the end), but the mask removes it
    from the loss, so the policy is never trained on made-up targets.
    """
    n = len(demos.actions)
    idx = np.arange(n)[:, None] + np.arange(chunk)[None, :]  # (N, K)
    ends = np.repeat(demos.episode_ends, np.diff(np.concatenate([[0], demos.episode_ends])))
    mask = idx < ends[:, None]
    idx = np.minimum(idx, ends[:, None] - 1)
    return demos.actions[idx], mask


def stack_history(demos: DemoSet, obs: np.ndarray, history: int) -> np.ndarray:
    """Concatenate each frame with the `history - 1` frames before it, oldest first.

    Frames before the episode start repeat its first frame, which is also what the policy sees
    at the start of a rollout. `obs` is `demos.observations`, usually normalised.
    """
    lengths = np.diff(np.concatenate([[0], demos.episode_ends]))
    starts = np.repeat(demos.episode_ends - lengths, lengths)
    idx = np.arange(len(obs))[:, None] + np.arange(1 - history, 1)[None, :]
    idx = np.maximum(idx, starts[:, None])
    return obs[idx].reshape(len(obs), -1)
