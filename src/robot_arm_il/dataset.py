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
