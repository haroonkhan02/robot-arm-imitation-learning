"""Generate scripted-expert demonstrations: `ra-collect --episodes 500`."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import build_demoset, save_demoset
from .env import PickPlaceEnv
from .expert import run_episode


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed-start", type=int, default=1000, help="env seed of the first episode")
    p.add_argument("--exec-noise", type=float, default=0.0, help="std (rad) of executed noise")
    p.add_argument("--out", default="data/demos.npz")
    p.add_argument("--keep-failures", action="store_true")
    args = p.parse_args(argv)

    env = PickPlaceEnv()
    episodes, seeds, n_fail = [], [], 0
    t0 = time.time()
    for k in range(args.episodes):
        seed = args.seed_start + k
        obs, act, ok = run_episode(env, seed, exec_noise=args.exec_noise)
        if ok or args.keep_failures:
            episodes.append((obs, act))
            seeds.append(seed)
        else:
            n_fail += 1
    demos = build_demoset(episodes, seeds)
    save_demoset(args.out, demos)
    lengths = np.diff(np.concatenate([[0], demos.episode_ends]))
    summary = {
        "episodes_kept": demos.n_episodes,
        "episodes_failed_and_dropped": n_fail,
        "frames": int(len(demos.observations)),
        "mean_len": float(lengths.mean()),
        "exec_noise": args.exec_noise,
        "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
