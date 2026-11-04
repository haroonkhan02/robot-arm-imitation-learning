"""Measure the scripted expert's success rate and optionally record a video: `ra-expert-eval`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import PickPlaceEnv
from .expert import run_episode


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-start", type=int, default=50_000)
    p.add_argument("--exec-noise", type=float, default=0.0)
    p.add_argument("--video", help="write an mp4/gif of the first episode here")
    p.add_argument("--out", default="reports/expert_baseline.json")
    args = p.parse_args(argv)

    env = PickPlaceEnv(EnvConfig(image_size=(360, 480)) if args.video else None)
    results, lengths = [], []
    for k in range(args.episodes):
        frames: list | None = [] if (args.video and k == 0) else None
        obs, _, ok = run_episode(env, args.seed_start + k, args.exec_noise, frames)
        results.append(ok)
        lengths.append(len(obs))
        if frames:
            import imageio.v3 as iio

            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(args.video, np.stack(frames), fps=20)
    summary = {
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "exec_noise": args.exec_noise,
        "success_rate": float(np.mean(results)),
        "mean_episode_steps": float(np.mean(lengths)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
