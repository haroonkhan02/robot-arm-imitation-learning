"""Side-by-side rollout GIF of two checkpoints on one seed: `ra-demo --left a.pt --right b.pt`."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import EnvConfig
from .env import PickPlaceEnv
from .policy_eval import PolicyRunner, rollout


def label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    width = draw.textlength(text)
    draw.rectangle((4, 4, 12 + width, 22), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return np.asarray(img)


def side_by_side(left: list, right: list, step: int = 1, hold: int = 10) -> list[np.ndarray]:
    """Join two frame lists. The shorter one holds its last frame, and so does the ending."""
    n = max(len(left), len(right))
    idx = list(range(0, n, step)) + [n - 1] * hold
    return [
        np.concatenate([left[min(i, len(left) - 1)], right[min(i, len(right) - 1)]], axis=1)
        for i in idx
    ]


def record(ckpt: str, name: str, seed: int, ensemble: float, size: tuple[int, int]) -> list:
    env = PickPlaceEnv(EnvConfig(image_size=size))
    frames: list[np.ndarray] = []
    res = rollout(env, PolicyRunner(ckpt, torch.device("cpu"), ensemble), seed, frames=frames)
    env.close()
    outcome = "success" if res["success"] else f"failed: never {res['failed_at']}"
    print(f"{name}: {outcome} after {res['steps']} steps")
    last = len(frames) - 1
    return [label(f, name if i < last else f"{name}: {outcome}") for i, f in enumerate(frames)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--left-name", default="left")
    p.add_argument("--right-name", default="right")
    p.add_argument("--seed", type=int, default=50_000)
    p.add_argument("--ensemble", type=float, default=0.01)
    p.add_argument("--size", type=int, nargs=2, default=(240, 320), metavar=("H", "W"))
    p.add_argument("--step", type=int, default=2, help="keep every n-th control step")
    p.add_argument("--out", default="assets/rollouts.gif")
    args = p.parse_args(argv)

    size = tuple(args.size)
    left = record(args.left, args.left_name, args.seed, args.ensemble, size)
    right = record(args.right, args.right_name, args.seed, args.ensemble, size)
    frames = side_by_side(left, right, args.step)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, np.stack(frames), duration=1000 * args.step / 20, loop=0)
    print(f"{args.out}: {len(frames)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
