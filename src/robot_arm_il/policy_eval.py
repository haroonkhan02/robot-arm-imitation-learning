"""Roll a trained policy out on seeded episodes: `ra-policy-eval --ckpt models/act.pt`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import ACTION_DIM
from .env import PickPlaceEnv
from .policies import build_policy, pick_device


class PolicyRunner:
    """Wraps a checkpoint: observation in, action out, with optional temporal ensembling."""

    def __init__(self, ckpt_path: str, device: torch.device | None = None, ensemble: float = 0.0):
        self.device = device or pick_device()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = build_policy(ckpt["kind"], **ckpt["kwargs"]).to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        st = ckpt["stats"]
        self.obs_mean, self.obs_std = np.array(st["obs"]["mean"]), np.array(st["obs"]["std"])
        self.act_mean, self.act_std = np.array(st["act"]["mean"]), np.array(st["act"]["std"])
        self.chunk = self.model.chunk
        self.ensemble = ensemble  # weight exp(-m * age); 0 = uniform; <0 = newest chunk only
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.pending: dict[int, list[tuple[int, np.ndarray]]] = {}  # step -> [(age, action)]

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(((obs - self.obs_mean) / self.obs_std)[None], dtype=torch.float32)
        chunk = self.model(x.to(self.device))[0].cpu().numpy() * self.act_std + self.act_mean
        if self.chunk == 1:
            return chunk[0]
        for k in range(self.chunk):  # a chunk predicted now covers steps t .. t+K-1
            self.pending.setdefault(self.t + k, []).append((k, chunk[k]))
        cands = self.pending.pop(self.t)
        self.t += 1
        if self.ensemble < 0:  # no ensembling: newest chunk only (receding horizon)
            return min(cands, key=lambda c: c[0])[1]
        ages = np.array([a for a, _ in cands], dtype=float)
        w = np.exp(-self.ensemble * ages)
        return (np.stack([c for _, c in cands]) * w[:, None]).sum(0) / w.sum()


def rollout(env: PickPlaceEnv, runner: PolicyRunner, seed: int) -> dict:
    obs = env.reset(seed=seed)
    runner.reset()
    info = {"success": False, "cube_goal_dist": float("nan")}
    grasped = False
    steps = 0
    while True:
        action = runner.act(obs)
        assert action.shape == (ACTION_DIM,)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        grasped = grasped or bool(env.cube_pos[2] > 0.05)
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "success": bool(info["success"]),
        "steps": steps,
        "lifted": bool(grasped),
        "final_cube_goal_dist": float(info["cube_goal_dist"]),
    }


def evaluate_checkpoint(ckpt: str, episodes: int, seed_start: int, ensemble: float) -> dict:
    env = PickPlaceEnv()
    runner = PolicyRunner(ckpt, ensemble=ensemble)
    rows = [rollout(env, runner, seed_start + k) for k in range(episodes)]
    ok = np.array([r["success"] for r in rows])
    return {
        "checkpoint": ckpt,
        "episodes": episodes,
        "seed_start": seed_start,
        "ensemble": ensemble,
        "success_rate": float(ok.mean()),
        "lifted_rate": float(np.mean([r["lifted"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "episodes_detail": rows,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed-start", type=int, default=50_000)
    p.add_argument("--ensemble", type=float, default=0.01, help="ensemble decay; <0 = newest chunk")
    p.add_argument("--out", help="write the JSON summary here")
    args = p.parse_args(argv)
    res = evaluate_checkpoint(args.ckpt, args.episodes, args.seed_start, args.ensemble)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "episodes_detail"}))


if __name__ == "__main__":
    main()
