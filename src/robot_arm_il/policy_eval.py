"""Roll a trained policy out on seeded episodes: `ra-policy-eval --ckpt models/act.pt`."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import ACTION_DIM, GRIPPER_MAX, N_ARM_JOINTS
from .env import PickPlaceEnv
from .policies import build_policy, pick_device

STAGES = ("reached", "grasped", "lifted", "at_goal", "released")


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
        self.history = getattr(self.model, "history", 1)
        self.ensemble = ensemble  # weight exp(-m * age); 0 = uniform; <0 = newest chunk only
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.pending: dict[int, list[tuple[int, np.ndarray]]] = {}  # step -> [(age, action)]
        self.frames: deque[np.ndarray] = deque(maxlen=self.history)

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """The raw (K, A) action chunk for this observation, in environment units."""
        norm = (obs - self.obs_mean) / self.obs_std
        if not self.frames:  # before the first step the history repeats the first frame
            self.frames.extend([norm] * self.history)
        self.frames.append(norm)
        x = torch.as_tensor(np.concatenate(self.frames)[None], dtype=torch.float32)
        return self.model(x.to(self.device))[0].cpu().numpy() * self.act_std + self.act_mean

    def act(self, obs: np.ndarray) -> np.ndarray:
        chunk = self.predict(obs)
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


class StageTracker:
    """Records the first step at which an episode reaches each stage of the task."""

    def __init__(self, env: PickPlaceEnv):
        self.first: dict[str, int] = {}
        self.start_xy = env.cube_pos[:2]
        self.knocked = False  # cube pushed more than 1 cm before it was grasped
        self.dropped = False  # cube back on the table away from the goal after a lift

    def update(self, env: PickPlaceEnv, step: int) -> None:
        cube, ee, opening = env.cube_pos, env.ee_pos, env.gripper_opening()
        goal_dist = float(np.linalg.norm(cube[:2] - env.goal_xy))
        at_cube = np.linalg.norm(ee[:2] - cube[:2]) < 0.02 and ee[2] < 0.05
        if at_cube:
            self.first.setdefault("reached", step)
        if at_cube and opening < 0.045:
            self.first.setdefault("grasped", step)
        if "grasped" not in self.first and np.linalg.norm(cube[:2] - self.start_xy) > 0.01:
            self.knocked = True
        if cube[2] > 0.05:
            self.first.setdefault("lifted", step)
        if "lifted" in self.first:
            if goal_dist < env.cfg.goal_tol:
                self.first.setdefault("at_goal", step)
            elif cube[2] < 0.03:
                self.dropped = True
        if "at_goal" in self.first and cube[2] < 0.03 and opening > 0.8 * 2 * GRIPPER_MAX:
            self.first.setdefault("released", step)

    def failed_at(self, success: bool) -> str | None:
        """The first stage the episode never reached; None for a success."""
        if success:
            return None
        return next((s for s in STAGES if s not in self.first), "settled")


def rollout(
    env: PickPlaceEnv,
    runner: PolicyRunner,
    seed: int,
    exec_noise: float = 0.0,
    frames: list | None = None,
) -> dict:
    """One seeded episode. `exec_noise` adds Gaussian noise (rad) to the executed joint targets."""
    obs = env.reset(seed=seed)
    runner.reset()
    rng = np.random.default_rng(seed + 10_000)
    tracker = StageTracker(env)
    layout = {
        "cube_xy": [float(v) for v in env.cube_pos[:2]],
        "cube_yaw": env.cube_yaw,
        "goal_xy": [float(v) for v in env.goal_xy],
    }
    info = {"success": False, "cube_goal_dist": float("nan")}
    steps = 0
    while True:
        action = runner.act(obs)
        assert action.shape == (ACTION_DIM,)
        if frames is not None:
            frames.append(env.render("front"))
        executed = action.copy()
        if exec_noise > 0:
            executed[:N_ARM_JOINTS] += rng.normal(0.0, exec_noise, N_ARM_JOINTS)
        obs, _, terminated, truncated, info = env.step(executed)
        steps += 1
        tracker.update(env, steps)
        if terminated or truncated:
            break
    success = bool(info["success"])
    return {
        "seed": seed,
        "success": success,
        "steps": steps,
        "lifted": "lifted" in tracker.first,
        "final_cube_goal_dist": float(info["cube_goal_dist"]),
        "stages": tracker.first,
        "failed_at": tracker.failed_at(success),
        "knocked": tracker.knocked,
        "dropped": tracker.dropped,
        **layout,
    }


def evaluate_checkpoint(
    ckpt: str,
    episodes: int,
    seed_start: int,
    ensemble: float,
    exec_noise: float = 0.0,
    device: str | None = None,
) -> dict:
    env = PickPlaceEnv()
    runner = PolicyRunner(ckpt, torch.device(device) if device else None, ensemble)
    rows = [rollout(env, runner, seed_start + k, exec_noise) for k in range(episodes)]
    ok = np.array([r["success"] for r in rows])
    return {
        "checkpoint": ckpt,
        "episodes": episodes,
        "seed_start": seed_start,
        "ensemble": ensemble,
        "exec_noise": exec_noise,
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
    p.add_argument("--exec-noise", type=float, default=0.0, help="joint noise while executing, rad")
    p.add_argument("--device", help="cpu or mps (default: mps when available)")
    p.add_argument("--out", help="write the JSON summary here")
    args = p.parse_args(argv)
    res = evaluate_checkpoint(
        args.ckpt, args.episodes, args.seed_start, args.ensemble, args.exec_noise, args.device
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "episodes_detail"}))


if __name__ == "__main__":
    main()
