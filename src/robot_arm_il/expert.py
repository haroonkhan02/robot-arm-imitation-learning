"""Scripted pick-and-place expert that produces the demonstrations.

It reads the simulator's ground truth (cube pose, goal), plans Cartesian waypoints, and turns
them into joint-position targets with IK. The learner only ever sees the resulting
(observation, action) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .config import ACTION_DIM, N_ARM_JOINTS
from .env import PickPlaceEnv
from .kinematics import snap_yaw, solve_ik

HOVER_Z = 0.17  # grip-site height while carrying / approaching
GRASP_Z = 0.025  # site height at the cube centre when grasping
PLACE_Z = 0.03
MAX_SPEED = 0.35  # m/s, Cartesian
CARRY_SPEED = 0.15  # m/s while the cube is in the gripper
GRIPPER_WAIT = 8  # control steps to let the fingers open / close


@dataclass
class Waypoint:
    pos: np.ndarray
    yaw: float
    grip: float  # 1 open, 0 closed
    wait: int = 0  # steps to hold after arriving


class ScriptedExpert:
    """Stateful expert: call `reset(env)` after `env.reset()`, then `act(env)` each step."""

    def __init__(self):
        self._scratch: mujoco.MjData | None = None

    def reset(self, env: PickPlaceEnv) -> None:
        self._scratch = mujoco.MjData(env.model)
        cube = env.cube_pos
        goal = env.goal_xy
        yaw = self._pick_yaw(env, cube)
        up = HOVER_Z
        self.plan = [
            Waypoint(np.array([cube[0], cube[1], up]), yaw, 1.0),
            Waypoint(np.array([cube[0], cube[1], GRASP_Z]), yaw, 1.0),
            Waypoint(np.array([cube[0], cube[1], GRASP_Z]), yaw, 0.0, GRIPPER_WAIT),
            Waypoint(np.array([cube[0], cube[1], up]), yaw, 0.0),
            Waypoint(np.array([goal[0], goal[1], up]), yaw, 0.0),
            Waypoint(np.array([goal[0], goal[1], PLACE_Z]), yaw, 0.0),
            Waypoint(np.array([goal[0], goal[1], PLACE_Z]), yaw, 1.0, GRIPPER_WAIT),
            Waypoint(np.array([goal[0], goal[1], up]), yaw, 1.0, 3),
        ]
        self.idx = 0
        self.hold = 0
        self.target = env.ee_pos
        self.q_cmd = env.data.qpos[:N_ARM_JOINTS].copy()
        self.grip = 1.0
        self.dt = env.cfg.substeps * env.model.opt.timestep

    def _pick_yaw(self, env: PickPlaceEnv, cube: np.ndarray) -> float:
        """Grasp yaw that needs the least wrist roll.

        The two fingers are symmetric and so is the cube, so every yaw + k * 90 degrees is a valid
        grasp. IK from the home pose happily lands on the one that winds the roll joint to its
        limit, so try all four and keep the smallest roll.
        """
        base = snap_yaw(env.cube_yaw)
        best, best_roll = base, np.inf
        grasp = np.array([cube[0], cube[1], GRASP_Z])
        for k in range(4):
            yaw = base + k * np.pi / 2
            q, err = solve_ik(env.model, self._scratch, grasp, yaw, env.data.qpos[:N_ARM_JOINTS])
            if err < 1e-3 and abs(q[4]) < best_roll:
                best, best_roll = yaw, abs(q[4])
        return float(best)

    @property
    def done(self) -> bool:
        return self.idx >= len(self.plan)

    def act(self, env: PickPlaceEnv) -> np.ndarray:
        if not self.done:
            wp = self.plan[self.idx]
            delta = wp.pos - self.target
            dist = float(np.linalg.norm(delta))
            carrying = 3 <= self.idx <= 5
            step = (CARRY_SPEED if carrying else MAX_SPEED) * self.dt
            self.target = wp.pos.copy() if dist <= step else self.target + delta / dist * step
            self.grip = wp.grip
            reached = np.linalg.norm(wp.pos - self.target) < 1e-6
            if reached and np.linalg.norm(wp.pos - env.ee_pos) < 0.012:
                if self.hold >= wp.wait:
                    self.idx += 1
                    self.hold = 0
                else:
                    self.hold += 1
            wp = self.plan[min(self.idx, len(self.plan) - 1)]
            q, _ = solve_ik(env.model, self._scratch, self.target, wp.yaw, self.q_cmd)
            self.q_cmd = q
        action = np.empty(ACTION_DIM)
        action[:N_ARM_JOINTS] = self.q_cmd
        action[-1] = self.grip
        return action


def run_episode(env: PickPlaceEnv, seed: int, exec_noise: float = 0.0, frames: list | None = None):
    """Roll the expert out once. Returns (observations, actions, success).

    `exec_noise` perturbs the joint targets that are *executed*; the recorded action stays
    clean, so the dataset also contains recoveries from off-trajectory states (DART-style).
    If `frames` is a list, front-camera images are appended to it.
    """
    expert = ScriptedExpert()
    rng = np.random.default_rng(seed + 10_000)
    obs = env.reset(seed=seed)
    expert.reset(env)
    observations, actions = [], []
    info = {"success": False}
    while True:
        a = expert.act(env)
        observations.append(obs)
        actions.append(a)
        if frames is not None:
            frames.append(env.render("front"))
        executed = a.copy()
        if exec_noise > 0:
            executed[:N_ARM_JOINTS] += rng.normal(0.0, exec_noise, N_ARM_JOINTS)
        obs, _, terminated, truncated, info = env.step(executed)
        if terminated or truncated:
            break
    return np.array(observations), np.array(actions, dtype=np.float32), bool(info["success"])
