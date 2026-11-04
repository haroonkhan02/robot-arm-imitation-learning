"""MuJoCo pick-and-place environment: grasp the cube and put it on the green marker."""

from __future__ import annotations

import mujoco
import numpy as np

from .config import ACTION_DIM, GRIPPER_MAX, N_ARM_JOINTS, STATE_DIM, EnvConfig, model_path


def quat_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class PickPlaceEnv:
    """Minimal gym-style environment (no gymnasium dependency).

    Action: 5 absolute joint-position targets (rad) + gripper command in [0, 1] (1 = open).
    State observation (13): joints(5), gripper opening(1), cube xyz(3), sin/cos cube yaw(2),
    goal xy(2). Images come from `render()` on demand.
    """

    def __init__(self, config: EnvConfig | None = None):
        self.cfg = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_path(model_path())
        self.data = mujoco.MjData(self.model)
        self._cube_qadr = self.model.joint("cube_free").qposadr[0]
        self._goal_site = self.model.site("goal").id
        self._renderers: dict[str, mujoco.Renderer] = {}
        self.rng = np.random.default_rng(0)
        self.goal_xy = np.zeros(2)
        self.steps = 0
        self._stable = 0

    # ------------------------------------------------------------------ helpers
    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._cube_qadr : self._cube_qadr + 3].copy()

    @property
    def cube_yaw(self) -> float:
        return quat_yaw(self.data.qpos[self._cube_qadr + 3 : self._cube_qadr + 7])

    @property
    def ee_pos(self) -> np.ndarray:
        return self.data.site("grip").xpos.copy()

    def gripper_opening(self) -> float:
        return float(self.data.qpos[N_ARM_JOINTS] + self.data.qpos[N_ARM_JOINTS + 1])

    def observation(self) -> np.ndarray:
        yaw = self.cube_yaw
        obs = np.concatenate(
            [
                self.data.qpos[:N_ARM_JOINTS],
                [self.gripper_opening()],
                self.cube_pos,
                [np.sin(yaw), np.cos(yaw)],
                self.goal_xy,
            ]
        )
        assert obs.shape == (STATE_DIM,)
        return obs.astype(np.float32)

    # ------------------------------------------------------------------- API
    def sample_layout(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, float]:
        c = self.cfg
        while True:
            cube = np.array([rng.uniform(*c.cube_x), rng.uniform(*c.cube_y)])
            goal = np.array([rng.uniform(*c.goal_x), rng.uniform(*c.goal_y)])
            if np.linalg.norm(cube - goal) >= c.min_cube_goal_dist:
                return cube, goal, float(rng.uniform(-np.pi, np.pi))

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        cube, goal, yaw = self.sample_layout(self.rng)
        self.goal_xy = goal
        self.model.site_pos[self._goal_site] = [goal[0], goal[1], 0.001]
        adr = self._cube_qadr
        self.data.qpos[adr : adr + 3] = [cube[0], cube[1], 0.02]
        self.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qpos[:N_ARM_JOINTS] = self.cfg.home_q
        self.data.ctrl[:N_ARM_JOINTS] = self.cfg.home_q
        self.data.ctrl[N_ARM_JOINTS:] = GRIPPER_MAX
        self.data.qpos[N_ARM_JOINTS : N_ARM_JOINTS + 2] = GRIPPER_MAX
        mujoco.mj_forward(self.model, self.data)
        for _ in range(40):  # let the cube settle and the arm reach its home pose
            mujoco.mj_step(self.model, self.data)
        self.steps = 0
        self._stable = 0
        return self.observation()

    def is_success(self) -> bool:
        pos = self.cube_pos
        speed = np.linalg.norm(self.data.qvel[self.model.joint("cube_free").dofadr[0] :][:3])
        near = np.linalg.norm(pos[:2] - self.goal_xy) < self.cfg.goal_tol
        return bool(near and pos[2] < 0.03 and speed < 0.05)

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {action.shape}")
        lo = self.model.jnt_range[:N_ARM_JOINTS, 0]
        hi = self.model.jnt_range[:N_ARM_JOINTS, 1]
        self.data.ctrl[:N_ARM_JOINTS] = np.clip(action[:N_ARM_JOINTS], lo, hi)
        self.data.ctrl[N_ARM_JOINTS:] = GRIPPER_MAX * float(np.clip(action[-1], 0.0, 1.0))
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        # success counts once the cube rests on the marker with the gripper open again
        released = self.gripper_opening() > 0.8 * 2 * GRIPPER_MAX
        self._stable = self._stable + 1 if (self.is_success() and released) else 0
        terminated = self._stable >= 5
        truncated = self.steps >= self.cfg.max_steps
        dist = float(np.linalg.norm(self.cube_pos[:2] - self.goal_xy))
        info = {"success": terminated, "cube_goal_dist": dist}
        return self.observation(), float(terminated), terminated, truncated, info

    def render(self, camera: str = "front") -> np.ndarray:
        if camera not in self._renderers:
            h, w = self.cfg.image_size
            self._renderers[camera] = mujoco.Renderer(self.model, height=h, width=w)
        r = self._renderers[camera]
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
