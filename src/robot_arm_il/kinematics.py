"""Damped-least-squares inverse kinematics for the gripper site."""

from __future__ import annotations

import mujoco
import numpy as np

from .config import N_ARM_JOINTS


def grasp_rotation(yaw: float) -> np.ndarray:
    """Gripper pointing straight down with its local x axis at `yaw` in the xy plane."""
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def snap_yaw(cube_yaw: float) -> float:
    """A square cube looks the same every 90 degrees: pick the smallest rotation."""
    return float((cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def solve_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    yaw: float,
    q0: np.ndarray,
    site: str = "grip",
    iters: int = 60,
    damping: float = 1e-2,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Return arm joint angles that put `site` at `pos` pointing down with the given yaw.

    Works on `data`, so pass a scratch MjData. Returns (q, residual norm).
    """
    r_target = grasp_rotation(yaw)
    sid = model.site(site).id
    q = np.array(q0, dtype=float)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    lo = model.jnt_range[:N_ARM_JOINTS, 0]
    hi = model.jnt_range[:N_ARM_JOINTS, 1]
    err = np.inf
    for _ in range(iters):
        data.qpos[:N_ARM_JOINTS] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        r_cur = data.site_xmat[sid].reshape(3, 3)
        e_pos = pos - data.site_xpos[sid]
        e_rot = 0.5 * sum(np.cross(r_cur[:, i], r_target[:, i]) for i in range(3))
        e = np.concatenate([e_pos, e_rot])
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)
        jac = np.vstack([jacp[:, :N_ARM_JOINTS], jacr[:, :N_ARM_JOINTS]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), e)
        q = np.clip(q + dq, lo, hi)
    return q, err
