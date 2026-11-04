"""Environment constants and the task randomisation ranges."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources

N_ARM_JOINTS = 5
ACTION_DIM = N_ARM_JOINTS + 1  # five joint targets + gripper command (1 = open)
STATE_DIM = 13
GRIPPER_MAX = 0.045  # slide range of each finger, metres


def model_path() -> str:
    return str(resources.files("robot_arm_il") / "assets" / "arm_pick_place.xml")


@dataclass(frozen=True)
class EnvConfig:
    """Control rate is 1 / (substeps * timestep) = 20 Hz with the shipped model."""

    substeps: int = 10
    max_steps: int = 220
    cube_x: tuple[float, float] = (0.27, 0.42)
    cube_y: tuple[float, float] = (-0.22, -0.02)
    goal_x: tuple[float, float] = (0.27, 0.42)
    goal_y: tuple[float, float] = (0.08, 0.24)
    goal_tol: float = 0.025  # xy distance between cube centre and goal centre
    min_cube_goal_dist: float = 0.15
    image_size: tuple[int, int] = (96, 96)
    home_q: tuple[float, ...] = field(default=(0.0, 0.2, 1.3, 1.64, 0.0))
