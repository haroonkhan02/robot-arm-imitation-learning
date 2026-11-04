import mujoco
import numpy as np
import pytest

from robot_arm_il.config import model_path
from robot_arm_il.kinematics import grasp_rotation, snap_yaw, solve_ik


def test_grasp_rotation_is_a_proper_rotation_pointing_down():
    r = grasp_rotation(0.7)
    assert np.allclose(r.T @ r, np.eye(3))
    assert np.linalg.det(r) == pytest.approx(1.0)
    assert np.allclose(r[:, 2], [0, 0, -1])


@pytest.mark.parametrize("yaw", np.linspace(-np.pi, np.pi, 25))
def test_snap_yaw_is_within_45_degrees_and_cube_symmetric(yaw):
    snapped = snap_yaw(yaw)
    assert -np.pi / 4 - 1e-9 <= snapped <= np.pi / 4 + 1e-9
    k = (yaw - snapped) / (np.pi / 2)
    assert k == pytest.approx(round(k), abs=1e-9)


@pytest.mark.parametrize("target", [(0.3, -0.15, 0.025), (0.4, 0.1, 0.17), (0.28, 0.2, 0.03)])
def test_ik_reaches_grasp_poses(target):
    model = mujoco.MjModel.from_xml_path(model_path())
    data = mujoco.MjData(model)
    q0 = np.array([0.0, 0.2, 1.3, 1.64, 0.0])
    # IK is local: from home some yaw choices wind the roll joint into a limit. The expert
    # tries the four equivalent 90-degree grasp yaws; at least one must converge.
    sols = [solve_ik(model, data, np.array(target), 0.3 + k * np.pi / 2, q0) for k in range(4)]
    q, err = min(sols, key=lambda s: s[1])
    assert err < 1e-3
    data.qpos[:5] = q
    mujoco.mj_kinematics(model, data)
    assert np.allclose(data.site("grip").xpos, target, atol=2e-3)
    lo, hi = model.jnt_range[:5, 0], model.jnt_range[:5, 1]
    assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
