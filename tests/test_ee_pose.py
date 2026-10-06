from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET

import pytest

from arena_so101 import CUROBO_ROBOT_YML, HOME_JOINT_POS, PAN_AXIS_XY
from arena_so101.ee_pose import HOME_NATURAL_POSE, natural_ee_quat_xyzw

np = pytest.importorskip("numpy")

URDF = CUROBO_ROBOT_YML.parent / "urdf" / "SO-ARM101-USD.urdf"


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _mat_from_quat_xyzw(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _angle_between(r1, r2) -> float:
    return math.acos(max(-1.0, min(1.0, (np.trace(r1.T @ r2) - 1) / 2)))


def _tcp_pose_from_urdf(joint_pos: dict[str, float]):
    """Forward kinematics of the shipped URDF: the ``tcp`` link in the ``base`` frame."""
    joints = {j.get("name"): j for j in ET.parse(URDF).getroot().findall("joint")}
    pose = np.eye(4)
    for name in ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "tcp_joint"):
        origin = joints[name].find("origin")
        step = np.eye(4)
        step[:3, :3] = _rpy(*(float(v) for v in origin.get("rpy").split()))
        step[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        pose = pose @ step
        if joints[name].get("type") == "revolute":
            assert joints[name].find("axis").get("xyz").split() == ["0", "0", "1"]
            turn = np.eye(4)
            turn[:3, :3] = _rpy(0.0, 0.0, joint_pos[name])
            pose = pose @ turn
    return pose[:3, 3], pose[:3, :3]


def test_quaternion_is_the_zyz_rotation_azimuth_tilt_roll():
    # The matrix form this was ported from: rot0 (x radial, z up) @ Ry(tilt) @ Rz(roll).
    rng = random.Random(0)
    for _ in range(200):
        x, y = rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4)
        tilt, roll = rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)
        psi = math.atan2(y - PAN_AXIS_XY[1], x - PAN_AXIS_XY[0])
        expected = _rpy(0, 0, psi) @ _rpy(0, tilt, 0) @ _rpy(0, 0, roll)
        q = natural_ee_quat_xyzw(x, y, tilt, roll)
        assert math.hypot(*q) == pytest.approx(1.0)
        assert np.allclose(_mat_from_quat_xyzw(q), expected, atol=1e-9)


def test_tool_frame_faces_away_from_the_pan_axis_with_the_jaws_down():
    for x, y in ((0.3, 0.0), (0.0, -0.3), (-0.1, 0.2)):
        rot = _mat_from_quat_xyzw(natural_ee_quat_xyzw(x, y))
        radial = np.array([x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1], 0.0])
        assert np.allclose(rot[:, 0], radial / np.linalg.norm(radial), atol=1e-9)  # tool X: out along the arm
        assert np.allclose(rot[:, 2], [0.0, 0.0, 1.0], atol=1e-9)  # tool Z up: the jaws point down
    # Over the pan axis the azimuth is taken as 0, not NaN.
    assert math.hypot(*natural_ee_quat_xyzw(*PAN_AXIS_XY)) == pytest.approx(1.0)
    # Negative tilt swings the fingertips (-Z) away from the base.
    tips = -_mat_from_quat_xyzw(natural_ee_quat_xyzw(0.3, 0.0, tilt=-0.5))[:, 2]
    assert tips[0] > 0 and tips[2] < 0


def test_home_pose_is_the_urdf_tcp_at_the_home_joints():
    pos, rot = _tcp_pose_from_urdf(dict(HOME_JOINT_POS))
    x, y, z, tilt, roll = HOME_NATURAL_POSE
    assert np.allclose(pos, [x, y, z], atol=1e-3)
    # The arm's real orientation is natural up to the TCP's 7 mm offset from the roll axis (~1.3° here).
    natural = _mat_from_quat_xyzw(natural_ee_quat_xyzw(x, y, tilt, roll))
    assert _angle_between(natural, rot) < math.radians(2.0)
    # And the recorded tilt/roll are the best fit, not merely close: nudging either makes it worse.
    for d_tilt, d_roll in ((0.02, 0.0), (-0.02, 0.0), (0.0, 0.02), (0.0, -0.02)):
        nudged = _mat_from_quat_xyzw(natural_ee_quat_xyzw(x, y, tilt + d_tilt, roll + d_roll))
        assert _angle_between(nudged, rot) > _angle_between(natural, rot)
