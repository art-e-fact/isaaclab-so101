from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET

import pytest

from isaaclab_so101 import CUROBO_ROBOT_YML, HOME_JOINT_POS, JOINT_LIMITS_RAD, PAN_AXIS_XY, TCP_OFFSET
from isaaclab_so101.ee_pose import HOME_NATURAL_POSE, natural_ee_quat_xyzw, pan_reach_from_xy, xy_from_pan_reach

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


def _natural_from_urdf(joint_pos: dict[str, float], *, on_axis: bool = False):
    """The URDF TCP pose as (x, y, z, tilt, roll), the tilt and roll read off the gripper's rotation. ``on_axis``
    takes the point on the roll axis beside the TCP instead, where natural_ik's model has it."""
    pos, rot = _tcp_pose_from_urdf(joint_pos)
    if on_axis:
        pos = pos - rot @ [TCP_OFFSET[0], 0.0, 0.0]
    psi = math.atan2(pos[1] - PAN_AXIS_XY[1], pos[0] - PAN_AXIS_XY[0])
    local = _rpy(0, 0, -psi) @ rot  # Ry(tilt) @ Rz(roll)
    return (*pos, math.atan2(local[0, 2], local[2, 2]), math.atan2(local[1, 0], local[1, 1]))


def test_natural_ik_inverts_the_urdf_forward_kinematics():
    from isaaclab_so101.ee_pose import natural_ik

    rng = random.Random(0)
    names = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
    checked = 0
    for _ in range(2000):
        joints = {n: rng.uniform(*JOINT_LIMITS_RAD[i]) for i, n in enumerate(names)}
        x, y, z, tilt, roll = _natural_from_urdf(joints, on_axis=True)
        psi, reach = pan_reach_from_xy(x, y)
        # Skip poses the natural coordinates do not describe: fingertips folded back behind the pan axis (seen
        # from the pan axis they face the wrong way) or right over it.
        if reach < 0.05 or math.cos(psi - (-math.pi / 2 - joints["Rotation"])) <= 0:
            continue
        if joints["Elbow"] < math.radians(2.21 - 76.03) + 0.01:  # hyperextended elbow: not the bend natural_ik gives
            continue
        q = natural_ik(x, y, z, tilt, roll)
        assert q is not None, joints
        # The model is the URDF's planar chain: the solution puts the roll axis back exactly where it was.
        assert _natural_from_urdf(dict(zip(names, q)), on_axis=True)[:4] == pytest.approx((x, y, z, tilt), abs=1e-3)
        # The roll axis misses the pan axis by 0.2 mm, which skews the azimuth by 0.0002 / reach; the roll absorbs it.
        roll2 = _natural_from_urdf(dict(zip(names, q)), on_axis=True)[4]
        assert math.remainder(roll2 - roll, math.tau) == pytest.approx(0, abs=1e-3 + 0.0003 / reach)
        checked += 1
    assert checked > 1000
    # The real TCP is 7 mm off the roll axis, which the model ignores: at home the joints come out within 3° (the
    # offset's skew of the azimuth lands on the pan) and the URDF TCP lands within 7 mm of the target.
    q = natural_ik(*HOME_NATURAL_POSE)
    assert q is not None
    assert np.allclose(q, [HOME_JOINT_POS[n] for n in names], atol=math.radians(3))
    pos, _ = _tcp_pose_from_urdf(dict(zip(names, q)))
    assert np.linalg.norm(pos - HOME_NATURAL_POSE[:3]) < 0.0075


def test_natural_ik_refuses_what_the_arm_cannot_do():
    from isaaclab_so101.ee_pose import natural_ik

    x, y, z, tilt, roll = HOME_NATURAL_POSE
    assert natural_ik(x, y - 0.3, z, tilt, roll) is None  # 50 cm out: past the upper arm plus forearm
    assert natural_ik(*PAN_AXIS_XY, z, tilt, roll) is None  # over the pan axis
    assert natural_ik(x, -y, z, tilt, roll) is None  # behind the base: facing out, the pan cannot turn that far
    assert natural_ik(x, y, z, math.pi / 2 + 0.3, roll) is None  # jaws pointing back at the base: Wrist_Pitch
    assert natural_ik(x, y, z, tilt, 3.0) is None  # past Wrist_Roll
    assert natural_ik(x, y, z, tilt, roll, margin=0.0) is not None
    assert natural_ik(x, y, z, tilt, roll, margin=0.2) is None  # home's Wrist_Pitch is 0.14 rad inside its limit
    # Straight up from the shoulder with the jaws horizontal puts the wrist 5 mm short of full stretch: reachable,
    # but not with a 2 cm stretch margin.
    stretched = (*xy_from_pan_reach(-math.pi / 2, 0.2), 0.387, -math.pi / 2, 0.0)
    assert natural_ik(*stretched) is not None
    assert natural_ik(*stretched, stretch_margin=0.02) is None
    # Jaws horizontal 5 cm above the base origin: the wrist-pitch axis is at the same height, below a 6 cm floor.
    low = (*xy_from_pan_reach(-math.pi / 2, 0.35), 0.05, -math.pi / 2, 0.0)
    assert natural_ik(*low) is not None
    assert natural_ik(*low, wrist_floor=0.06) is None
    assert natural_ik(*low[:3], -1.0, 0.0, wrist_floor=0.06) is not None  # leaning down lifts the wrist
