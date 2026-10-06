"""The SO-101's natural end-effector coordinates: ``(x, y, z, tilt, roll)`` → a pose the 5-DoF arm can hold.

Pure Python (``math`` only). Positions are in the robot base frame (the ``base`` link; forward is -Y, as
``eef_pos`` and the IK action use it), quaternions are (x, y, z, w) as in this Isaac Lab.

A 5-DoF arm cannot point its gripper any way it likes from any position. The shoulder pan turns a vertical plane,
and everything from the shoulder to the jaws stays in it: the gripper's axis lies in that plane, pointing out
along the arm. So once the TCP is at ``(x, y)``, its yaw is spoken for — it faces away from the pan axis — and
what remains free is how far the gripper leans within the plane (``tilt``) and how it spins about its own axis
(``roll``). That is ``R = Rz(azimuth) · Ry(tilt) · Rz(roll)``, a ZYZ Euler sequence with a closed-form quaternion.

Ported from arena-shape-sorting's ``so101_ee_pose_xyzw`` (its cuRobo policy builds grasp and insert poses with
it), with the pivot defaulting to the real pan axis. The model is exact up to the TCP's 7 mm offset from the
roll axis (``TCP_OFFSET[0]``), which swings out of the arm plane with the roll: under 3° of yaw beyond 15 cm
from the pan axis, which the differential IK absorbs.
"""

from __future__ import annotations

import math

from arena_so101.constants import PAN_AXIS_XY

# The TCP at HOME_JOINT_POS in these coordinates: forward kinematics of the shipped URDF (tests/test_ee_pose.py
# recomputes it). The gripper leans 42° forward from straight down; the jaws open across the arm (roll ≈ -90°,
# Wrist_Roll's home angle).
HOME_NATURAL_POSE = (0.0815, -0.2140, 0.2266, -0.7411, -1.6285)


def natural_ee_quat_xyzw(
    x: float, y: float, tilt: float = 0.0, roll: float = 0.0, *, pivot_xy: tuple[float, float] = PAN_AXIS_XY
) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) of a TCP at ``(x, y)`` that the arm can hold: facing along the arm.

    At ``tilt = roll = 0`` the tool frame has X pointing away from the pan axis, Y to its left and Z up, so the
    jaws point straight down and open along the arm. ``tilt`` leans the gripper within the arm plane (negative:
    the fingertips swing away from the base), ``roll`` spins it about its own axis (positive: counter-clockwise
    seen from above when the jaws point down). The azimuth is measured about ``pivot_xy``; right over it, it is 0.
    """
    dx, dy = x - pivot_xy[0], y - pivot_xy[1]
    psi = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-8 else 0.0
    # qz(psi) ⊗ qy(tilt) ⊗ qz(roll), multiplied out.
    ct, st = math.cos(tilt / 2), math.sin(tilt / 2)
    return (
        -st * math.sin((psi - roll) / 2),
        st * math.cos((psi - roll) / 2),
        ct * math.sin((psi + roll) / 2),
        ct * math.cos((psi + roll) / 2),
    )
