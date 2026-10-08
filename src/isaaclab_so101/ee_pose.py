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

from isaaclab_so101.constants import JOINT_LIMITS_RAD, PAN_AXIS_XY

# The TCP at HOME_JOINT_POS in these coordinates: forward kinematics of the shipped URDF (tests/test_ee_pose.py
# recomputes it). The gripper leans 42° forward from straight down; the jaws open across the arm (roll ≈ -90°,
# Wrist_Roll's home angle).
HOME_NATURAL_POSE = (0.0815, -0.2140, 0.2266, -0.7411, -1.6285)


def natural_ee_quat_xyzw(
    x: float,
    y: float,
    tilt: float = 0.0,
    roll: float = 0.0,
    *,
    pivot_xy: tuple[float, float] = PAN_AXIS_XY,
    azimuth: float | None = None,
) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) of a TCP at ``(x, y)`` that the arm can hold: facing along the arm.

    At ``tilt = roll = 0`` the tool frame has X pointing away from the pan axis, Y to its left and Z up, so the
    jaws point straight down and open along the arm. ``tilt`` leans the gripper within the arm plane (negative:
    the fingertips swing away from the base), ``roll`` spins it about its own axis (positive: counter-clockwise
    seen from above when the jaws point down). The azimuth is measured about ``pivot_xy``; right over it, it is 0.
    A caller that tracks the azimuth itself can pass it as ``azimuth``: the same angle plus 2π gives the negated
    quaternion (the same rotation), so an integrated, unwrapped azimuth keeps a stream of quaternions continuous.
    """
    dx, dy = x - pivot_xy[0], y - pivot_xy[1]
    if azimuth is not None:
        psi = azimuth
    else:
        psi = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-8 else 0.0
    # qz(psi) ⊗ qy(tilt) ⊗ qz(roll), multiplied out.
    ct, st = math.cos(tilt / 2), math.sin(tilt / 2)
    return (
        -st * math.sin((psi - roll) / 2),
        st * math.cos((psi - roll) / 2),
        ct * math.sin((psi + roll) / 2),
        ct * math.cos((psi + roll) / 2),
    )


def pan_reach_from_xy(x: float, y: float, *, pivot_xy: tuple[float, float] = PAN_AXIS_XY) -> tuple[float, float]:
    """``(pan, reach)``: the azimuth about the pan axis (forward, -Y, is -π/2; left is more) and the distance from it."""
    dx, dy = x - pivot_xy[0], y - pivot_xy[1]
    return math.atan2(dy, dx), math.hypot(dx, dy)


def xy_from_pan_reach(pan: float, reach: float, *, pivot_xy: tuple[float, float] = PAN_AXIS_XY) -> tuple[float, float]:
    return pivot_xy[0] + reach * math.cos(pan), pivot_xy[1] + reach * math.sin(pan)


# The arm after the pan, as a planar chain in its vertical plane, from the URDF (tests/test_ee_pose.py checks it
# against the URDF's forward kinematics). Coordinates are (ρ, z): out from the pan axis, up. The shoulder pitch
# axis is at _SHOULDER; the upper arm and forearm follow, each with the direction it has at zero joint angles;
# every pitch joint turns the links the same way, so a link's direction is its zero direction minus the joint
# angles before it. The jaws point out along +ρ at zero angles, and the TCP is _WRIST_TO_TCP from the wrist-pitch
# axis along them (0.0611 to the roll axis, 0.102 on to the TCP), 7 mm off that line, which is ignored here as in
# natural_ee_quat_xyzw. The pan axis points down, so +Rotation turns the arm clockwise seen from above.
_SHOULDER = (0.0304, 0.1491)
_L1, _A1 = 0.1160, math.radians(76.03)
_L2, _A2 = 0.1350, math.radians(2.21)
_WRIST_TO_TCP = 0.1631
_REACH_SLACK = 1e-3  # the model is good to 0.2 mm; a wrist this far past full stretch is treated as at it


def _wrap(angle: float) -> float:
    return math.remainder(angle, math.tau)


def natural_ik(
    x: float,
    y: float,
    z: float,
    tilt: float,
    roll: float,
    *,
    margin: float = 0.0,
    stretch_margin: float = 0.0,
    wrist_floor: float = -math.inf,
) -> tuple[float, float, float, float, float] | None:
    """Joint angles ``(Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)`` holding the TCP at this natural pose, or
    ``None`` when the arm cannot while facing out from the pan axis as the natural pose has it: the fingertips
    right over the pan axis, the wrist out of reach (or within ``stretch_margin`` metres of full stretch), the
    wrist-pitch axis below ``wrist_floor`` (the wrist and gripper housings hang below it), or a joint past its
    limit less ``margin``. Only the home elbow bend (elbow up) is used: the hyperextended one spans 26° and
    dead-ends at the Elbow limit. (Folding the arm back over the base reaches some of these points facing the
    other way; that is not a natural pose.)
    """
    pan, reach = pan_reach_from_xy(x, y)
    if reach < 1e-6:
        return None
    # The wrist-pitch axis sits _WRIST_TO_TCP back from the TCP along the jaws, which point down at tilt 0 and
    # swing out (+ρ) as the tilt goes negative.
    w_r = reach - _SHOULDER[0] + _WRIST_TO_TCP * math.sin(tilt)
    w_z = z - _SHOULDER[1] + _WRIST_TO_TCP * math.cos(tilt)
    if w_z + _SHOULDER[1] < wrist_floor:
        return None
    r = math.hypot(w_r, w_z)
    if not abs(_L1 - _L2) - _REACH_SLACK <= r <= _L1 + _L2 + _REACH_SLACK - stretch_margin:
        return None
    r = min(max(r, abs(_L1 - _L2)), _L1 + _L2)
    alpha = math.acos(max(-1.0, min(1.0, (_L1 * _L1 + r * r - _L2 * _L2) / (2 * _L1 * r))))  # at the shoulder
    beta = math.acos(max(-1.0, min(1.0, (_L1 * _L1 + _L2 * _L2 - r * r) / (2 * _L1 * _L2))))  # inside the elbow
    theta1 = math.atan2(w_z, w_r) + alpha  # the upper arm, above the shoulder-to-wrist line: elbow up
    theta2 = theta1 - (math.pi - beta)
    q1 = _wrap(-(pan + math.pi / 2))
    q2 = _wrap(_A1 - theta1)
    q3 = _wrap(_A2 - theta2 - q2)
    q4 = _wrap(tilt + math.pi / 2 - q2 - q3)  # the jaws' direction is -(q2 + q3 + q4), and tilt = -(that + π/2)
    q = (q1, q2, q3, q4, _wrap(roll))
    if all(lo + margin <= v <= hi - margin for v, (lo, hi) in zip(q, JOINT_LIMITS_RAD)):
        return q
    return None
