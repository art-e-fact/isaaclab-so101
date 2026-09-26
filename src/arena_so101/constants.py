"""SO-101 constants for the NVIDIA Sim-to-Real workshop USD.

Pure Python: safe to import without Isaac Sim, Isaac Lab, or torch.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import MappingProxyType

# Sim USD joint order (must match ArticulationCfg / action cfg).
SIM_JOINT_NAMES = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)

# Arm joints only (Jaw is a separate binary gripper term for IK).
ARM_JOINT_NAMES = SIM_JOINT_NAMES[:5]

# USD joint limits in degrees (from the NVIDIA SO-101 workshop), in SIM_JOINT_NAMES order.
JOINT_LIMITS_DEG = (
    (-110.0, 110.0),  # Rotation
    (-100.0, 100.0),  # Pitch
    (-100.0, 90.0),  # Elbow
    (-95.0, 95.0),  # Wrist_Pitch
    (-160.0, 160.0),  # Wrist_Roll
    (-10.0, 100.0),  # Jaw
)

JOINT_LIMITS_RAD = tuple((math.radians(lo), math.radians(hi)) for lo, hi in JOINT_LIMITS_DEG)

# Jaw targets for the binary gripper: the USD Jaw limits.
JAW_CLOSE_RAD, JAW_OPEN_RAD = JOINT_LIMITS_RAD[-1]

# Tool center point: between the jaw tips with the jaw closed, in the ``gripper`` link frame (metres), whose
# -Z axis runs along the jaws. Measured from the USD meshes: fixed tip (-0.010, 0.000, -0.103), moving tip at
# Jaw = -10° (-0.004, 0.000, -0.101). The IK action, ``ee_frame`` and the cuRobo tool frame all target it.
TCP_OFFSET = (-0.007, 0.0, -0.102)

# Jaw geometry for the gap between the tips: the moving tip at Jaw = 0 and the fixed tip, relative to the Jaw
# pivot in the ``gripper`` frame's XZ plane (the jaw swings about the pivot's -Y axis). Measured from the USD.
JAW_TIP_XZ = (-0.0100, -0.0810)
FIXED_JAW_TIP_XZ = (-0.0304, -0.0799)

# Default / home joint pose in radians (ArticulationCfg init_state, gamepad reset, cuRobo retract).
# Read-only; copy with dict(HOME_JOINT_POS) before modifying.
HOME_JOINT_POS = MappingProxyType(
    {
        "Rotation": -0.2736,
        "Pitch": -0.6109,
        "Elbow": -0.0745,
        "Wrist_Pitch": 1.5148,
        "Wrist_Roll": -1.6034,
        "Jaw": -0.1465,
    }
)

_DATA_DIR = Path(__file__).resolve().parent / "embodiments" / "data"

# Robot USD shipped with the package.
USD_PATH = _DATA_DIR / "SO-ARM101-USD.usd"

# cuRobo robot config, shipped (regenerate with ``python -m arena_so101.generate_curobo_config``). Its
# ``urdf_path`` is relative to the file: load it with ``arena_so101.curobo.robot_cfg``.
CUROBO_ROBOT_YML = _DATA_DIR / "curobo" / "so101.yml"
