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

# cuRobo robot config written by ``python -m arena_so101.generate_curobo_config``.
# Not shipped: this file exists only after running the generator.
CUROBO_ROBOT_YML = _DATA_DIR / "curobo" / "so101.yml"
