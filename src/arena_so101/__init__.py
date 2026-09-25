"""SO-101 for Isaac Lab Arena.

Call :func:`register` after ``SimulationApp`` starts (imports Isaac Lab).
The constants below are pure Python and safe to import anywhere.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from arena_so101.constants import (
    ARM_JOINT_NAMES,
    CUROBO_ROBOT_YML,
    HOME_JOINT_POS,
    JAW_CLOSE_RAD,
    JAW_OPEN_RAD,
    JOINT_LIMITS_DEG,
    JOINT_LIMITS_RAD,
    SIM_JOINT_NAMES,
    USD_PATH,
)

try:
    __version__ = version("arena-so101")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "ARM_JOINT_NAMES",
    "CUROBO_ROBOT_YML",
    "HOME_JOINT_POS",
    "JAW_CLOSE_RAD",
    "JAW_OPEN_RAD",
    "JOINT_LIMITS_DEG",
    "JOINT_LIMITS_RAD",
    "SIM_JOINT_NAMES",
    "USD_PATH",
    "__version__",
    "register",
]

_REGISTERED = False


def register() -> None:
    """Import modules so ``@register_asset`` / device / retargeter decorators run."""
    global _REGISTERED
    if _REGISTERED:
        return
    from arena_so101 import devices as _devices  # noqa: F401
    from arena_so101 import retargeters as _retargeters  # noqa: F401
    from arena_so101.embodiments import so101 as _so101  # noqa: F401

    _REGISTERED = True
