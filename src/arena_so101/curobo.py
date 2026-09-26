"""Load the shipped cuRobo robot config (``so101.yml``) with its file paths resolved."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from arena_so101.constants import CUROBO_ROBOT_YML


def robot_cfg(path: str | Path = CUROBO_ROBOT_YML) -> dict[str, Any]:
    """The robot YAML as a dict for ``MotionPlannerCfg.create(robot=...)`` or ``RobotCfg.from_dict``.

    ``urdf_path`` and ``asset_root_path`` are stored relative to the YAML (so the package relocates) and made
    absolute here; cuRobo would look for a relative path under its own assets directory.
    """
    import yaml

    path = Path(path).expanduser().resolve()
    with path.open() as f:
        data = yaml.safe_load(f)
    kinematics = data["robot_cfg"]["kinematics"]
    for key in ("urdf_path", "asset_root_path"):
        if kinematics.get(key):
            kinematics[key] = str((path.parent / kinematics[key]).resolve())
    return data
