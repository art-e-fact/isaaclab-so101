"""Camera placement helpers. Imports Isaac Lab, so import after the simulation app starts."""

from __future__ import annotations

import torch
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix


def look_at_offset(eye: tuple[float, float, float], target: tuple[float, float, float]) -> CameraCfg.OffsetCfg:
    """A ``CameraCfg.OffsetCfg`` that puts the camera at ``eye`` looking at ``target``.

    Both points are in the camera's parent frame (the parent of ``CameraCfg.prim_path``). ``CameraCfg`` has
    no look-at of its own; this is ``Camera.set_world_poses_from_view`` as a config.
    """
    eyes, targets = torch.tensor([eye], dtype=torch.float32), torch.tensor([target], dtype=torch.float32)
    quat = quat_from_matrix(create_rotation_matrix_from_view(eyes, targets, up_axis="Z"))[0]
    return CameraCfg.OffsetCfg(pos=tuple(eye), rot=tuple(quat.tolist()), convention="opengl")
