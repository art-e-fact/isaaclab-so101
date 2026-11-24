"""SO-101 configs for Isaac Lab: the arm, its wrist camera and its colour. Imports Isaac Lab, not Arena.

Like ``isaaclab_assets``, import this after the simulation app starts::

    from isaaclab_so101.assets import SO101_CFG

    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

To paint the arm, add ``SO101_COLOR_CFG`` to the scene after the robot and use ``color_event`` or ``write_color``.

The arm faces +X: ``init_state`` yaws the USD base 90° about Z, so world +X is the base frame's -Y.
The workshop USD authors that yaw on its default prim; Isaac Lab's spawner replaces the prim's transform
with ``init_state``, so ``init_state.rot`` has to repeat it, or the arm faces -Y.
USD and joint names follow the NVIDIA Sim-to-Real SO-101 workshop (``Rotation`` … ``Jaw``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import VisualMaterialCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_so101.constants import HOME_JOINT_POS, PRINTED_MATERIAL, USD_PATH


def _quat_xyzw_from_euler_deg(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Intrinsic XYZ Euler (degrees) → quaternion (x, y, z, w)."""
    quat = quat_from_euler_xyz(
        torch.tensor(math.radians(roll)),
        torch.tensor(math.radians(pitch)),
        torch.tensor(math.radians(yaw)),
    )
    return tuple(quat.tolist())


# 90° yaw about Z (x, y, z, w).
_YAW_90 = _quat_xyzw_from_euler_deg(0.0, 0.0, 90.0)

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            # The USD has a world joint of its own (root_joint). Keep this True anyway: with None, PhysX
            # treats the base as floating, and robots outside env 0 blow up. Newton can't load the pair yet.
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=_YAW_90,
        joint_pos=dict(HOME_JOINT_POS),
    ),
    # Gear-aware gains from the NVIDIA workshop.
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=55, damping=0.7
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=30, damping=0.8
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=25, damping=0.7
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=12, damping=0.5
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=7, damping=0.5
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=4, damping=0.3
        ),
    },
)
"""The SO-101 with the workshop's gains. Set ``prim_path`` with ``.replace(prim_path=...)``."""

# Diff-IK assumes joint targets are tracked tightly; gravity + soft PD would make EE
# lag the SE(3) command. Gravity off removes that disturbance; high PD tracks targets.
SO101_HIGH_PD_CFG = SO101_CFG.copy()
SO101_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
SO101_HIGH_PD_CFG.actuators = {
    "rotation": ImplicitActuatorCfg(
        joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=400, damping=80
    ),
    "pitch": ImplicitActuatorCfg(
        joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=400, damping=80
    ),
    "elbow": ImplicitActuatorCfg(
        joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=400, damping=80
    ),
    "wrist_pitch": ImplicitActuatorCfg(
        joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=200, damping=40
    ),
    "wrist_roll": ImplicitActuatorCfg(
        joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=200, damping=40
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=20, damping=2
    ),
}
"""High-PD, gravity-off copy for differential IK (as ``FRANKA_PANDA_HIGH_PD_CFG``)."""

SO101_WRIST_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
    update_period=0.0,
    height=480,
    width=640,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        projection_type="pinhole",
        f_stop=100.0,
        focal_length=13.5,
        focus_distance=0.05,
    ),
    # Spawned at the gripper mount, offset into the real lens frame.
    offset=CameraCfg.OffsetCfg(
        pos=(-0.005, 0.06, -0.062),
        rot=_quat_xyzw_from_euler_deg(-45.0, 0.0, 0.0),
        convention="opengl",
    ),
)
"""The workshop's wrist camera. Its ``prim_path`` assumes the robot is at ``{ENV_REGEX_NS}/Robot``."""

SO101_COLOR_CFG = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Robot/" + PRINTED_MATERIAL, spawn=None)
"""The printed parts' material as a scene entity whose colour Isaac Lab writes at runtime, to Kit (RTX cameras and the
viewport) and to the Newton Warp renderer alike. Add it to the scene after the robot, which it is part of. Its
``prim_path`` assumes the robot is at ``{ENV_REGEX_NS}/Robot``.

Each env has its own copy, but Kit draws every env in env_0's colour unless the robot spawns with
``make_uninstanceable=True``: the USD binds the material from inside instanced prims, which all envs share. Write the
colour only through this entity, never through USD as well once the sim runs: Kit's writer would then index Fabric
prims that USD resynced, and crash the GPU."""


def linear_rgb(color: str | Sequence[float]) -> tuple[float, float, float]:
    """``color`` as materials hold it, in linear RGB. ``'#rrggbb'`` is sRGB, as a colour picker gives it, and is
    converted. ``(r, g, b)`` in [0, 1] is taken as linear already, as Isaac Lab's material configs take colours."""
    if isinstance(color, str):
        digits = color.removeprefix("#")
        assert len(digits) == 6, f"A colour string is '#rrggbb'; got {color!r}."
        srgb = (int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))
        return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)
    rgb = tuple(float(c) for c in color)
    assert len(rgb) == 3 and all(0.0 <= c <= 1.0 for c in rgb), f"A colour is (r, g, b) in [0, 1]; got {color!r}."
    return rgb


def color_event(colors: Sequence[str | Sequence[float]], mode: str = "reset", material: str = "robot_color") -> EventTermCfg:
    """An event term that paints the printed parts of each env it runs on a colour drawn from ``colors``, each as
    :func:`linear_rgb` reads it; one colour paints every env that colour. ``material`` names the scene's
    :data:`SO101_COLOR_CFG` entity. Mode ``"startup"`` paints once, ``"reset"`` at every reset."""
    choices = [linear_rgb(c) for c in colors]
    return EventTermCfg(
        func=mdp_isaac_lab.randomize_visual_material,
        mode=mode,
        params={"materials": SceneEntityCfg(material), "channels": {"color": {"choices": choices}}},
    )


def write_color(material, color: str | Sequence[float], env_ids: torch.Tensor | None = None) -> None:
    """Paint the printed parts of ``env_ids``, or of every env, ``color`` now. ``material`` is the scene's
    :data:`SO101_COLOR_CFG` entity, ``env.scene["robot_color"]``; it takes writes once ``sim.reset()`` has run."""
    n = material.num_instances if env_ids is None else len(env_ids)
    rgb = torch.tensor(linear_rgb(color), device=material.device).repeat(n, 1)
    material.write_channels([material], {"color": rgb[None]}, env_ids)
