"""SO-101 follower embodiments for Isaac Lab Arena, built on the Isaac Lab configs in ``arena_so101.assets``.

Cameras are Python ``CameraCfg`` sensors: wrist RGB on ``Robot/gripper/gripper_cam``, plus an
``external_camera`` on the base link (over-shoulder / table view), so it moves with the robot.
Neither is baked into the USD.
"""

from __future__ import annotations

import math

import torch
import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg, EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose, PosePerEnv

from arena_so101.assets import SO101_CFG, SO101_HIGH_PD_CFG, SO101_WRIST_CAMERA_CFG
from arena_so101.cameras import look_at_offset
from arena_so101.constants import ARM_JOINT_NAMES, JAW_CLOSE_RAD, JAW_OPEN_RAD, SIM_JOINT_NAMES, TCP_OFFSET

# The arm faces +X only because SO101_CFG.init_state yaws its base 90° (see assets.py). Arena writes the
# Pose it is given straight into init_state and the root-pose reset event, so a plain Pose() would drop
# that yaw. The embodiment therefore takes every pose, bounding box and mesh in the *placement* frame,
# in which the arm faces +X, and composes the base yaw in on the way to the sim.
_PLACEMENT_TO_BASE = Pose(rotation_xyzw=SO101_CFG.init_state.rot)
_BASE_TO_PLACEMENT = Pose(rotation_xyzw=(*(-c for c in SO101_CFG.init_state.rot[:3]), SO101_CFG.init_state.rot[3]))


@configclass
class SO101SceneCfg:
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # EE frame at the TCP, between the jaw tips: reach/place rewards and Arena's gripper read it.
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="end_effector",
                offset=OffsetCfg(pos=TCP_OFFSET),
            ),
        ],
    )


@configclass
class SO101AbsJointActionsCfg:
    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SIM_JOINT_NAMES),
        scale=1,
        use_default_offset=False,
    )


@configclass
class SO101RelJointActionsCfg:
    arm_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SIM_JOINT_NAMES),
        scale=0.05,
        use_zero_offset=True,
    )


@configclass
class SO101IKActionsCfg:
    """Relative SE(3) arm + binary Jaw — matches keyboard / gamepad / spacemouse teleop."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name="gripper",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        # Commands move the TCP, so rotations pivot about the jaw tips rather than the wrist.
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=TCP_OFFSET),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Jaw"],
        open_command_expr={"Jaw": JAW_OPEN_RAD},
        close_command_expr={"Jaw": JAW_CLOSE_RAD},
    )


@configclass
class SO101ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SO101EventCfg:
    # Isaac Lab's scene reset leaves joint state alone, so without this the arm starts each
    # episode where the last one ended. Resets to init_state.joint_pos, clamped to the soft limits.
    reset_robot_joints: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),  # set from reset_joint_noise
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class SO101CameraCfg(ArenaCameraCfg):
    camera_ego: CameraCfg = SO101_WRIST_CAMERA_CFG

    # Third-person / over-shoulder view on the base link, so it follows the robot. The offset is in the
    # base link's frame, where the arm points along -Y; SO101EmbodimentBase.set_external_camera_view takes
    # the view with the arm facing +X instead (this default is eye (0.55, -0.6, 0.45), target (0.12, 0, 0.1) there).
    external_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/external_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            f_stop=2000.0,
            focal_length=30.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 2.0),
        ),
        offset=look_at_offset(eye=(-0.6, -0.55, 0.45), target=(0.0, -0.12, 0.1)),
    )


class SO101EmbodimentBase(EmbodimentBase):
    """Shared SO-101 follower setup (workshop USD).

    Every episode reset returns the arm to ``init_state.joint_pos`` (the home pose unless changed
    with ``set_joint_initial_pos``), plus uniform noise of ``±reset_joint_noise`` rad on each joint.

    Poses, bounding boxes and meshes are in the placement frame, where the arm faces +X: ``Pose()``
    keeps it facing that way, and relations such as ``On(table)`` place it facing that way.
    """

    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        reset_joint_noise: float = 0.0,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = SO101SceneCfg()
        self.event_config = SO101EventCfg()
        self.event_config.reset_robot_joints.params["position_range"] = (-reset_joint_noise, reset_joint_noise)
        self.camera_config = SO101CameraCfg()
        # TiledCameraCfg is deprecated (isaaclab 4.6, Isaac Lab 3.0): CameraCfg renders tiled itself. Keep
        # Arena from converting the rig back to it.
        self.camera_config.set_use_tiled_camera(False)
        self.add_camera_variations(self.camera_config)
        self.observation_config = SO101ObservationsCfg()
        if concatenate_observation_terms:
            self.observation_config.policy.concatenate_terms = True
        self.action_config = None

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        # Scene entity name: Arena tasks look it up with SceneEntityCfg / env.scene[...].
        return "ee_frame"

    def get_command_body_name(self) -> str:
        return "gripper"

    def get_reach_body_name(self) -> str:
        """Rigid body used for reach rewards (workshop USD link name)."""
        return "gripper"

    def set_external_camera_view(self, eye: tuple[float, float, float], target: tuple[float, float, float]) -> None:
        """Aim ``external_camera`` from ``eye`` at ``target``, both relative to the robot base with the arm
        facing +X (the placement frame). The camera is attached to the base link and moves with the robot."""
        yaw = torch.tensor(_PLACEMENT_TO_BASE.rotation_xyzw)
        eye, target = (tuple(quat_apply_inverse(yaw, torch.tensor(p, dtype=torch.float32)).tolist()) for p in (eye, target))
        self.camera_config.external_camera.offset = look_at_offset(eye, target)

    # Placement frame (see _PLACEMENT_TO_BASE): the base yaw is composed in at the two places Arena
    # writes a pose to the sim, and taken out of what it reads back from the scene config.
    def get_initial_pose(self) -> Pose | PosePerEnv:
        if self.initial_pose is not None:
            return self.initial_pose
        return super().get_initial_pose().multiply(_BASE_TO_PLACEMENT)  # the base class reads init_state

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config, pose: Pose):
        return super()._update_scene_cfg_with_robot_initial_pose(scene_config, pose.multiply(_PLACEMENT_TO_BASE))

    def layout_pose_to_scene_writes(self, layout_pose: Pose) -> list[tuple[str, Pose]]:
        # Feeds the root-pose reset event and the relation solver's runtime placement.
        return super().layout_pose_to_scene_writes(layout_pose.multiply(_PLACEMENT_TO_BASE))

    def get_bounding_box(self, prim_path: str | None = None):
        return super().get_bounding_box(prim_path).rotated_90_around_z(1)  # +90°: the base yaw

    def get_collision_mesh(self):
        import trimesh

        mesh = super().get_collision_mesh()
        return None if mesh is None else mesh.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (0, 0, 1)))


@register_asset
class SO101AbsJointEmbodiment(SO101EmbodimentBase):
    """Absolute joint positions — preferred for SO-101 leader / joint gamepad teleop."""

    name = "so101_abs_joint"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_config = SO101AbsJointActionsCfg()


@register_asset
class SO101RelJointEmbodiment(SO101EmbodimentBase):
    """Relative joint position actions."""

    name = "so101_rel_joint"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_config = SO101RelJointActionsCfg()


@register_asset
class SO101IKEmbodiment(SO101EmbodimentBase):
    """Differential IK (relative SE(3)) arm control + binary Jaw gripper."""

    name = "so101_ik"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene_config.robot = SO101_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config = SO101IKActionsCfg()

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name
