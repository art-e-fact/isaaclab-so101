"""SO-101 in Isaac Lab's reach task: move the gripper to a random point above the table.

Isaac Lab's CLI imports this module through the ``isaaclab.tasks`` entry point in pyproject.toml, so its
own train and play commands find the task:

    uv run isaaclab train --task SO101-Reach
    uv run isaaclab play --task SO101-Reach --viz kit

This is how Isaac Lab adds a robot to one of its tasks (compare ``isaaclab_tasks.core.reach.config.franka``):
subclass the task config, then swap in the robot, its end-effector link and its action.
"""

from __future__ import annotations

import gymnasium as gym

import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_tasks.core.reach.config.franka.agents.rsl_rl_ppo_cfg import FrankaReachPPORunnerCfg
from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg

from arena_so101 import ARM_JOINT_NAMES
from arena_so101.assets import SO101_CFG


@configclass
class SO101ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # PhysX: Kit-less OvPhysX when nothing needs Kit, Isaac Sim's PhysX with --viz kit.
        # The SO-101 USD doesn't run on the task's default, Newton, yet.
        self.sim.physics.default = self.sim.physics.physx
        # Frame the small arm: the task's camera and spacing are sized for Franka (its table is 1.3 m wide).
        self.scene.env_spacing = 1.5
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.6, -1.6, 1.2), lookat=(0.0, 0.0, 0.0))

        # The arm acts on its five arm joints; the jaw holds its home pose.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=list(ARM_JOINT_NAMES), scale=0.5, use_default_offset=True
        )

        # Track the gripper link. A 5-DOF arm can't reach arbitrary orientations, so the goal is a position.
        self.commands.ee_pose.body_name = "gripper"
        self.commands.ee_pose.orientation_success_threshold = None
        self.commands.ee_pose.position_success_threshold = 0.02  # for the logged success rate; Franka's is 5 cm
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["gripper"]
        self.rewards.end_effector_orientation_tracking = None

        # Follow the goal for the whole episode (it moves every 4 s), as Isaac Lab's reach task did before 3.0.0rc1.
        # The rc1 task ends the episode on the first touch instead, and a position-only goal turns that into
        # swinging through it. Without the touch bonus, rc1's action-size penalty holds the arm back near home;
        # the older fine-grained term pulls it the last few centimetres.
        self.terminations.success = None
        self.rewards.success = None
        self.rewards.action_magnitude = None
        self.rewards.end_effector_position_tracking_fine_grained = RewTerm(
            func=mdp.position_command_error_tanh,
            weight=0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["gripper"]), "std": 0.05, "command_name": "ee_pose"},
        )
        # Goals are sampled in the robot base frame, which SO101_CFG yaws 90°: the arm reaches along base -Y.
        # This box is 10-30 cm above the table, 15-30 cm in front of the base, 10 cm to either side.
        ranges = self.commands.ee_pose.ranges
        ranges.pos_x = (-0.10, 0.10)
        ranges.pos_y = (-0.30, -0.15)
        ranges.pos_z = (0.10, 0.30)
        ranges.roll = ranges.pitch = ranges.yaw = (0.0, 0.0)


@configclass
class SO101ReachPPORunnerCfg(FrankaReachPPORunnerCfg):
    experiment_name = "reach_so101"
    max_iterations = 200  # tracking is best here (~1.3 cm); it slowly gets worse with longer training


gym.register(
    id="SO101-Reach",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:SO101ReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}:SO101ReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
