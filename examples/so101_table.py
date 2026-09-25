"""SO-101 on a table with a cube to lift: a minimal Arena environment for the arena_so101 embodiments.

Arena has no plugin discovery, so its CLIs load this class by path:

    --external_environment_class_path so101_table:SO101TableEnvironment so101_table --embodiment so101_ik

The robot sits at the env origin facing +X; the cube is 25 cm in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

# Env frame, metres. The table top is at z=0.03, where the robot mesh bottom is; the cube is 4.8 cm.
TABLE_POS = (-0.236, 0.0, 0.027)
CUBE_POS = (0.25, 0.0, 0.055)


@dataclass
class SO101TableEnvironmentCfg(ArenaEnvironmentCfg):
    embodiment: str = "so101_abs_joint"
    """so101_abs_joint, so101_rel_joint or so101_ik."""
    teleop_device: str | None = None
    """so101_gamepad or so101_leader with so101_abs_joint; keyboard, spacemouse or so101_gamepad with so101_ik."""
    leader_port: str = "/dev/ttyACM0"
    """Serial port of the physical leader arm (so101_leader)."""
    enable_cameras: bool = False
    rl_training_mode: bool = False
    """True: never end an episode on success (for RL training)."""


class SO101TableEnvironment(ArenaEnvironmentFactory[SO101TableEnvironmentCfg]):
    name = "so101_table"
    _legacy_argparse_cfg_type = SO101TableEnvironmentCfg

    def build(self, cfg: SO101TableEnvironmentCfg):
        import arena_so101
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL
        from isaaclab_arena.utils.pose import Pose

        # Registers the so101_* embodiments and devices. It imports Isaac Lab, so it must run
        # after the simulation app starts, which is why it lives here and not at module level.
        arena_so101.register()

        table = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        table.set_initial_pose(Pose(position_xyz=TABLE_POS))
        cube = self.asset_registry.get_asset_by_name("dex_cube")()
        cube.set_initial_pose(Pose(position_xyz=CUBE_POS))
        light = self.asset_registry.get_asset_by_name("light")()

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)

        teleop_device = None
        if cfg.teleop_device is not None:
            kwargs = {"port": cfg.leader_port} if cfg.teleop_device == "so101_leader" else {}
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)(**kwargs)

        # Arena's lift task: lift the cube to a goal marker. Its reward reads the embodiment's ee_frame.
        task = LiftObjectTaskRL(
            cube, table, embodiment, minimum_height_to_lift=0.04, rl_training_mode=cfg.rl_training_mode
        )
        # The goal is sampled in the robot base frame, which the SO-101's init_state yaws 90° from
        # the env frame (env +X is base -Y). Arena's default ranges assume they coincide, as for
        # Franka. Put the goal 8-15 cm above the cube.
        ranges = task.commands_cfg.object_pose.ranges
        ranges.pos_x = (CUBE_POS[1] - 0.05, CUBE_POS[1] + 0.05)
        ranges.pos_y = (-CUBE_POS[0] - 0.05, -CUBE_POS[0] + 0.05)
        ranges.pos_z = (CUBE_POS[2] + 0.08, CUBE_POS[2] + 0.15)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[table, cube, light]),
            task=task,
            teleop_device=teleop_device,
        )
