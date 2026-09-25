"""Headless smoke test for the SO-101 embodiments in Arena.

For each embodiment, builds so101_table (Arena's lift task, whose reward reads the ee_frame),
steps it, moves the arm away, resets, and checks the arm is back in its initial pose.

    uv run python smoke_test.py

Exits non-zero on the first failure.
"""

from __future__ import annotations

EMBODIMENTS = ("so101_abs_joint", "so101_rel_joint", "so101_ik")
STEPS = 30
TOLERANCE_RAD = 1e-3


def check(embodiment_name: str, args) -> None:
    import torch
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from so101_table import SO101TableEnvironment, SO101TableEnvironmentCfg

    arena_env = SO101TableEnvironment().build(SO101TableEnvironmentCfg(embodiment=embodiment_name))
    env = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args)).make_registered()
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        initial = robot.data.default_joint_pos.clone()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        if embodiment_name == "so101_abs_joint":
            actions[:] = initial  # absolute targets: hold the initial pose
        for _ in range(STEPS):
            env.step(actions)

        # Move the arm away, then reset: the embodiment's reset event must bring it back.
        robot.write_joint_state_to_sim(initial + 0.3, torch.zeros_like(initial))
        env.reset()
        error = (robot.data.joint_pos - initial).abs().max().item()
        assert error < TOLERANCE_RAD, f"{embodiment_name}: arm is {error:.3f} rad from its initial pose after reset"
        print(f"[smoke_test] {embodiment_name}: OK ({STEPS} steps, reset error {error:.1e} rad)")
    finally:
        env.close()


def main() -> None:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app

    args = get_isaaclab_arena_cli_parser().parse_args([])  # no --viz: headless
    with SimulationAppContext(args):
        for embodiment_name in EMBODIMENTS:
            check(embodiment_name, args)
            teardown_simulation_app(make_new_stage=True)
        print("[smoke_test] all embodiments OK")


if __name__ == "__main__":
    main()
