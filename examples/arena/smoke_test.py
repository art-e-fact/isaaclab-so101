"""Headless smoke test for the SO-101 embodiments in Arena.

For each embodiment, builds so101_table (Arena's lift task, whose reward reads the ee_frame),
steps it, moves the arm away, resets, and checks the arm is back in its initial pose. Then checks
that placing the arm, explicitly and with ``On(table)``, keeps it facing +X, that the cameras
render and the external camera follows the base, the Franka-parity extras (EE observations,
the Arena gripper, ``set_joint_initial_pos``), and that the LeRobot recorder takes the env's own frames and
``joint_targets`` for every embodiment.

    source ./setup.sh && python smoke_test.py

Exits non-zero on the first failure.
"""

from __future__ import annotations

import time
from pathlib import Path

EMBODIMENTS = ("so101_abs_joint", "so101_rel_joint", "so101_ik")
STEPS = 30
TOLERANCE_RAD = 1e-3
TOLERANCE_M = 1e-3


def build(args, cfg=None, prepare=None):
    """Build so101_table into a gym env (returned with the Arena env); ``prepare(arena_env)`` runs first."""
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from so101_table import SO101TableEnvironment, SO101TableEnvironmentCfg

    arena_env = SO101TableEnvironment().build(cfg or SO101TableEnvironmentCfg())
    if prepare is not None:
        prepare(arena_env)
    return ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args)).make_registered(), arena_env


def check(embodiment_name: str, args) -> None:
    import numpy as np
    import torch
    from isaaclab.utils.math import quat_apply

    from arena_so101 import JAW_OPEN_RAD, TCP_OFFSET
    from arena_so101.lerobot import joint_targets
    from so101_table import SO101TableEnvironmentCfg

    env, _ = build(args, SO101TableEnvironmentCfg(embodiment=embodiment_name))
    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        initial = robot.data.default_joint_pos.clone()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        if embodiment_name == "so101_abs_joint":
            actions[:] = initial  # absolute targets: hold the initial pose
        for _ in range(STEPS):
            env.step(actions)

        # ee_frame reports the TCP: the gripper link's pose composed with TCP_OFFSET.
        gripper = robot.find_bodies("gripper")[0][0]
        body_pos, body_quat = robot.data.body_pos_w.torch[0, gripper], robot.data.body_quat_w.torch[0, gripper]
        tcp = env.unwrapped.scene["ee_frame"].data.target_pos_w.torch[0, 0]
        expected = body_pos + quat_apply(body_quat, body_pos.new_tensor(TCP_OFFSET))
        assert (tcp - expected).norm() < TOLERANCE_M, f"{embodiment_name}: ee_frame at {tcp.tolist()}, TCP {expected.tolist()}"

        # joint_targets: the absolute targets the sim received, whatever the action space (the recorder's action).
        targets = joint_targets(env)[0]
        joint_pos = robot.data.joint_pos.torch[0].cpu().numpy()
        expected = initial[0].cpu().numpy() if embodiment_name == "so101_abs_joint" else joint_pos  # zero deltas hold
        if embodiment_name == "so101_ik":
            expected = np.append(joint_pos[:5], JAW_OPEN_RAD)  # a 0 jaw command opens
        assert np.allclose(targets, expected, atol=1e-2), f"{embodiment_name}: joint_targets {targets}, expected {expected}"

        # Move the arm away, then reset: the embodiment's reset event must bring it back.
        robot.write_joint_state_to_sim(initial + 0.3, torch.zeros_like(initial))
        env.reset()
        error = (robot.data.joint_pos - initial).abs().max().item()
        assert error < TOLERANCE_RAD, f"{embodiment_name}: arm is {error:.3f} rad from its initial pose after reset"
        print(f"[smoke_test] {embodiment_name}: OK ({STEPS} steps, ee_frame at the TCP, reset error {error:.1e} rad)")
    finally:
        env.close()


def check_placement(args) -> None:
    """A Pose with the default rotation and an On(table) relation both leave the arm facing +X."""
    import torch
    from isaaclab_arena.relations.relations import IsAnchor, On
    from isaaclab_arena.utils.pose import Pose

    from arena_so101.assets import SO101_CFG

    def on_table(arena_env) -> None:
        table = arena_env.scene.assets["maple_table_robolab"]
        table.add_relation(IsAnchor())  # the solver needs one fixed reference
        arena_env.embodiment.add_relation(On(table))

    facing_x = torch.tensor(SO101_CFG.init_state.rot)  # the base yaw that makes the arm face +X (x, y, z, w)
    cases = {
        "Pose": lambda arena_env: arena_env.embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.1, 0.0))),
        "On(table)": on_table,
    }
    for label, prepare in cases.items():
        started = time.perf_counter()
        env, arena_env = build(args, prepare=prepare)
        try:
            build_s = time.perf_counter() - started
            env.reset()  # runs the root-pose reset event / the relation placement event
            robot = env.unwrapped.scene["robot"]
            pos = (robot.data.root_pos_w.torch - env.unwrapped.scene.env_origins)[0]
            quat = robot.data.root_quat_w.torch[0]
            facing_error = 1.0 - torch.dot(quat, facing_x.to(quat)).abs().item()  # 0 when equal up to sign
            assert facing_error < 1e-4, f"{label}: base rotation {quat.tolist()} does not face +X {facing_x.tolist()}"
            if label == "Pose":
                assert (pos - pos.new_tensor((0.0, 0.1, 0.0))).abs().max() < TOLERANCE_M, f"{label}: base at {pos.tolist()}"
            else:
                # On() puts the arm's bounding box 1 cm (its default clearance) above the table top, inside its footprint.
                table_box = arena_env.scene.assets["maple_table_robolab"].get_world_bounding_box()
                table_top = table_box.max_point[..., 2].item()
                bottom = arena_env.embodiment.get_bounding_box().min_point[..., 2].item()
                assert abs(pos[2].item() + bottom - table_top - 0.01) < 5e-3, f"{label}: base at {pos.tolist()}, top {table_top}"
                inside = (table_box.min_point[..., :2] < pos[:2].cpu()).all() and (pos[:2].cpu() < table_box.max_point[..., :2]).all()
                assert inside, f"{label}: base at {pos.tolist()} is off the table {table_box}"
            print(
                f"[smoke_test] placement {label}: OK (base at {[round(v, 3) for v in pos.tolist()]}, "
                f"rotation {[round(v, 3) for v in quat.tolist()]}, built in {build_s:.0f} s)"
            )
        finally:
            env.close()


def check_cameras(args) -> None:
    """Both cameras render, and the external camera rides on the base, aimed by set_external_camera_view."""
    import torch
    from isaaclab.utils.math import quat_apply
    from isaaclab_arena.utils.pose import Pose

    from so101_table import SO101TableEnvironmentCfg

    eye, target = (0.5, -0.5, 0.4), (0.15, 0.0, 0.05)  # relative to the base, arm facing +X

    def prepare(arena_env) -> None:
        arena_env.embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.1, 0.0)))  # the camera has to follow
        arena_env.embodiment.set_external_camera_view(eye, target)

    env, _ = build(args, SO101TableEnvironmentCfg(embodiment="so101_rel_joint", enable_cameras=True), prepare)
    try:
        env.reset()
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)  # relative: hold still
        for _ in range(3):
            obs, *_ = env.step(actions)
        for key in ("camera_ego_rgb", "external_camera_rgb"):
            image = obs["camera_obs"][key]
            assert tuple(image.shape) == (1, 480, 640, 3), f"{key}: shape {tuple(image.shape)}"
            assert image.float().std() > 1.0, f"{key}: blank image"
        robot, camera = env.unwrapped.scene["robot"], env.unwrapped.scene["external_camera"]
        base = robot.data.root_pos_w.torch[0]
        eye_w, target_w = base + base.new_tensor(eye), base + base.new_tensor(target)  # the base faces +X
        camera_pos = camera.data.pos_w.torch[0]
        assert (camera_pos - eye_w).norm() < TOLERANCE_M, f"external camera at {camera_pos.tolist()}, not {eye_w.tolist()}"
        forward = quat_apply(camera.data.quat_w_world.torch[0], base.new_tensor((1.0, 0.0, 0.0)))  # world: +X forward
        aim = target_w - eye_w
        cosine = torch.dot(forward, aim / aim.norm()).item()
        assert cosine > 0.9999, f"external camera looks along {forward.tolist()}, not at the target (cos {cosine:.4f})"
        offset = [round(v, 3) for v in (camera_pos - base).tolist()]
        print(f"[smoke_test] cameras: OK (both render; external camera {offset} from the base, aimed at the target)")
    finally:
        env.close()


def check_parity(args) -> None:
    """Franka parity: EE observations in the base frame, the Arena gripper's jaw gap, set_joint_initial_pos."""
    import torch
    from isaaclab_arena.environments.arena_world import ArenaWorld

    from arena_so101 import JAW_OPEN_RAD
    from so101_table import SO101TableEnvironmentCfg

    env, arena_env = build(
        args,
        SO101TableEnvironmentCfg(embodiment="so101_ik"),
        prepare=lambda arena_env: arena_env.embodiment.set_joint_initial_pos({"Jaw": 0.5}),
    )
    try:
        obs, _ = env.reset()
        robot, ee = env.unwrapped.scene["robot"], env.unwrapped.scene["ee_frame"].data
        jaw = robot.find_joints("Jaw")[0][0]
        jaw_pos = robot.data.joint_pos.torch[0, jaw].item()
        assert abs(jaw_pos - 0.5) < TOLERANCE_RAD, f"set_joint_initial_pos: Jaw at {jaw_pos:.3f} rad after reset, not 0.5"
        policy = obs["policy"]
        assert torch.allclose(policy["eef_pos"][0], ee.target_pos_source.torch[0, 0], atol=TOLERANCE_M), "eef_pos"
        assert torch.allclose(policy["eef_quat"][0], ee.target_quat_source.torch[0, 0], atol=1e-4), "eef_quat"
        assert abs(policy["gripper_pos"][0, 0].item() - jaw_pos) < 1e-6, "gripper_pos is not the Jaw angle"

        world = ArenaWorld(env.unwrapped.scene)
        gripper = arena_env.embodiment.get_gripper()
        assert torch.allclose(gripper.get_position_w(world)[0], ee.target_pos_w.torch[0, 0], atol=TOLERANCE_M)
        gap_half = gripper.get_jaw_gap_m(world)[0].item()  # Jaw at 0.5 rad
        joint_pos = robot.data.joint_pos.torch.clone()
        joint_pos[:, jaw] = JAW_OPEN_RAD
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
        gap_open = gripper.get_jaw_gap_m(world)[0].item()
        assert gap_half < gap_open and abs(gap_open - 0.140) < 5e-3, f"jaw gap {gap_half:.3f} m at 0.5 rad, {gap_open:.3f} m open"
        print(f"[smoke_test] parity: OK (eef obs match ee_frame; jaw gap {gap_half * 1e3:.0f} mm at 0.5 rad, {gap_open * 1e3:.0f} mm open)")
    finally:
        env.close()


def check_recorder(args) -> None:
    """The recorder takes the env's own frames for any camera set, with joint_targets as the action."""
    import contextlib
    import sys
    import tempfile
    import types

    import numpy as np
    import torch

    from arena_so101 import SIM_JOINT_NAMES
    from arena_so101.lerobot import SO101LeRobotRecorder, camera_shapes, joint_targets
    from so101_table import SO101TableEnvironmentCfg

    # lerobot is not in Arena's venv: stand in for LeRobotDataset and keep the frames the recorder adds.
    frames: list[dict] = []
    dataset = types.SimpleNamespace(
        add_frame=frames.append, has_pending_frames=lambda: bool(frames), save_episode=frames.clear, num_episodes=0
    )
    sys.modules["lerobot.datasets"] = types.SimpleNamespace(
        LeRobotDataset=types.SimpleNamespace(create=lambda *_, **__: dataset),
        VideoEncodingManager=lambda _: contextlib.nullcontext(),
    )
    env, _ = build(args, SO101TableEnvironmentCfg(embodiment="so101_ik", enable_cameras=True))
    try:
        obs, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        assert robot.joint_names == list(SIM_JOINT_NAMES), robot.joint_names  # observation.state is in this order
        shapes = camera_shapes(env)
        assert shapes == {"camera_ego_rgb": (480, 640, 3), "external_camera_rgb": (480, 640, 3)}, shapes
        cameras = {"camera_ego_rgb": "observation.images.wrist"}  # a one-camera set
        root = Path(tempfile.mkdtemp()) / "dataset"
        with SO101LeRobotRecorder(root=root, repo_id="smoke/so101", fps=30, cameras=cameras, image_shapes=shapes) as recorder:
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for _ in range(2):
                snapshot = recorder.snapshot_observation(obs)
                obs, *_ = env.step(actions)
                recorder.add_transition(snapshot, joint_targets(env), task="smoke")
            frame = frames[-1]
            assert set(frame) == {"observation.state", "action", "observation.images.wrist", "task"}, sorted(frame)
            image = frame["observation.images.wrist"]
            assert image.dtype == np.uint8 and image.shape == (480, 640, 3) and image.std() > 1.0, "wrist frame"
            assert frame["action"].dtype == np.float32 and frame["action"].shape == (6,)
            assert np.allclose(frame["action"], joint_targets(env)[0]), "action is not the step's joint targets"
            recorder.save_episode()
        print(f"[smoke_test] recorder: OK (2 frames, cameras {shapes}, one recorded)")
    finally:
        env.close()


def main() -> None:
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext, teardown_simulation_app

    args = get_isaaclab_arena_cli_parser().parse_args(["--enable_cameras"])  # no --viz: headless
    with SimulationAppContext(args):
        for embodiment_name in EMBODIMENTS:
            check(embodiment_name, args)
            teardown_simulation_app(make_new_stage=True)
        for check_fn in (check_placement, check_cameras, check_parity, check_recorder):
            check_fn(args)
            teardown_simulation_app(make_new_stage=True)
        print("[smoke_test] all checks OK")


if __name__ == "__main__":
    main()
