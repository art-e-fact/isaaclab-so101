"""Spawn the SO-101 and sweep each joint in turn, with Isaac Lab's scene API: no managers, no Arena.

    uv run python sweep.py             # headless: one pass over the joints, then checks the arm settled home
    uv run python sweep.py --viz kit   # in the Isaac Sim window, until you close it
    uv run python sweep.py --out DIR   # headless, and leaves sweep.mp4, joint_tracking.png and sweep.json in DIR
"""

# ruff: noqa: E402  (Isaac Lab modules are imported after the app starts)

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep the SO-101's joints one at a time.")
parser.add_argument("--out", type=Path, help="Also record a camera video, a joint-angle plot and sweep.json here.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.out and args.visualizer:
    parser.error("--out records headless: drop --viz")
app = AppLauncher(args, enable_cameras=args.out is not None).app  # the camera renders offscreen for the video

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_so101 import SIM_JOINT_NAMES
from isaaclab_so101.assets import SO101_CFG
from isaaclab_so101.cameras import look_at_offset

AMPLITUDE_RAD = 0.5
SWEEP_S = 2.0  # per joint: one sine period, so each joint is home again when the next starts
SETTLE_S = 1.0
TOLERANCE_RAD = 0.05
VIDEO_FPS = 30
VIEW = {"eye": (0.7, -0.7, 0.5), "target": (0.1, 0.0, 0.15)}


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class VideoSceneCfg(SceneCfg):
    # The window's view as a camera sensor, so a headless run can record it.
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        offset=look_at_offset(**VIEW),
    )


def report(out: Path, angles: list, peak: list[float], error: float, dt: float, sweep_steps: int) -> None:
    """joint_tracking.png, target vs measured angle per joint, and sweep.json, what the asserts below check."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.array(angles)  # (steps, 2, joints): target, measured
    t = np.arange(len(data)) * dt
    fig, axes = plt.subplots(len(SIM_JOINT_NAMES), 1, figsize=(12, 12), sharex=True)
    joints = {}
    for i, (ax, name) in enumerate(zip(axes, SIM_JOINT_NAMES)):
        own = slice(i * sweep_steps, (i + 1) * sweep_steps)  # the steps on which this joint swept
        joints[name] = {"peak_rad": peak[i], "max_error_rad": float(np.abs(data[own, 0, i] - data[own, 1, i]).max())}
        ax.plot(t, data[:, 0, i], "--", label="target")
        ax.plot(t, data[:, 1, i], label="measured")
        ax.axvspan(t[own.start], t[own.stop - 1], color="k", alpha=0.05)
        ax.set_ylabel(f"{name} [rad]")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Each joint swept ±{AMPLITUDE_RAD} rad in turn with the workshop gains, then the arm settles home")
    fig.tight_layout()
    fig.savefig(out / "joint_tracking.png", dpi=100)
    results = {"amplitude_rad": AMPLITUDE_RAD, "tolerance_rad": TOLERANCE_RAD, "settle_error_rad": error, "joints": joints}
    (out / "sweep.json").write_text(json.dumps(results, indent=2) + "\n")
    metrics = {
        "settle_error_rad": error,
        "min_peak_rad": min(peak),
        "max_tracking_error_rad": max(joint["max_error_rad"] for joint in joints.values()),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
sim.set_camera_view(**VIEW)
scene = InteractiveScene((VideoSceneCfg if args.out else SceneCfg)(num_envs=1, env_spacing=1.0))
sim.reset()

robot = scene["robot"]
joint_ids, _ = robot.find_joints(list(SIM_JOINT_NAMES), preserve_order=True)
home = robot.data.default_joint_pos.torch.clone()
lower, upper = robot.data.soft_joint_pos_limits.torch.unbind(-1)
dt = sim.get_physics_dt()
sweep_steps = round(SWEEP_S / dt)
pass_steps = len(joint_ids) * sweep_steps
if args.out:
    import imageio.v2 as imageio

    args.out.mkdir(parents=True, exist_ok=True)
    video = imageio.get_writer(args.out / "sweep.mp4", fps=VIDEO_FPS, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
    record_every = round(1 / (VIDEO_FPS * dt))
    angles = []  # per step: the targets and the measured angles, in SIM_JOINT_NAMES order

peak = [0.0] * len(joint_ids)  # how far each joint got from home during its own sweep
step = 0
while app.is_running():
    sweeping = sim.has_gui or step < pass_steps  # headless: one pass, then hold home to settle
    if not sweeping and step >= pass_steps + round(SETTLE_S / dt):
        break
    index, phase = divmod(step % pass_steps, sweep_steps)
    target = home.clone()
    if sweeping:
        target[:, joint_ids[index]] += AMPLITUDE_RAD * math.sin(2 * math.pi * phase / sweep_steps)
        if phase == 0:
            print(f"[sweep] {SIM_JOINT_NAMES[index]}")
    target = target.clamp(lower, upper)
    robot.actuators.target_command.set_position_index(value=target)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt)
    measured = robot.data.joint_pos.torch
    if sweeping:
        peak[index] = max(peak[index], (measured - home)[0, joint_ids[index]].abs().item())
    if args.out:
        angles.append((target[0, joint_ids].tolist(), measured[0, joint_ids].tolist()))
        if step % record_every == 0:
            video.append_data(scene["camera"].data.output["rgb"].torch[0, :, :, :3].cpu().numpy())
    step += 1

if not sim.has_gui:
    error = (robot.data.joint_pos.torch - home).abs().max().item()
    print(f"[sweep] peak travel {[round(p, 2) for p in peak]} rad, then {error:.3f} rad from home")
    if args.out:
        video.close()
        report(args.out, angles, peak, error, dt, sweep_steps)
    assert min(peak) > 0.8 * AMPLITUDE_RAD, "a joint did not follow its target"
    assert error < TOLERANCE_RAD, "the arm did not settle back home"
app.close()
