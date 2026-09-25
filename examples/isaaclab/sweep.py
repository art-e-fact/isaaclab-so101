"""Spawn the SO-101 and sweep each joint in turn, with Isaac Lab's scene API: no managers, no Arena.

    uv run python sweep.py             # headless: one pass over the joints, then checks the arm settled home
    uv run python sweep.py --viz kit   # in the Isaac Sim window, until you close it
"""

# ruff: noqa: E402  (Isaac Lab modules are imported after the app starts)

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep the SO-101's joints one at a time.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from arena_so101 import SIM_JOINT_NAMES
from arena_so101.assets import SO101_CFG

AMPLITUDE_RAD = 0.5
SWEEP_S = 2.0  # per joint: one sine period, so each joint is home again when the next starts
SETTLE_S = 1.0
TOLERANCE_RAD = 0.05


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device=args.device))
sim.set_camera_view(eye=(0.7, -0.7, 0.5), target=(0.1, 0.0, 0.15))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0))
sim.reset()

robot = scene["robot"]
joint_ids, _ = robot.find_joints(list(SIM_JOINT_NAMES), preserve_order=True)
home = robot.data.default_joint_pos.torch.clone()
lower, upper = robot.data.soft_joint_pos_limits.torch.unbind(-1)
dt = sim.get_physics_dt()
sweep_steps = round(SWEEP_S / dt)
pass_steps = len(joint_ids) * sweep_steps

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
    robot.actuators.target_command.set_position_index(value=target.clamp(lower, upper))
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt)
    if sweeping:
        peak[index] = max(peak[index], (robot.data.joint_pos.torch - home)[0, joint_ids[index]].abs().item())
    step += 1

if not sim.has_gui:
    error = (robot.data.joint_pos.torch - home).abs().max().item()
    print(f"[sweep] peak travel {[round(p, 2) for p in peak]} rad, then {error:.3f} rad from home")
    assert min(peak) > 0.8 * AMPLITUDE_RAD, "a joint did not follow its target"
    assert error < TOLERANCE_RAD, "the arm did not settle back home"
app.close()
