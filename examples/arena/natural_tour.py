"""The natural gamepad layout played by a scripted gamepad, recorded: the README's input table, one input at a time.

    source ./setup.sh && python natural_tour.py --out /tmp/so101-showcase

Headless. Builds so101_table with ``so101_abs_ik`` and the real :class:`SO101NaturalGamepad` (against stand-ins
for the input plugin a headless Kit does not load), holds one input per phase, and steps the env with what
``advance()`` emits, as ``teleop_se3_agent.py`` does. ``--out`` receives the embodiment's camera videos
(``external_camera.mp4``, ``wrist_camera.mp4``) and a still of each, ``tcp_tracking.png`` (commanded vs measured
TCP) and ``natural_tour.json``, one entry per phase, which ``showcase/test_natural_gamepad.py`` turns into a
verdict. The ``natural_gamepad`` job in artefacts.yaml runs it that way.

The videos are the embodiment's own cameras, not the viewport: this Isaac Lab's ``env.render()`` no longer
returns frames (``render_mode="rgb_array"`` is deprecated, so Arena's viewport recorder gets none), and the
external camera rides on the base, which never moves here, so it is a fixed third-person view.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

SETTLE_STEPS = 20  # unheld steps after each input, so the arm reaches the target before it is measured (15 Hz: 1.3 s)


@dataclass(frozen=True)
class Phase:
    name: str
    input: str | None
    """The ``carb.input.GamepadInput`` held for ``steps``, or None."""
    steps: int
    axis: str | None = None
    """The target axis the input moves (pan, reach, height, tilt, roll), or ``jaw``."""
    direction: int = 0
    """The sign it moves it in."""


AXES = ("pan", "reach", "height", "tilt", "roll")  # the device's target, in its order

# The README's layout table, one input at a time, from the home pose. Tilt and roll first, while the gripper has
# room to lean either way; a lift leans the gripper by itself where the wrist pitch runs out, as the README says.
PHASES = (
    Phase("settle", None, 10),
    Phase("tilt_away", "RIGHT_STICK_UP", 20, "tilt", -1),
    Phase("tilt_back", "RIGHT_STICK_DOWN", 20, "tilt", +1),
    Phase("roll_clockwise", "RIGHT_STICK_RIGHT", 25, "roll", -1),
    Phase("roll_back", "RIGHT_STICK_LEFT", 25, "roll", +1),
    Phase("jaw_close", "X", 15, "jaw", -1),
    Phase("jaw_open", "X", 15, "jaw", +1),
    Phase("pan_left", "LEFT_STICK_LEFT", 30, "pan", +1),
    Phase("pan_right", "LEFT_STICK_RIGHT", 30, "pan", -1),
    Phase("lift", "RIGHT_TRIGGER", 15, "height", +1),
    Phase("reach_out", "LEFT_STICK_UP", 20, "reach", +1),
    Phase("reach_in", "LEFT_STICK_DOWN", 20, "reach", -1),
    Phase("lower", "LEFT_TRIGGER", 15, "height", -1),
    # Held far past the arm's reach: the target goes to the edge and stops there, and the arm holds it.
    Phase("edge", "LEFT_STICK_UP", 90, "reach", +1),
)

GAMEPAD_INPUTS = (
    "LEFT_STICK_LEFT", "LEFT_STICK_RIGHT", "LEFT_STICK_UP", "LEFT_STICK_DOWN",
    "RIGHT_STICK_UP", "RIGHT_STICK_DOWN", "RIGHT_STICK_RIGHT", "RIGHT_STICK_LEFT",
    "X", "LEFT_SHOULDER", "DPAD_UP", "DPAD_DOWN", "MENU1", "RIGHT_TRIGGER", "LEFT_TRIGGER",
)  # fmt: skip


class _NoInput:
    """carb's input interface as far as the device uses it, for a Kit with no gamepad to subscribe to."""

    def subscribe_to_gamepad_events(self, gamepad, callback):
        return 1

    def subscribe_to_keyboard_events(self, keyboard, callback):
        return 1

    def unsubscribe_to_gamepad_events(self, gamepad, subscription):
        pass

    def unsubscribe_to_keyboard_events(self, keyboard, subscription):
        pass

    def get_gamepad_name(self, gamepad):
        return "scripted"


def scripted_gamepad(sim_device: str):
    """The real device, fed by :func:`press` instead of carb. A headless Kit loads no input plugin, so the two
    interfaces the device looks up when constructed are stood in for."""
    import carb
    import omni

    from isaaclab_so101.gamepad_device import SO101NaturalGamepad, SO101NaturalGamepadCfg

    if not hasattr(carb, "input"):
        carb.input = SimpleNamespace(
            GamepadInput=SimpleNamespace(**{name: name for name in GAMEPAD_INPUTS}),
            KeyboardEventType=SimpleNamespace(KEY_PRESS=1),
            acquire_input_interface=_NoInput,
        )
    if not hasattr(omni, "appwindow"):
        window = SimpleNamespace(get_gamepad=lambda index: None, get_keyboard=lambda: None)
        omni.appwindow = SimpleNamespace(get_default_app_window=lambda: window)
    return SO101NaturalGamepad(SO101NaturalGamepadCfg(sim_device=sim_device))


def press(pad, name: str, value: float) -> None:
    """What carb sends when an input moves to ``value``: 1.0 presses a button or deflects a stick fully, 0.0 releases."""
    import carb

    pad._on_gamepad_event(SimpleNamespace(input=getattr(carb.input.GamepadInput, name), value=value))


def to_uint8(frame):
    """An (H, W, 3) camera observation as uint8: camera_obs gives uint8, or float in [0, 1]."""
    import numpy as np

    frame = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    return frame if frame.dtype == np.uint8 else np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def tour(args, out: Path) -> dict:
    import imageio.v2 as imageio
    import numpy as np
    from isaaclab.utils.math import quat_error_magnitude

    from smoke_test import build
    from so101_table import SO101TableEnvironmentCfg

    def prepare(arena_env) -> None:
        arena_env.task.episode_length_s = 600.0  # the lift task's 5 s would reset the arm mid-tour
        # Frame the arm's whole sweep from the front right and above, at video resolution. The embodiment's
        # lens is long for a 16:9 frame: shorten it so the raised, leaning gripper stays in the picture.
        arena_env.embodiment.set_external_camera_view(eye=(0.75, -0.75, 0.8), target=(0.15, 0.0, 0.12))
        camera = arena_env.embodiment.camera_config.external_camera
        camera.height, camera.width = 720, 1280
        camera.spawn.focal_length = 20.0

    env, _ = build(args, SO101TableEnvironmentCfg(embodiment="so101_abs_ik", enable_cameras=True), prepare)
    try:
        obs, _ = env.reset()
        pad = scripted_gamepad(env.unwrapped.device)
        print(pad)
        robot = env.unwrapped.scene["robot"]
        jaw = robot.find_joints("Jaw")[0][0]
        step_dt = env.unwrapped.step_dt
        cameras = {"external_camera": "external_camera_rgb", "wrist_camera": "camera_ego_rgb"}
        writers = {
            name: imageio.get_writer(out / f"{name}.mp4", fps=round(1 / step_dt), codec="libx264", pixelformat="yuv420p", macro_block_size=None)
            for name in cameras
        }
        stds = {name: [] for name in cameras}
        trace = []  # per step: phase, commanded TCP, measured TCP, position and rotation error
        phases = {}
        started = time.perf_counter()
        for phase in PHASES:
            before = pad._target.copy()  # the held (pan, reach, height, tilt, roll); the device keeps it private
            held = []
            if phase.input:
                press(pad, phase.input, 1.0)
            for i in range(phase.steps + SETTLE_STEPS):
                if i == phase.steps and phase.input:
                    press(pad, phase.input, 0.0)
                action = pad.advance()
                obs, *_ = env.step(action[None])
                if i < phase.steps:
                    held.append(pad._target.copy())
                for name, key in cameras.items():
                    frame = to_uint8(obs["camera_obs"][key][0])
                    writers[name].append_data(frame)
                    stds[name].append(float(frame.std()))
                    if phase.name == "settle" and i == phase.steps + SETTLE_STEPS - 1:
                        imageio.imwrite(out / f"{name}.png", frame)
                pos, quat = obs["policy"]["eef_pos"][0], obs["policy"]["eef_quat"][0]
                pos_error = (pos - action[:3].to(pos)).norm().item()
                rot_error = math.degrees(quat_error_magnitude(quat[None], action[3:7].to(quat)[None])[0].item())
                trace.append((phase.name, *action[:3].tolist(), *pos.tolist(), pos_error, rot_error))
            delta = pad._target - before
            stalled = 0  # trailing held steps on which the target did not move: the device refusing the step
            for current, previous in zip(reversed(held), reversed(held[:-1])):
                if not np.array_equal(current, previous):
                    break
                stalled += 1
            # What the held stick asks for per step; pan is m/step at the fingertips, as the device scales it.
            step = pad.cfg.pos_delta_scale if phase.axis in ("pan", "reach", "height") else pad.cfg.delta_scale
            if phase.axis == "pan":
                step /= max(float(before[1]), 0.1)
            phases[phase.name] = {
                "input": phase.input,
                "steps": phase.steps,
                "axis": phase.axis,
                "direction": phase.direction,
                "target_delta": dict(zip(AXES, delta.tolist())),
                "target": dict(zip(AXES, pad._target.tolist())),
                "nominal_delta": phase.steps * step if phase.axis in AXES else None,
                "stalled_steps": stalled,
                "pos_error_m": pos_error,
                "rot_error_deg": rot_error,
                "jaw_rad": robot.data.joint_pos.torch[0, jaw].item(),
            }
            moved = ", ".join(f"{axis} {value:+.3f}" for axis, value in zip(AXES, delta.tolist()) if abs(value) > 1e-6)
            print(f"[natural_tour] {phase.name}: {moved or 'held'}; {pos_error * 1e3:.1f} mm, {rot_error:.1f} deg off", flush=True)
        wall = time.perf_counter() - started
        for writer in writers.values():
            writer.close()
        plot(trace, step_dt, out)
    finally:
        env.close()
    return {
        "step_dt": step_dt,
        "phases": phases,
        "camera_std": {name: float(np.mean(values)) for name, values in stds.items()},
        "max_pos_error_m": max(p["pos_error_m"] for p in phases.values()),
        "max_rot_error_deg": max(p["rot_error_deg"] for p in phases.values()),
        "steps_per_s": len(trace) / wall,
        "wall_s": wall,
    }


def plot(trace: list, step_dt: float, out: Path) -> None:
    """tcp_tracking.png: commanded vs measured TCP in the base frame, the errors, and the phases as bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [row[0] for row in trace]
    data = np.array([row[1:] for row in trace])  # commanded xyz, measured xyz, position error, rotation error
    t = np.arange(len(trace)) * step_dt
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, "xyz")):
        ax.plot(t, data[:, i], "--", label="commanded")
        ax.plot(t, data[:, 3 + i], label="measured")
        ax.set_ylabel(f"TCP {label} [m]")
    axes[0].legend(loc="upper right")
    axes[3].plot(t, data[:, 6] * 1e3, label="position error [mm]")
    axes[3].plot(t, data[:, 7], label="rotation error [deg]")
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("time [s]")
    starts = [0] + [i for i in range(1, len(names)) if names[i] != names[i - 1]]
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(names)
        for ax in axes:
            ax.axvspan(t[start], t[end - 1], color="k", alpha=0.05 if k % 2 else 0.0)
        axes[0].text(t[start], axes[0].get_ylim()[1], names[start], rotation=90, va="top", fontsize=8)
    fig.suptitle("so101_abs_ik following the natural gamepad: commanded vs measured TCP, base frame")
    fig.tight_layout()
    fig.savefig(out / "tcp_tracking.png", dpi=100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Where the videos, stills, plot and natural_tour.json go.")
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    args = get_isaaclab_arena_cli_parser().parse_args(["--enable_cameras"])  # no --viz: headless
    with SimulationAppContext(args):
        results = tour(args, out)
        (out / "natural_tour.json").write_text(json.dumps(results, indent=2) + "\n")
        metrics = {key: results[key] for key in ("max_pos_error_m", "max_rot_error_deg", "steps_per_s")}
        metrics["edge_reach_m"] = results["phases"]["edge"]["target"]["reach"]
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        # Inside the app context: Kit's exit can skip what Python would print after it.
        print(f"[natural_tour] done: {metrics}", flush=True)


if __name__ == "__main__":
    main()
