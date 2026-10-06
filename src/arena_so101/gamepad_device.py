"""Isaac Lab teleop device: the SO-101 natural gamepad layout.

:class:`SO101NaturalGamepad` emits an (8,) absolute TCP pose + jaw for ``so101_abs_ik``: the sticks pan the arm,
reach out along it and lift, the gripper's yaw follows the arm (:mod:`arena_so101.ee_pose`), and a step is only
taken when :func:`natural_ik` finds joints for it, so the target never leaves what the arm can reach.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable

import numpy as np
import torch

import carb
import omni
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils.configclass import configclass

from arena_so101.constants import JOINT_LIMITS_RAD
from arena_so101.ee_pose import (
    HOME_NATURAL_POSE,
    natural_ee_quat_xyzw,
    natural_ik,
    pan_reach_from_xy,
    xy_from_pan_reach,
)


class SO101NaturalGamepad(DeviceBase):
    """Gamepad controller emitting an (8,) absolute TCP pose + jaw command for ``so101_abs_ik``:
    ``(x, y, z, qx, qy, qz, qw, jaw)`` in the base frame, jaw +1 open / -1 closed.

    The sticks and triggers integrate a held target in the arm's own coordinates ``(pan, reach, z, tilt, roll)``:
    the left stick pans the arm and reaches out along it, the triggers lift, the right stick leans and spins the
    gripper. The orientation is :func:`natural_ee_quat_xyzw` of it, so the gripper always faces along the arm.
    Each step is tried one axis at a time and kept only if :func:`natural_ik` finds joints inside their limits,
    so a stick held past the arm's reach stops the target at the edge while the other axes keep moving. (An
    open-loop target the arm cannot follow makes the differential IK oscillate.) A position step the arm cannot
    take at the current tilt is retried with the gripper leaning one step either way: the SO-101's Wrist_Pitch
    runs out early (with the jaws vertical the fingertips cannot rise above ~12 cm), so the lean, like the heading,
    takes care of itself when it must. Releasing the sticks holds the target. X toggles the jaw once per press; LB
    held slows everything; the D-pad scales the speed; Back resets the episode, as keyboard R does. An environment
    reset re-seeds the target (:func:`reseed_natural_gamepads`).
    """

    _live: weakref.WeakSet[SO101NaturalGamepad] = weakref.WeakSet()
    _AXES = ("pan", "reach", "height", "tilt", "roll")

    def __init__(self, cfg: SO101NaturalGamepadCfg):
        super().__init__(retargeters=None)
        self.cfg = cfg
        self.dead_zone = cfg.dead_zone
        self._sim_device = cfg.sim_device
        self._step = np.asarray(cfg.signs, dtype=np.float32) * np.asarray(
            [cfg.pos_delta_scale] * 3 + [cfg.delta_scale] * 2, dtype=np.float32
        )
        self._default_target = np.asarray(self._cylindrical(cfg.default_pose), dtype=np.float32)
        self._limits = np.asarray(cfg.limits, dtype=np.float32)

        # Stick input → (0 positive / 1 negative, target axis). Built here, not at import: ``carb.input`` only
        # exists once Kit has loaded its input plugin, which is after this module is imported headless.
        gi = carb.input.GamepadInput
        self._stick_map = {
            gi.LEFT_STICK_LEFT: (0, 0),  # pan left: counter-clockwise seen from above
            gi.LEFT_STICK_RIGHT: (1, 0),
            gi.LEFT_STICK_UP: (0, 1),  # reach out along the arm
            gi.LEFT_STICK_DOWN: (1, 1),
            gi.RIGHT_STICK_UP: (1, 3),  # tilt: fingertips swing away from the base
            gi.RIGHT_STICK_DOWN: (0, 3),
            gi.RIGHT_STICK_RIGHT: (1, 4),  # roll: clockwise seen from above
            gi.RIGHT_STICK_LEFT: (0, 4),
        }

        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._gamepad = self._appwindow.get_gamepad(0)
        self._gamepad_sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
        )
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # (positive, negative) x axis. A stick reports both directions (left 0.8 and right 0.0), so both are kept
        # and _resolve_command_buffer takes the larger.
        self._stick_raw = np.zeros([2, 5], dtype=np.float32)
        # Whether each input is held (value > 0.5), so X and callbacks act once per press.
        self._pressed: dict[carb.input.GamepadInput, bool] = {}
        self._additional_callbacks: dict[str | carb.input.GamepadInput, Callable] = {}
        self._speed = 1.0  # D-pad; kept across resets like a volume knob
        self._slow = False  # LB held
        self._blocked: set[int] = set()  # axes the arm could not follow during the current hold (reported once)
        self.reset()
        SO101NaturalGamepad._live.add(self)

    def __del__(self):
        try:
            if getattr(self, "_gamepad_sub", None) is not None:
                self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
        except Exception:
            pass
        try:
            if getattr(self, "_keyboard_sub", None) is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception:
            pass

    def __str__(self) -> str:
        return (
            f"SO-101 Natural Gamepad (TCP pose): {self.__class__.__name__}\n"
            f"\tDevice name: {self._input.get_gamepad_name(self._gamepad)}\n"
            "\t----------------------------------------------\n"
            "\tPan the arm left/right: Left Stick Left/Right\n"
            "\tReach out/in along the arm: Left Stick Up/Down\n"
            "\tFingertips up/down: RT / LT\n"
            "\tTilt: Right Stick Up (fingertips away from the base) / Down\n"
            "\tRoll: Right Stick Right (clockwise seen from above, jaws down) / Left\n"
            "\tJaw open/close: X (toggle)\n"
            f"\tSlow (x{self.cfg.slow_scale:g}): hold LB. Speed up/down: D-pad Up/Down\n"
            "\tReset episode: Back, or keyboard R (when callbacks registered)."
        )

    def reset(self):
        """Isaac Lab's device reset: forget held inputs and return the target to the reset pose, jaw open."""
        self._lt_val = 0.0
        self._rt_val = 0.0
        self._stick_raw.fill(0.0)
        self.reseed()

    def reseed(self):
        """Return the target to the reset pose (the home TCP) and open the jaw; held sticks keep working from there."""
        self._close_gripper = False
        self._target = self._default_target.copy()

    def add_callback(self, key: str | carb.input.GamepadInput, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        scale = self._speed * (self.cfg.slow_scale if self._slow else 1.0)
        rates = self._rates() * scale
        moving = np.flatnonzero(rates)
        self._blocked.intersection_update(moving.tolist())
        step = self._step.copy()
        # Pan: m/step at the fingertips → rad/step, so the sideways speed is the same at any reach (the default limits keep
        # the reach at 0.1 or more; the max() guards a custom floor).
        step[0] /= max(self._target[1], 0.1)
        for axis in moving:
            # One axis at a time: an axis the arm cannot follow stops at the edge, the others keep moving.
            trial = self._target.copy()
            trial[axis] = np.clip(trial[axis] + rates[axis] * step[axis], *self._limits[axis])
            candidates = [trial]
            if axis < 3:  # a position step: if the tilt forbids it, lean a step either way (forward first)
                for lean in (-1.0, 1.0):
                    leaned = trial.copy()
                    leaned[3] = np.clip(trial[3] + lean * abs(step[3]) * scale, *self._limits[3])
                    candidates.append(leaned)
            for candidate in candidates:
                if self._feasible(candidate):
                    self._target = candidate
                    break
            else:
                if axis not in self._blocked:  # once per hold: cleared above when the stick is released
                    self._blocked.add(axis)
                    print(f"[so101_gamepad] {self._AXES[axis]} stopped: the arm cannot go further")
        x, y, z, tilt, roll = self._natural(self._target)
        # The integrated pan keeps the quaternion continuous past ±π, where atan2 would flip its sign.
        quat = natural_ee_quat_xyzw(x, y, tilt, roll, azimuth=float(self._target[0]))
        jaw = -1.0 if self._close_gripper else 1.0
        return torch.tensor([x, y, z, *quat, jaw], dtype=torch.float32, device=self._sim_device)

    def _feasible(self, target: np.ndarray) -> bool:
        cfg = self.cfg
        joints = natural_ik(
            *self._natural(target),
            margin=cfg.limit_margin,
            stretch_margin=cfg.stretch_margin,
            wrist_floor=cfg.wrist_floor,
        )
        return joints is not None

    @staticmethod
    def _natural(target: np.ndarray) -> tuple[float, float, float, float, float]:
        pan, reach, z, tilt, roll = target.tolist()
        return (*xy_from_pan_reach(pan, reach), z, tilt, roll)

    @staticmethod
    def _cylindrical(pose: tuple[float, ...]) -> tuple[float, float, float, float, float]:
        x, y, z, tilt, roll = pose
        return (*pan_reach_from_xy(x, y), z, tilt, roll)

    def _rates(self) -> np.ndarray:
        """Per-axis rates in [-1, 1] from the stick buffers; RT − LT drives z."""
        rates = self._resolve_command_buffer(self._stick_raw)
        rates[2] = self._rt_val - self._lt_val
        return rates

    def _fire(self, *keys: str | carb.input.GamepadInput) -> None:
        """Call the first registered callback among ``keys`` (teleop_se3_agent binds R and RESET to one reset)."""
        for key in keys:
            if key in self._additional_callbacks:
                self._additional_callbacks[key]()
                return

    def _on_gamepad_event(self, event, *args, **kwargs) -> bool:
        cur_val = event.value
        if abs(cur_val) < self.dead_zone:
            cur_val = 0.0

        # Act on the press edge only: an analog ramp (0.6 -> 1.0) or the release is not a new press.
        pressed = cur_val > 0.5
        just_pressed = pressed and not self._pressed.get(event.input, False)
        self._pressed[event.input] = pressed

        gi = carb.input.GamepadInput
        if event.input == gi.X and just_pressed:
            self._close_gripper = not self._close_gripper
        elif event.input == gi.LEFT_SHOULDER:
            self._slow = pressed
        elif event.input in (gi.DPAD_UP, gi.DPAD_DOWN) and just_pressed:
            factor = self.cfg.speed_step if event.input == gi.DPAD_UP else 1.0 / self.cfg.speed_step
            self._speed = float(np.clip(self._speed * factor, *self.cfg.speed_range))
            print(f"[so101_gamepad] speed x{self._speed:.2f}")
        elif event.input == gi.MENU1 and just_pressed:  # Back
            self._fire("RESET", "R")

        if event.input == gi.RIGHT_TRIGGER:
            self._rt_val = cur_val
        elif event.input == gi.LEFT_TRIGGER:
            self._lt_val = cur_val

        if event.input in self._stick_map:
            direction, axis = self._stick_map[event.input]
            self._stick_raw[direction, axis] = cur_val

        if just_pressed:
            self._fire(event.input)

        return True

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name
        self._fire(name, "RESET") if name == "R" else self._fire(name)
        return True

    @staticmethod
    def _resolve_command_buffer(raw_command: np.ndarray) -> np.ndarray:
        delta_command_sign = raw_command[1, :] > raw_command[0, :]
        delta_command = raw_command.max(axis=0)
        delta_command[delta_command_sign] *= -1
        return delta_command


def reseed_natural_gamepads(env, env_ids) -> None:
    """Isaac Lab reset event (``mode="reset"``): return every live gamepad's target to its reset pose.

    The RL environment resets itself inside ``step`` when the task succeeds, and ``teleop_se3_agent.py`` only
    resets the device on a manual R. The device holds an absolute target, so without this the arm would snap home
    and then lunge back to the stale target. Isaac Lab's own devices emit deltas and never notice.
    """
    for pad in list(SO101NaturalGamepad._live):
        pad.reseed()


@configclass
class SO101NaturalGamepadCfg(DeviceCfg):
    """Config for :class:`SO101NaturalGamepad`. The target is ``(pan, reach, z, tilt, roll)``: the azimuth about
    the pan axis (forward, -Y, is -π/2; left is more), the distance from it, the height, the gripper's lean and spin.
    """

    pos_delta_scale: float = 0.004  # m/step at the fingertips, full deflection, speed x1
    delta_scale: float = 0.03  # rad/step (tilt, roll)
    speed_step: float = 1.25  # D-pad up / down multiplies / divides the speed by this
    speed_range: tuple[float, float] = (0.25, 2.0)
    slow_scale: float = 0.25  # while LB is held
    limit_margin: float = 0.02  # rad inside each joint limit a step must stay: the IK fights a target on a limit
    # m inside full stretch the wrist must stay. The last 2 cm of reach take the last 46° of elbow bend: there the
    # differential IK is sluggish and can wander onto the hyperextended elbow, which dead-ends at the Elbow limit
    # with the arm flat on the table and does not come back.
    stretch_margin: float = 0.02
    # Lowest height (m, base frame) for the wrist-pitch axis: the forearm, wrist and gripper housings hang 1.5-3 cm
    # below it, and the table top is 3 cm above the base origin. (The wrist camera mount can hang 9 cm below the
    # roll axis at some rolls; that is left to the operator's eye.)
    wrist_floor: float = 0.06
    dead_zone: float = 0.01
    signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)  # flip an axis (pan, reach, z, tilt, roll) for taste
    # Reset target, a natural pose (x, y, z, tilt, roll): the home TCP.
    # ponytail: fixed to the home pose; following a custom initial_joint_pose needs FK the device has no model for,
    # so such an arm is pulled to the home TCP on the first step after a reset.
    default_pose: tuple[float, ...] = HOME_NATURAL_POSE
    # Where the target may go before natural_ik has its say: the pan within Rotation, the fingertips at least 10 cm
    # out from the pan axis (closer, the gripper hangs into the base, which natural_ik does not know about), no
    # lower than the surface the robot stands on (its mesh bottom is 3 cm above the base origin), the gripper at
    # most horizontal, the roll within Wrist_Roll.
    limits: tuple[tuple[float, float], ...] = (
        (-np.pi / 2 - JOINT_LIMITS_RAD[0][1], -np.pi / 2 - JOINT_LIMITS_RAD[0][0]),
        (0.10, 0.5),
        (0.03, 0.5),
        (-np.pi / 2, np.pi / 2),
        JOINT_LIMITS_RAD[4],
    )
    retargeters: None = None
    class_type: type[SO101NaturalGamepad] = SO101NaturalGamepad
