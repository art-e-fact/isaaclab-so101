"""Isaac Lab teleop device: the SO-101 natural gamepad layout.

:class:`SO101NaturalGamepad` emits an (8,) absolute TCP pose + jaw for ``so101_abs_ik``: the sticks move the
fingertips in the base frame and the gripper's yaw follows the arm (:mod:`arena_so101.ee_pose`), so every target
is one the arm can reach.
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
from arena_so101.ee_pose import HOME_NATURAL_POSE, natural_ee_quat_xyzw


class SO101NaturalGamepad(DeviceBase):
    """Gamepad controller emitting an (8,) absolute TCP pose + jaw command for ``so101_abs_ik``:
    ``(x, y, z, qx, qy, qz, qw, jaw)`` in the base frame, jaw +1 open / -1 closed.

    The sticks and triggers integrate a held target in the natural coordinates ``(x, y, z, tilt, roll)``; the
    orientation is :func:`natural_ee_quat_xyzw` of it, so the gripper always faces along the arm and the IK gets a
    pose it can reach. Directions are the robot's: forward is where the arm faces (-Y of the base frame), left is
    its left (+X). The target is clipped to ``limits`` so a push past the arm's reach does not run away.
    Releasing the sticks holds the target. X toggles the jaw once per press; a reset opens it. Keyboard R resets.
    """

    def __init__(self, cfg: SO101NaturalGamepadCfg):
        super().__init__(retargeters=None)
        self.cfg = cfg
        self.dead_zone = cfg.dead_zone
        self._sim_device = cfg.sim_device
        self._step = np.asarray(cfg.signs, dtype=np.float32) * np.asarray(
            [cfg.pos_delta_scale] * 3 + [cfg.delta_scale] * 2, dtype=np.float32
        )
        self._default_pose = np.asarray(cfg.default_pose, dtype=np.float32).copy()
        self._limits = np.asarray(cfg.limits, dtype=np.float32)

        # Stick input → (0 positive / 1 negative, target axis). Built here, not at import: ``carb.input`` only
        # exists once Kit has loaded its input plugin, which is after this module is imported headless.
        gi = carb.input.GamepadInput
        self._stick_map = {
            gi.LEFT_STICK_UP: (1, 1),  # forward = -y
            gi.LEFT_STICK_DOWN: (0, 1),
            gi.LEFT_STICK_LEFT: (0, 0),  # left = +x
            gi.LEFT_STICK_RIGHT: (1, 0),
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
        self.reset()

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
            "\tFingertips forward/back: Left Stick Up/Down\n"
            "\tFingertips left/right: Left Stick Left/Right\n"
            "\tFingertips up/down: RT / LT\n"
            "\tTilt: Right Stick Up (fingertips away from the base) / Down\n"
            "\tRoll: Right Stick Right (clockwise seen from above, jaws down) / Left\n"
            "\tJaw open/close: X (toggle)\n"
            "\tKeyboard R: reset episode (when callbacks registered)."
        )

    def reset(self):
        self._lt_val = 0.0
        self._rt_val = 0.0
        self._stick_raw.fill(0.0)
        self._close_gripper = False
        self._target = self._default_pose.copy()

    def add_callback(self, key: str | carb.input.GamepadInput, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        self._target = np.clip(self._target + self._rates() * self._step, self._limits[:, 0], self._limits[:, 1])
        x, y, z, tilt, roll = self._target.tolist()
        quat = natural_ee_quat_xyzw(x, y, tilt, roll)
        jaw = -1.0 if self._close_gripper else 1.0
        return torch.tensor([x, y, z, *quat, jaw], dtype=torch.float32, device=self._sim_device)

    def _rates(self) -> np.ndarray:
        """Per-axis rates in [-1, 1] from the stick buffers; RT − LT drives z."""
        rates = self._resolve_command_buffer(self._stick_raw)
        rates[2] = self._rt_val - self._lt_val
        return rates

    def _on_gamepad_event(self, event, *args, **kwargs) -> bool:
        cur_val = event.value
        if abs(cur_val) < self.dead_zone:
            cur_val = 0.0

        # Act on the press edge only: an analog ramp (0.6 -> 1.0) or the release is not a new press.
        pressed = cur_val > 0.5
        just_pressed = pressed and not self._pressed.get(event.input, False)
        self._pressed[event.input] = pressed

        gamepad_input = carb.input.GamepadInput
        if event.input == gamepad_input.X and just_pressed:
            self._close_gripper = not self._close_gripper

        if event.input == gamepad_input.RIGHT_TRIGGER:
            self._rt_val = cur_val
        elif event.input == gamepad_input.LEFT_TRIGGER:
            self._lt_val = cur_val

        if event.input in self._stick_map:
            direction, axis = self._stick_map[event.input]
            self._stick_raw[direction, axis] = cur_val

        if event.input in self._additional_callbacks and just_pressed:
            self._additional_callbacks[event.input]()

        return True

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        if event.input.name in self._additional_callbacks:
            self._additional_callbacks[event.input.name]()
        if event.input.name == "R" and "RESET" in self._additional_callbacks:
            self._additional_callbacks["RESET"]()
        return True

    @staticmethod
    def _resolve_command_buffer(raw_command: np.ndarray) -> np.ndarray:
        delta_command_sign = raw_command[1, :] > raw_command[0, :]
        delta_command = raw_command.max(axis=0)
        delta_command[delta_command_sign] *= -1
        return delta_command


@configclass
class SO101NaturalGamepadCfg(DeviceCfg):
    """Config for :class:`SO101NaturalGamepad`. Poses are ``(x, y, z, tilt, roll)`` in the base frame."""

    pos_delta_scale: float = 0.004  # m/step at full stick/trigger deflection
    delta_scale: float = 0.03  # rad/step (tilt, roll)
    dead_zone: float = 0.01
    signs: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)  # flip an axis for a mirrored camera or taste
    # Reset target: the home TCP.
    # ponytail: fixed to the home pose; following a custom initial_joint_pose needs FK the device has no model for,
    # so such an arm is pulled to the home TCP on the first step after a reset.
    default_pose: tuple[float, ...] = HOME_NATURAL_POSE
    # Where the target may go: forward of the pan axis (behind it the heading would flip and the pan cannot follow),
    # no lower than the surface the robot stands on (its mesh bottom is 3 cm above the base origin), within reach
    # upward, the gripper at most horizontal, the roll within Wrist_Roll.
    limits: tuple[tuple[float, float], ...] = (
        (-0.4, 0.4),
        (-0.4, -0.05),
        (0.03, 0.45),
        (-np.pi / 2, np.pi / 2),
        JOINT_LIMITS_RAD[4],
    )
    retargeters: None = None
    class_type: type[SO101NaturalGamepad] = SO101NaturalGamepad
