"""Isaac Lab teleop device: gamepad → absolute joint targets for SO-101."""

from __future__ import annotations

import weakref
from collections.abc import Callable

import numpy as np
import torch

import carb
import omni
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils.configclass import configclass

from arena_so101.constants import HOME_JOINT_POS, JAW_CLOSE_RAD, JAW_OPEN_RAD, JOINT_LIMITS_RAD, SIM_JOINT_NAMES

_JOINT_LIMITS_RAD = np.asarray(JOINT_LIMITS_RAD, dtype=np.float32)
_DEFAULT_JOINT_SIGNS = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


class SO101JointGamepad(DeviceBase):
    """Gamepad controller emitting a (6,) absolute joint command for ``so101_abs_joint``.

    Sticks/triggers integrate into a held joint target. X toggles jaw open/close
    to absolute limits (no wind-up against a gripped object), once per press. After a
    reset the jaw holds its reset pose until the next X press.
    """

    def __init__(self, cfg: SO101JointGamepadCfg):
        super().__init__(retargeters=None)
        self.cfg = cfg
        self.delta_scale = cfg.delta_scale
        self.dead_zone = cfg.dead_zone
        self._joint_signs = np.asarray(cfg.joint_signs, dtype=np.float32)
        self._sim_device = cfg.sim_device
        self._default_joint_pos = np.asarray(cfg.default_joint_pos, dtype=np.float32).copy()

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

        self._create_key_bindings()
        # (positive, negative) x joint index for sticks (joints 1–4).
        self._delta_joint_raw = np.zeros([2, 6], dtype=np.float32)
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
            f"SO-101 Joint Gamepad (abs): {self.__class__.__name__}\n"
            f"\tDevice name: {self._input.get_gamepad_name(self._gamepad)}\n"
            f"\tJoints: {', '.join(SIM_JOINT_NAMES)}\n"
            "\t----------------------------------------------\n"
            "\tRotation (base): RT (+) / LT (-)\n"
            "\tPitch: Left Stick Up/Down\n"
            "\tElbow: Left Stick Left/Right\n"
            "\tWrist_Pitch: Right Stick Up/Down\n"
            "\tWrist_Roll: Right Stick Left/Right\n"
            "\tJaw open/close: X (toggle)\n"
            "\tKeyboard R: reset episode (when callbacks registered)."
        )

    def reset(self):
        self._lt_val = 0.0
        self._rt_val = 0.0
        self._rotation_rate = 0.0
        self._delta_joint_raw.fill(0.0)
        self._joint_targets = self._default_joint_pos.copy()
        # The jaw target stays at its reset pose; the first X press flips it to the other limit.
        self._close_gripper = bool(self._joint_targets[5] < (JAW_CLOSE_RAD + JAW_OPEN_RAD) / 2)

    def add_callback(self, key: str | carb.input.GamepadInput, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        rates = np.zeros(6, dtype=np.float32)
        rates[0] = self._rotation_rate
        rates[1:5] = self._resolve_command_buffer(self._delta_joint_raw[:, 1:5])
        rates *= self._joint_signs * self.delta_scale

        self._joint_targets[:5] = np.clip(
            self._joint_targets[:5] + rates[:5],
            _JOINT_LIMITS_RAD[:5, 0],
            _JOINT_LIMITS_RAD[:5, 1],
        )

        return torch.tensor(self._joint_targets, dtype=torch.float32, device=self._sim_device)

    def _on_gamepad_event(self, event, *args, **kwargs) -> bool:
        cur_val = event.value
        if abs(cur_val) < self.dead_zone:
            cur_val = 0.0

        # Act on the press edge only: an analog ramp (0.6 -> 1.0) or the release is not a new press.
        pressed = cur_val > 0.5
        just_pressed = pressed and not self._pressed.get(event.input, False)
        self._pressed[event.input] = pressed

        if event.input == carb.input.GamepadInput.X and just_pressed:
            self._close_gripper = not self._close_gripper
            self._joint_targets[5] = JAW_CLOSE_RAD if self._close_gripper else JAW_OPEN_RAD

        if event.input == carb.input.GamepadInput.RIGHT_TRIGGER:
            self._rt_val = cur_val
            self._rotation_rate = self._rt_val - self._lt_val
        elif event.input == carb.input.GamepadInput.LEFT_TRIGGER:
            self._lt_val = cur_val
            self._rotation_rate = self._rt_val - self._lt_val

        if event.input in self._INPUT_STICK_VALUE_MAPPING:
            direction, joint_idx, value = self._INPUT_STICK_VALUE_MAPPING[event.input]
            self._delta_joint_raw[direction, joint_idx] = value * cur_val

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

    def _create_key_bindings(self):
        self._INPUT_STICK_VALUE_MAPPING = {
            carb.input.GamepadInput.LEFT_STICK_UP: (0, 1, 1.0),
            carb.input.GamepadInput.LEFT_STICK_DOWN: (1, 1, 1.0),
            carb.input.GamepadInput.LEFT_STICK_RIGHT: (0, 2, 1.0),
            carb.input.GamepadInput.LEFT_STICK_LEFT: (1, 2, 1.0),
            carb.input.GamepadInput.RIGHT_STICK_UP: (0, 3, 1.0),
            carb.input.GamepadInput.RIGHT_STICK_DOWN: (1, 3, 1.0),
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (0, 4, 1.0),
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (1, 4, 1.0),
        }

    @staticmethod
    def _resolve_command_buffer(raw_command: np.ndarray) -> np.ndarray:
        delta_command_sign = raw_command[1, :] > raw_command[0, :]
        delta_command = raw_command.max(axis=0)
        delta_command[delta_command_sign] *= -1
        return delta_command


@configclass
class SO101JointGamepadCfg(DeviceCfg):
    """Config for :class:`SO101JointGamepad`."""

    delta_scale: float = 0.03  # rad/step at full stick/trigger deflection
    dead_zone: float = 0.01
    joint_signs: tuple[float, ...] = _DEFAULT_JOINT_SIGNS
    # Reset target; SO101GamepadCfg fills this from the embodiment's init_state.
    default_joint_pos: tuple[float, ...] = tuple(HOME_JOINT_POS[name] for name in SIM_JOINT_NAMES)
    retargeters: None = None
    # Concrete device class (avoid "{DIR}...." so create_teleop_device cannot
    # resolve to the wrong module if the cfg is nested under DevicesCfg).
    class_type: type[SO101JointGamepad] = SO101JointGamepad
