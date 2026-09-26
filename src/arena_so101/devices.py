"""Teleop device registrations for SO-101 (leader + gamepad).

Arena's built-in device library has keyboard / spacemouse / openxr but not
gamepad. We register ``so101_gamepad`` here so ``@register_retargeter`` pairs with
``so101_ik`` / ``so101_abs_joint`` resolve through ArenaEnvBuilder. The name is
SO-101 specific so it cannot collide with a generic Arena ``gamepad`` device.

``so101_leader`` returns an Isaac Lab ``DeviceCfg`` that emits absolute joint
targets for ``so101_abs_joint`` (not SE3).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from isaaclab.devices import Se3GamepadCfg
from isaaclab.devices.device_base import DeviceCfg

from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device

from arena_so101.constants import HOME_JOINT_POS, SIM_JOINT_NAMES
from arena_so101.joint_gamepad_device import SO101JointGamepadCfg
from arena_so101.leader_device import SO101LeaderDeviceCfg


def _resolve_joint_pos(joint_pos: dict[str, float]) -> tuple[float, ...]:
    """Per-joint values from an ``init_state.joint_pos`` dict, in ``SIM_JOINT_NAMES`` order.

    Keys may be exact names or regexes (Isaac Lab full-matches them); unmatched joints use the home pose.
    """
    return tuple(
        float(next((v for key, v in joint_pos.items() if re.fullmatch(key, name)), HOME_JOINT_POS[name]))
        for name in SIM_JOINT_NAMES
    )


@register_device
class SO101GamepadCfg(TeleopDeviceBase):
    """Registered as ``so101_gamepad``.

    Layout depends on the paired embodiment:
    - ``so101_abs_joint`` → absolute joint gamepad (:class:`SO101JointGamepadCfg`)
    - ``so101_ik`` → SE(3) gamepad (:class:`Se3GamepadCfg`)
    """

    name = "so101_gamepad"

    def __init__(
        self,
        sim_device: str | None = None,
        pos_sensitivity: float = 0.1,
        rot_sensitivity: float = 0.1,
        delta_scale: float = 0.03,
    ):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.delta_scale = delta_scale

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> DeviceCfg:
        emb_name = getattr(embodiment, "name", None)
        if emb_name == "so101_abs_joint":
            # Reset to the embodiment's current init pose (tracks set_joint_initial_pos).
            return SO101JointGamepadCfg(
                delta_scale=self.delta_scale,
                default_joint_pos=_resolve_joint_pos(embodiment.scene_config.robot.init_state.joint_pos),
                sim_device=self.sim_device or "cpu",
            )
        if emb_name == "so101_ik":
            return Se3GamepadCfg(
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
        raise ValueError(
            f"so101_gamepad has no layout for embodiment {emb_name!r}. "
            "Use --embodiment so101_abs_joint (joint-space) or so101_ik (SE3)."
        )


@register_device
class SO101LeaderCfg(TeleopDeviceBase):
    """Registered as ``so101_leader`` — absolute joints via LeRobot."""

    name = "so101_leader"

    def __init__(
        self,
        sim_device: str | None = None,
        port: str = "/dev/ttyACM0",
        leader_id: str = "leader",
        leader_recalibrate: bool = False,
        calibration_dir: str | None = None,
        num_read_retries: int = 2,
        max_consecutive_read_failures: int = 10,
    ):
        super().__init__(sim_device=sim_device)
        self.port = port
        self.leader_id = leader_id
        self.leader_recalibrate = leader_recalibrate
        self.calibration_dir = calibration_dir
        self.num_read_retries = num_read_retries
        self.max_consecutive_read_failures = max_consecutive_read_failures

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> SO101LeaderDeviceCfg:
        return SO101LeaderDeviceCfg(
            port=self.port,
            leader_id=self.leader_id,
            leader_recalibrate=self.leader_recalibrate,
            calibration_dir=self.calibration_dir,
            num_read_retries=self.num_read_retries,
            max_consecutive_read_failures=self.max_consecutive_read_failures,
            sim_device=self.sim_device or "cpu",
        )
