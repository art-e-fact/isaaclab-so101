"""Teleop device registrations for SO-101 (leader + gamepad).

Arena's built-in device library has keyboard / spacemouse / openxr but not
gamepad. We register ``so101_gamepad`` here so ``@register_retargeter`` pairs with
``so101_abs_ik`` / ``so101_ik`` resolve through ArenaEnvBuilder. The name is
SO-101 specific so it cannot collide with a generic Arena ``gamepad`` device.

``so101_leader`` returns an Isaac Lab ``DeviceCfg`` that emits absolute joint
targets for ``so101_abs_joint`` (not SE3).
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.devices import Se3GamepadCfg
from isaaclab.devices.device_base import DeviceCfg

from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device

from isaaclab_so101.gamepad_device import SO101NaturalGamepadCfg
from isaaclab_so101.leader_device import SO101LeaderDeviceCfg


@register_device
class SO101GamepadCfg(TeleopDeviceBase):
    """Registered as ``so101_gamepad``.

    Layout depends on the paired embodiment:
    - ``so101_abs_ik`` → natural TCP-pose gamepad (:class:`SO101NaturalGamepadCfg`)
    - ``so101_ik`` → Isaac Lab's SE(3) gamepad (:class:`Se3GamepadCfg`)
    """

    name = "so101_gamepad"

    def __init__(
        self,
        sim_device: str | None = None,
        pos_sensitivity: float = 0.1,
        rot_sensitivity: float = 0.1,
        delta_scale: float = 0.03,
        pos_delta_scale: float = 0.004,
    ):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.delta_scale = delta_scale  # rad/step: the natural layout's tilt/roll
        self.pos_delta_scale = pos_delta_scale  # m/step: the natural layout's pan, reach and height

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> DeviceCfg:
        emb_name = getattr(embodiment, "name", None)
        if emb_name == "so101_abs_ik":
            return SO101NaturalGamepadCfg(
                delta_scale=self.delta_scale,
                pos_delta_scale=self.pos_delta_scale,
                sim_device=self.sim_device or "cpu",
            )
        if emb_name == "so101_ik":
            return Se3GamepadCfg(
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
        raise ValueError(
            f"so101_gamepad has no layout for embodiment {emb_name!r}. "
            "Use --embodiment so101_abs_ik (natural TCP pose) or so101_ik (SE3)."
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
