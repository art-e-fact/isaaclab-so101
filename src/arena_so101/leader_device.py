"""Isaac Lab teleop device: physical SO-101 leader → absolute joint targets."""

from __future__ import annotations

import logging
import weakref
from collections.abc import Callable
from pathlib import Path

import torch

import carb
import omni
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils import configclass

from arena_so101.mapping import leader_dict_to_sim_radians

logger = logging.getLogger(__name__)


class SO101LeaderDevice(DeviceBase):
    """Read LeRobot SO-101 leader joints and emit a (6,) absolute joint action.

    Matches ``so101_abs_joint``. Keyboard callbacks registered via ``add_callback``
    are forwarded (stock record_demos uses ``R`` / ``RESET``).

    A failed bus read (e.g. a corrupted Feetech status packet) holds the last good
    action instead of ending the session; it raises only after
    ``cfg.max_consecutive_read_failures`` failures in a row.
    """

    def __init__(self, cfg: SO101LeaderDeviceCfg):
        super().__init__(retargeters=None)
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._additional_callbacks: dict[str, Callable] = {}
        self._last_action: torch.Tensor | None = None
        self._read_failures = 0
        self._leader = self._connect_leader(cfg)

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def __del__(self):
        try:
            if getattr(self, "_keyboard_sub", None) is not None:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception:
            pass
        try:
            leader = getattr(self, "_leader", None)
            if leader is not None:
                leader.disconnect()
        except Exception:
            pass

    def __str__(self) -> str:
        return (
            f"SO-101 Leader (abs joints): port={self.cfg.port!r} id={self.cfg.leader_id!r}\n"
            "\tMove the physical leader to drive the sim follower.\n"
            "\tKeyboard R: reset episode (when callbacks registered)."
        )

    def reset(self):
        """No buffered command state to clear."""

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        try:
            action = self._leader.get_action()
        except ConnectionError as exc:
            self._read_failures += 1
            if self._last_action is None or self._read_failures > self.cfg.max_consecutive_read_failures:
                raise
            logger.warning(
                "SO-101 leader read failed (%d in a row), holding last action: %s", self._read_failures, exc
            )
            return self._last_action
        self._read_failures = 0
        self._last_action = leader_dict_to_sim_radians(action, device=self._sim_device)
        return self._last_action

    @staticmethod
    def _connect_leader(cfg: SO101LeaderDeviceCfg):
        try:
            from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        except ImportError as exc:
            raise ImportError(
                "SO-101 leader needs lerobot[feetech]>=0.6.1,<0.7 (the arena-so101 `leader` extra). "
                "Install it into the Isaac Sim Python: python -m pip install 'lerobot[feetech]>=0.6.1,<0.7'"
            ) from exc

        leader = SO101Leader(
            SO101LeaderConfig(
                port=cfg.port,
                id=cfg.leader_id,
                # LeRobot mkdirs this as-is; expand "~" so it doesn't create a literal ./~ directory.
                calibration_dir=Path(cfg.calibration_dir).expanduser() if cfg.calibration_dir else None,
                use_degrees=False,
                num_read_retries=cfg.num_read_retries,
            )
        )
        leader.connect(calibrate=not cfg.leader_recalibrate)
        if cfg.leader_recalibrate:
            leader.calibration = {}  # skip "use existing file?" prompt
            leader.calibrate()
        return leader

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        # Forward any registered key (R/RESET used by stock record_demos; others optional).
        if event.input.name in self._additional_callbacks:
            self._additional_callbacks[event.input.name]()
        if event.input.name == "R" and "RESET" in self._additional_callbacks:
            self._additional_callbacks["RESET"]()
        return True


@configclass
class SO101LeaderDeviceCfg(DeviceCfg):
    """Config for :class:`SO101LeaderDevice`."""

    port: str = "/dev/ttyACM0"
    leader_id: str = "leader"
    leader_recalibrate: bool = False
    # None → LeRobot default (~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/).
    calibration_dir: str | None = None
    # Immediate retries inside LeRobot's sync_read.
    num_read_retries: int = 2
    # Consecutive failed reads to tolerate (holding the last action) before raising.
    max_consecutive_read_failures: int = 10
    retargeters: None = None
    # {DIR} is the parent package (arena_so101), same pattern as Se3KeyboardCfg.
    class_type: type[SO101LeaderDevice] | str = "{DIR}.leader_device:SO101LeaderDevice"
