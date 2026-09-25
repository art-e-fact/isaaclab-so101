"""Leader device logic, with Isaac Sim / Isaac Lab / Arena stubbed out."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")


def _module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def modules(monkeypatch):
    class DeviceBase:
        def __init__(self, retargeters=None):
            pass

    @dataclasses.dataclass
    class DeviceCfg:
        sim_device: str = "cpu"

    class TeleopDeviceBase:
        def __init__(self, sim_device=None):
            self.sim_device = sim_device

    stubs = {
        "carb": MagicMock(),
        "omni": MagicMock(),
        "isaaclab": _module("isaaclab"),
        "isaaclab.devices": _module("isaaclab.devices", Se3GamepadCfg=MagicMock()),
        "isaaclab.devices.device_base": _module(
            "isaaclab.devices.device_base", DeviceBase=DeviceBase, DeviceCfg=DeviceCfg
        ),
        "isaaclab.utils": _module("isaaclab.utils", configclass=dataclasses.dataclass),
        "isaaclab_arena": _module("isaaclab_arena"),
        "isaaclab_arena.assets": _module("isaaclab_arena.assets"),
        "isaaclab_arena.assets.device_library": _module(
            "isaaclab_arena.assets.device_library", TeleopDeviceBase=TeleopDeviceBase
        ),
        "isaaclab_arena.assets.register": _module("isaaclab_arena.assets.register", register_device=lambda cls: cls),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for name in ("arena_so101.devices", "arena_so101.leader_device", "arena_so101.joint_gamepad_device"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("arena_so101.devices"), importlib.import_module("arena_so101.leader_device")


def _cfg(leader_device, **overrides):
    return leader_device.SO101LeaderDeviceCfg(**overrides)


class _FlakyLeader:
    def __init__(self, fail_after: int, failures: int):
        self.calls = 0
        self.fail_after = fail_after
        self.failures = failures

    def get_action(self):
        self.calls += 1
        if self.fail_after < self.calls <= self.fail_after + self.failures:
            raise ConnectionError("Incorrect status packet!")
        keys = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
        return {f"{k}.pos": float(self.calls) for k in keys}


def test_registered_device_forwards_leader_options(modules):
    devices, _ = modules
    device = devices.SO101LeaderCfg()
    # arena-shape-sorting sets this attribute after construction.
    device.leader_recalibrate = True
    device.calibration_dir = "/tmp/calib"
    device.num_read_retries = 5

    cfg = device.get_device_cfg()

    assert cfg.leader_recalibrate is True
    assert cfg.calibration_dir == "/tmp/calib"
    assert cfg.num_read_retries == 5


def test_holds_last_action_on_read_failure(modules, monkeypatch):
    _, leader_device = modules
    leader = _FlakyLeader(fail_after=1, failures=2)
    monkeypatch.setattr(leader_device.SO101LeaderDevice, "_connect_leader", staticmethod(lambda cfg: leader))
    device = leader_device.SO101LeaderDevice(_cfg(leader_device))

    first = device.advance()
    assert torch.equal(device.advance(), first)
    assert torch.equal(device.advance(), first)
    assert not torch.equal(device.advance(), first)  # recovered: fresh reading


def test_raises_after_too_many_consecutive_failures(modules, monkeypatch):
    _, leader_device = modules
    leader = _FlakyLeader(fail_after=1, failures=100)
    monkeypatch.setattr(leader_device.SO101LeaderDevice, "_connect_leader", staticmethod(lambda cfg: leader))
    device = leader_device.SO101LeaderDevice(_cfg(leader_device, max_consecutive_read_failures=3))

    device.advance()
    for _ in range(3):
        device.advance()
    with pytest.raises(ConnectionError):
        device.advance()


def test_raises_if_first_read_fails(modules, monkeypatch):
    _, leader_device = modules
    leader = _FlakyLeader(fail_after=0, failures=1)
    monkeypatch.setattr(leader_device.SO101LeaderDevice, "_connect_leader", staticmethod(lambda cfg: leader))
    device = leader_device.SO101LeaderDevice(_cfg(leader_device))

    with pytest.raises(ConnectionError):
        device.advance()
