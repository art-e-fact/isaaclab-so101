"""Leader device logic, with Isaac Sim / Isaac Lab / Arena stubbed (see conftest.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest


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


def _device(isaac, monkeypatch, leader, **cfg):
    device_cls = isaac.leader_device.SO101LeaderDevice
    monkeypatch.setattr(device_cls, "_connect_leader", staticmethod(lambda _cfg: leader))
    return device_cls(isaac.leader_device.SO101LeaderDeviceCfg(**cfg))


def test_registered_device_forwards_leader_options(isaac):
    device = isaac.devices.SO101LeaderCfg()
    # arena-shape-sorting sets this attribute after construction.
    device.leader_recalibrate = True
    device.calibration_dir = "/tmp/calib"
    device.num_read_retries = 5
    device.max_consecutive_read_failures = 3

    cfg = device.get_device_cfg()

    assert cfg.leader_recalibrate is True
    assert cfg.calibration_dir == "/tmp/calib"
    assert cfg.num_read_retries == 5
    assert cfg.max_consecutive_read_failures == 3


def test_connect_passes_config_to_lerobot(isaac, monkeypatch):
    calls = []

    class FakeLeader:
        def __init__(self, config):
            calls.append(("init", config))

        def connect(self, calibrate):
            calls.append(("connect", calibrate))

        def calibrate(self):
            calls.append(("calibrate", self.calibration))

    so_leader = ModuleType("lerobot.teleoperators.so_leader")
    so_leader.SO101Leader = FakeLeader
    so_leader.SO101LeaderConfig = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "lerobot.teleoperators.so_leader", so_leader)
    cfg = isaac.leader_device.SO101LeaderDeviceCfg(calibration_dir="~/calib", num_read_retries=4, leader_recalibrate=True)

    isaac.leader_device.SO101LeaderDevice._connect_leader(cfg)

    (_, config), connect, calibrate = calls
    assert config["calibration_dir"] == Path.home() / "calib"
    assert config["num_read_retries"] == 4
    assert config["use_degrees"] is False
    # Recalibrate: connect without LeRobot's own prompt, then a fresh sweep with no stored calibration.
    assert connect == ("connect", False)
    assert calibrate == ("calibrate", {})


def test_holds_last_action_on_read_failure(isaac, monkeypatch):
    torch = pytest.importorskip("torch")
    device = _device(isaac, monkeypatch, _FlakyLeader(fail_after=1, failures=2))

    first = device.advance()
    assert torch.equal(device.advance(), first)
    assert torch.equal(device.advance(), first)
    assert not torch.equal(device.advance(), first)  # recovered: fresh reading


def test_raises_after_too_many_consecutive_failures(isaac, monkeypatch):
    device = _device(isaac, monkeypatch, _FlakyLeader(fail_after=1, failures=100), max_consecutive_read_failures=3)

    device.advance()
    for _ in range(3):
        device.advance()
    with pytest.raises(ConnectionError):
        device.advance()


def test_raises_if_first_read_fails(isaac, monkeypatch):
    device = _device(isaac, monkeypatch, _FlakyLeader(fail_after=0, failures=1))

    with pytest.raises(ConnectionError):
        device.advance()
