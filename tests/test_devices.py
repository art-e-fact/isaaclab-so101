from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_natural_gamepad_pairs_with_the_absolute_ik_embodiment(isaac):
    gamepad = isaac.devices.SO101GamepadCfg(pos_delta_scale=0.002)
    cfg = gamepad.get_device_cfg(embodiment=SimpleNamespace(name="so101_abs_ik"))

    assert isinstance(cfg, isaac.gamepad_device.SO101NaturalGamepadCfg)
    assert cfg.pos_delta_scale == 0.002 and cfg.delta_scale == 0.03
    with pytest.raises(ValueError, match="no layout"):  # the joint-space layout is gone
        gamepad.get_device_cfg(embodiment=SimpleNamespace(name="so101_abs_joint"))


def test_isaac_fixture_restores_modules(isaac):
    assert sys.modules["arena_so101.devices"] is isaac.devices


def test_isaac_fixture_left_no_stub_built_modules():
    # Runs after the tests above (file order): the stub-built modules must be gone.
    assert "arena_so101.devices" not in sys.modules
    assert "arena_so101.embodiments" not in sys.modules
    assert "arena_so101.assets" not in sys.modules
    assert "carb" not in sys.modules
    assert "isaaclab" not in sys.modules
