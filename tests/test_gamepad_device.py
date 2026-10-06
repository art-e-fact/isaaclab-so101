from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena_so101 import JOINT_LIMITS_RAD
from arena_so101.ee_pose import HOME_NATURAL_POSE, natural_ee_quat_xyzw


def _events(pad, gamepad_input, *values):
    for value in values:
        pad._on_gamepad_event(SimpleNamespace(input=gamepad_input, value=value))


def test_callbacks_fire_once_per_press(isaac):
    mod = isaac.gamepad_device
    b = mod.carb.input.GamepadInput.B
    pad = mod.SO101NaturalGamepad(mod.SO101NaturalGamepadCfg())
    calls = []
    pad.add_callback(b, lambda: calls.append(1))

    _events(pad, b, 0.6, 1.0, 0.0, 1.0, 0.0)

    assert len(calls) == 2


def test_natural_gamepad_holds_a_reachable_tcp_pose(isaac):
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = mod.SO101NaturalGamepad(mod.SO101NaturalGamepadCfg(pos_delta_scale=0.01, delta_scale=0.1))
    x, y, z, tilt, roll = HOME_NATURAL_POSE

    out = pad.advance().tolist()
    assert len(out) == 8
    assert out[:3] == pytest.approx([x, y, z])
    assert out[3:7] == pytest.approx(natural_ee_quat_xyzw(x, y, tilt, roll), abs=1e-6)
    assert out[7] == 1.0  # a reset opens the jaw

    _events(pad, gi.LEFT_STICK_UP, 1.0)  # forward: -y in the base frame
    _events(pad, gi.RIGHT_TRIGGER, 0.5)  # up, half speed
    _events(pad, gi.RIGHT_STICK_UP, 1.0)  # fingertips swing away from the base: tilt decreases
    for _ in range(3):
        out = pad.advance().tolist()
    assert out[:3] == pytest.approx([x, y - 0.03, z + 0.015])
    assert out[3:7] == pytest.approx(natural_ee_quat_xyzw(x, y - 0.03, tilt - 0.3, roll), abs=1e-6)
    for key in (gi.LEFT_STICK_UP, gi.RIGHT_TRIGGER, gi.RIGHT_STICK_UP):
        _events(pad, key, 0.0)
    assert pad.advance().tolist() == pytest.approx(out)  # released: holds

    _events(pad, gi.LEFT_STICK_LEFT, 1.0)  # left: +x; the yaw follows the new azimuth
    moved = pad.advance().tolist()
    assert moved[0] == pytest.approx(x + 0.01)
    assert moved[3:7] == pytest.approx(natural_ee_quat_xyzw(x + 0.01, y - 0.03, tilt - 0.3, roll), abs=1e-6)
    assert moved[3:7] != pytest.approx(out[3:7], abs=1e-4)
    _events(pad, gi.LEFT_STICK_LEFT, 0.0)

    _events(pad, gi.X, 0.6, 1.0, 0.0)  # analog ramp, then release: one press closes
    assert pad.advance()[7].item() == -1.0
    _events(pad, gi.X, 1.0, 0.0)  # the next one opens
    assert pad.advance()[7].item() == 1.0
    _events(pad, gi.X, 1.0, 0.0)

    _events(pad, gi.RIGHT_TRIGGER, 1.0)  # held past the reach: the target stops at the limit
    for _ in range(200):
        out = pad.advance().tolist()
    assert out[2] == pytest.approx(0.45)
    _events(pad, gi.RIGHT_STICK_LEFT, 1.0)
    for _ in range(200):
        out = pad.advance().tolist()
    assert out[3:7] == pytest.approx(natural_ee_quat_xyzw(out[0], out[1], tilt - 0.3, JOINT_LIMITS_RAD[4][1]), abs=1e-6)

    pad.reset()
    out = pad.advance().tolist()
    assert out[:3] == pytest.approx([x, y, z]) and out[7] == 1.0
