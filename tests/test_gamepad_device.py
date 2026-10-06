from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena_so101 import JOINT_LIMITS_RAD
from arena_so101.ee_pose import HOME_NATURAL_POSE, natural_ee_quat_xyzw, pan_reach_from_xy


def _events(pad, gamepad_input, *values):
    for value in values:
        pad._on_gamepad_event(SimpleNamespace(input=gamepad_input, value=value))


def _pad(mod, **cfg):
    return mod.SO101NaturalGamepad(mod.SO101NaturalGamepadCfg(**cfg))


def _natural(pad):
    """The device's target as (x, y, z, tilt, roll)."""
    return pad._natural(pad._target)


def test_callbacks_fire_once_per_press(isaac):
    mod = isaac.gamepad_device
    b = mod.carb.input.GamepadInput.B
    pad = _pad(mod)
    calls = []
    pad.add_callback(b, lambda: calls.append(1))

    _events(pad, b, 0.6, 1.0, 0.0, 1.0, 0.0)

    assert len(calls) == 2


def test_keyboard_r_and_back_button_reset_once_when_bound_as_r_and_reset(isaac):
    mod = isaac.gamepad_device
    pad = _pad(mod)
    calls = []
    pad.add_callback("R", lambda: calls.append("R"))  # what teleop_se3_agent.py binds
    pad.add_callback("RESET", lambda: calls.append("RESET"))

    pad._on_keyboard_event(
        SimpleNamespace(type=mod.carb.input.KeyboardEventType.KEY_PRESS, input=SimpleNamespace(name="R"))
    )
    assert calls == ["R"]

    _events(pad, mod.carb.input.GamepadInput.MENU1, 0.6, 1.0, 0.0)  # Back, one press
    assert calls == ["R", "RESET"]


def test_natural_gamepad_moves_the_fingertips_in_the_arms_own_directions(isaac):
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = _pad(mod, pos_delta_scale=0.01, delta_scale=0.1)
    x, y, z, tilt, roll = HOME_NATURAL_POSE
    pan, reach = pan_reach_from_xy(x, y)

    out = pad.advance().tolist()
    assert len(out) == 8
    assert out[:3] == pytest.approx([x, y, z], abs=1e-6)
    assert out[3:7] == pytest.approx(natural_ee_quat_xyzw(x, y, tilt, roll), abs=1e-6)
    assert out[7] == 1.0  # a reset opens the jaw

    _events(pad, gi.LEFT_STICK_UP, 1.0)  # reach out along the arm
    _events(pad, gi.RIGHT_TRIGGER, 0.5)  # up, half speed
    _events(pad, gi.RIGHT_STICK_UP, 1.0)  # fingertips swing away from the base: tilt decreases
    for _ in range(3):
        out = pad.advance().tolist()
    assert pan_reach_from_xy(*out[:2]) == pytest.approx((pan, reach + 0.03), abs=1e-6)
    assert out[2] == pytest.approx(z + 0.015, abs=1e-6)
    assert out[3:7] == pytest.approx(natural_ee_quat_xyzw(*out[:2], tilt - 0.3, roll), abs=1e-6)
    for key in (gi.LEFT_STICK_UP, gi.RIGHT_TRIGGER, gi.RIGHT_STICK_UP):
        _events(pad, key, 0.0)
    assert pad.advance().tolist() == pytest.approx(out)  # released: holds

    _events(pad, gi.LEFT_STICK_LEFT, 1.0)  # pan left: 1 cm sideways at the fingertips, the heading follows
    moved = pad.advance().tolist()
    new_pan, new_reach = pan_reach_from_xy(*moved[:2])
    assert new_reach == pytest.approx(reach + 0.03, abs=1e-6)
    assert new_pan == pytest.approx(pan + 0.01 / (reach + 0.03), abs=1e-6)
    assert moved[0] > out[0]  # left is +x
    assert moved[3:7] == pytest.approx(natural_ee_quat_xyzw(*moved[:2], tilt - 0.3, roll), abs=1e-6)
    assert moved[3:7] != pytest.approx(out[3:7], abs=1e-4)
    _events(pad, gi.LEFT_STICK_LEFT, 0.0)

    _events(pad, gi.X, 0.6, 1.0, 0.0)  # analog ramp, then release: one press closes
    assert pad.advance()[7].item() == -1.0
    _events(pad, gi.X, 1.0, 0.0)  # the next one opens
    assert pad.advance()[7].item() == 1.0

    pad.reset()
    out = pad.advance().tolist()
    assert out[:3] == pytest.approx([x, y, z], abs=1e-6) and out[7] == 1.0


def test_target_stops_where_the_arm_runs_out_and_the_other_axes_go_on(isaac, capsys):
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = _pad(mod)
    margin = pad.cfg.limit_margin
    _, reach0, z0, tilt0, _ = pad._target.tolist()

    def feasible():
        return pad._feasible(pad._target)

    _events(pad, gi.LEFT_STICK_UP, 1.0)  # reach out and lift, held for 20 s
    _events(pad, gi.RIGHT_TRIGGER, 1.0)
    for _ in range(300):
        pad.advance()
        assert feasible()
    pan, reach, z, tilt, roll = pad._target.tolist()
    assert reach > reach0 + 0.03 and z > z0 + 0.05  # both moved...
    assert (
        reach < pad.cfg.limits[1][1] - 0.05 and z < pad.cfg.limits[2][1] - 0.05
    )  # ...and the arm, not the box, stopped them
    assert pad.advance()[:3].tolist() == pytest.approx(pad.advance()[:3].tolist())  # and the target stays put
    assert capsys.readouterr().out.count("stopped") == 2  # each axis reported once
    for key in (gi.LEFT_STICK_UP, gi.RIGHT_TRIGGER):
        _events(pad, key, 0.0)

    pad.reseed()
    _events(pad, gi.LEFT_TRIGGER, 1.0)  # down to the table from home, where the jaws can turn down and back
    for _ in range(40):
        pad.advance()
    _events(pad, gi.LEFT_TRIGGER, 0.0)
    _events(pad, gi.RIGHT_STICK_DOWN, 1.0)  # jaws toward the base: Wrist_Pitch runs out before the box's π/2
    for _ in range(200):
        pad.advance()
        assert feasible()
    assert tilt0 + 0.5 < pad._target[3] < pad.cfg.limits[3][1] - 0.2
    _events(pad, gi.RIGHT_STICK_DOWN, 0.0)

    _events(pad, gi.RIGHT_STICK_LEFT, 1.0)  # roll: the box is Wrist_Roll, the margin stops it a step inside
    for _ in range(200):
        pad.advance()
    hi = JOINT_LIMITS_RAD[4][1] - margin
    assert hi - pad.cfg.delta_scale <= pad._target[4] <= hi + 1e-6


def test_stopped_is_reported_once_per_hold_even_when_the_lean_lets_steps_through(isaac, capsys):
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = _pad(mod)
    _events(pad, gi.LEFT_STICK_DOWN, 1.0)  # reach in and down: the reach is blocked on and off as the lean helps
    _events(pad, gi.LEFT_TRIGGER, 1.0)
    for _ in range(300):
        pad.advance()
    assert capsys.readouterr().out.count("stopped") <= 2
    for key in (gi.LEFT_STICK_DOWN, gi.LEFT_TRIGGER):
        _events(pad, key, 0.0)

    pad.reseed()
    _events(pad, gi.RIGHT_STICK_DOWN, 1.0)  # jaws toward the base: at home the wrist runs out within a few steps
    for _ in range(20):
        pad.advance()
    assert capsys.readouterr().out.count("tilt stopped") == 1
    _events(pad, gi.RIGHT_STICK_DOWN, 0.0)  # released and pushed again: reported again
    pad.advance()
    _events(pad, gi.RIGHT_STICK_DOWN, 1.0)
    pad.advance()
    assert capsys.readouterr().out.count("tilt stopped") == 1


def test_quaternion_stays_continuous_when_the_pan_crosses_minus_pi(isaac):
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = _pad(mod, pos_delta_scale=0.01)
    _events(pad, gi.LEFT_STICK_RIGHT, 1.0)  # pan right from home (-1.26 rad), past -π (90° right of forward)
    prev = pad.advance()[3:7]
    for _ in range(80):
        out = pad.advance()
        assert float(prev @ out[3:7]) > 0.9, pad._target[0]
        prev = out[3:7]
    assert pad._target[0] < -3.3  # it did cross, and the pan has more to go (Rotation's limit is -π/2 - 1.92)


def test_lifting_leans_the_gripper_forward_when_the_wrist_runs_out(isaac):
    # From home the Wrist_Pitch is 0.14 rad from its limit: a centimetre of lift at the home tilt is all it has.
    # The device leans the gripper a step forward per blocked step instead of stopping, so the lift goes on.
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = _pad(mod)
    _, _, z0, tilt0, _ = pad._target.tolist()
    _events(pad, gi.RIGHT_TRIGGER, 1.0)
    for _ in range(25):  # 10 cm; higher, the arm would run out of stretch
        pad.advance()
        assert pad._feasible(pad._target)
    assert pad._target[2] == pytest.approx(z0 + 25 * pad.cfg.pos_delta_scale, abs=1e-6)  # every step taken
    assert pad._target[3] < tilt0 - 0.1  # the gripper leaned forward to allow it
    assert pad._target[3] > tilt0 - 25 * pad.cfg.delta_scale  # but no more than a tilt step per lift step


def test_dpad_scales_the_speed_and_lb_slows(isaac):
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = _pad(mod, pos_delta_scale=0.01)

    def dz():
        before = pad._target[2]
        pad.advance()
        return before - pad._target[2]

    _events(pad, gi.LEFT_TRIGGER, 1.0)  # down: from home, plenty of room
    assert dz() == pytest.approx(0.01, abs=1e-6)
    _events(pad, gi.DPAD_UP, 1.0, 0.0)
    assert dz() == pytest.approx(0.0125, abs=1e-6)
    _events(pad, gi.LEFT_SHOULDER, 1.0)  # held: a quarter
    assert dz() == pytest.approx(0.0125 * 0.25, abs=1e-6)
    _events(pad, gi.LEFT_SHOULDER, 0.0)
    _events(pad, gi.DPAD_DOWN, 1.0, 0.0, 1.0, 0.0)
    assert dz() == pytest.approx(0.008, abs=1e-6)
    for _ in range(20):
        _events(pad, gi.DPAD_DOWN, 1.0, 0.0)
    assert dz() == pytest.approx(0.0025, abs=1e-6)  # floor x0.25
    pad.reset()  # forgets the held trigger, keeps the speed
    assert dz() == 0.0
    _events(pad, gi.LEFT_TRIGGER, 1.0)
    assert dz() == pytest.approx(0.0025, abs=1e-6)


def test_env_reset_event_reseeds_the_target_but_not_the_sticks(isaac):
    mod = isaac.gamepad_device
    gi = mod.carb.input.GamepadInput
    pad = _pad(mod, pos_delta_scale=0.01)
    reach_home = pad._target[1]
    _events(pad, gi.LEFT_STICK_UP, 1.0)
    _events(pad, gi.X, 1.0, 0.0)
    for _ in range(5):
        out = pad.advance()
    assert out[7] == -1.0 and pad._target[1] == pytest.approx(reach_home + 0.05, abs=1e-6)

    mod.reseed_natural_gamepads(env=None, env_ids=None)  # what the embodiment's reset event calls

    out = pad.advance().tolist()
    assert out[7] == 1.0
    assert pan_reach_from_xy(*out[:2])[1] == pytest.approx(reach_home + 0.01, abs=1e-6)  # home, then one step out
    term = isaac.so101.SO101AbsIKEmbodiment().get_events_cfg().reseed_gamepad_target  # what the env installs
    assert term.func is mod.reseed_natural_gamepads and term.mode == "reset"
