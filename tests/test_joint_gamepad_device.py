from __future__ import annotations

from types import SimpleNamespace

import pytest

from arena_so101 import HOME_JOINT_POS, JAW_CLOSE_RAD, JAW_OPEN_RAD, SIM_JOINT_NAMES


def _events(pad, gamepad_input, *values):
    for value in values:
        pad._on_gamepad_event(SimpleNamespace(input=gamepad_input, value=value))


def _jaw(pad) -> float:
    return pad.advance()[5].item()


@pytest.mark.parametrize(
    ("reset_jaw", "after_first_press", "after_second_press"),
    [
        (HOME_JOINT_POS["Jaw"], JAW_OPEN_RAD, JAW_CLOSE_RAD),  # home jaw is nearly closed
        (1.5, JAW_CLOSE_RAD, JAW_OPEN_RAD),
    ],
)
def test_x_toggles_the_jaw_once_per_press_from_its_reset_pose(isaac, reset_jaw, after_first_press, after_second_press):
    mod = isaac.joint_gamepad_device
    x = mod.carb.input.GamepadInput.X
    default_joint_pos = tuple(dict(HOME_JOINT_POS, Jaw=reset_jaw)[name] for name in SIM_JOINT_NAMES)
    pad = mod.SO101JointGamepad(mod.SO101JointGamepadCfg(default_joint_pos=default_joint_pos))

    assert _jaw(pad) == pytest.approx(reset_jaw)  # no snap to a limit
    _events(pad, x, 0.6, 1.0, 0.0)  # analog ramp, then release: one press
    assert _jaw(pad) == pytest.approx(after_first_press)
    _events(pad, x, 1.0, 0.0)
    assert _jaw(pad) == pytest.approx(after_second_press)

    _events(pad, x, 1.0, 0.0)
    pad.reset()
    assert _jaw(pad) == pytest.approx(reset_jaw)


def test_callbacks_fire_once_per_press(isaac):
    mod = isaac.joint_gamepad_device
    b = mod.carb.input.GamepadInput.B
    pad = mod.SO101JointGamepad(mod.SO101JointGamepadCfg())
    calls = []
    pad.add_callback(b, lambda: calls.append(1))

    _events(pad, b, 0.6, 1.0, 0.0, 1.0, 0.0)

    assert len(calls) == 2
