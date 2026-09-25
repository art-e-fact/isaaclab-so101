from __future__ import annotations

import sys
from types import SimpleNamespace

from arena_so101 import HOME_JOINT_POS, SIM_JOINT_NAMES


def _embodiment(joint_pos: dict[str, float], name: str = "so101_abs_joint"):
    init_state = SimpleNamespace(joint_pos=joint_pos)
    return SimpleNamespace(name=name, scene_config=SimpleNamespace(robot=SimpleNamespace(init_state=init_state)))


def test_joint_gamepad_resets_to_embodiment_init_pose(isaac):
    joint_pos = dict(HOME_JOINT_POS)
    joint_pos["Jaw"] = 0.5  # what EmbodimentBase.set_joint_initial_pos does

    cfg = isaac.devices.SO101GamepadCfg().get_device_cfg(embodiment=_embodiment(joint_pos))

    assert cfg.default_joint_pos == tuple(joint_pos[name] for name in SIM_JOINT_NAMES)


def test_joint_gamepad_resolves_regex_keys_like_isaac_lab(isaac):
    cfg = isaac.devices.SO101GamepadCfg().get_device_cfg(embodiment=_embodiment({"Wrist_.*": 0.25, "Jaw": 0.5}))

    expected = dict(HOME_JOINT_POS, Wrist_Pitch=0.25, Wrist_Roll=0.25, Jaw=0.5)
    assert cfg.default_joint_pos == tuple(expected[name] for name in SIM_JOINT_NAMES)


def test_isaac_fixture_restores_modules(isaac):
    assert sys.modules["arena_so101.devices"] is isaac.devices


def test_isaac_fixture_left_no_stub_built_modules():
    # Runs after the tests above (file order): the stub-built modules must be gone.
    assert "arena_so101.devices" not in sys.modules
    assert "carb" not in sys.modules
