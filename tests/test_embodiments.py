from __future__ import annotations

import dataclasses

import pytest

EMBODIMENTS = ("SO101AbsJointEmbodiment", "SO101RelJointEmbodiment", "SO101IKEmbodiment")


@pytest.mark.parametrize("cls_name", EMBODIMENTS)
def test_ee_frame_name_is_a_scene_entity(isaac, cls_name):
    embodiment = getattr(isaac.so101, cls_name)()
    # Arena tasks resolve it with SceneEntityCfg(name), i.e. env.scene[name].
    scene_entities = {field.name for field in dataclasses.fields(embodiment.scene_config)}
    assert embodiment.get_ee_frame_name(None) in scene_entities


@pytest.mark.parametrize("cls_name", EMBODIMENTS)
def test_reset_returns_the_arm_to_its_initial_pose(isaac, cls_name):
    cls = getattr(isaac.so101, cls_name)

    term = cls().get_events_cfg().reset_robot_joints
    noisy = cls(reset_joint_noise=0.05).get_events_cfg().reset_robot_joints

    # reset_joints_by_offset resets to the articulation's default_joint_pos (= init_state.joint_pos).
    assert term.func is isaac.so101.mdp_isaac_lab.reset_joints_by_offset
    assert term.mode == "reset"
    assert term.params["asset_cfg"].name == "robot"
    assert term.params["position_range"] == (0.0, 0.0)
    assert term.params["velocity_range"] == (0.0, 0.0)
    assert noisy.params["position_range"] == (-0.05, 0.05)
