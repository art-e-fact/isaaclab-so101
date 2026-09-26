from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

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


def test_jaw_gap_matches_the_usd_tips(isaac):
    import torch

    world = SimpleNamespace(get_joint_position=lambda scene_key, joint: torch.tensor([math.radians(-10.0), 0.0, math.radians(100.0)]))
    gap = isaac.so101.SO101Gripper().get_jaw_gap_m(world)
    # Tip-to-tip distances measured on the USD meshes (pxr) at the Jaw limits and at zero.
    assert torch.allclose(gap, torch.tensor([0.0068, 0.0204, 0.1400]), atol=1e-3)
    assert torch.equal(isaac.so101.SO101Gripper().get_opening_width_m(world), gap)


@pytest.mark.parametrize("cls_name", EMBODIMENTS)
def test_constructor_forwards_arena_kwargs_and_the_joint_pose(isaac, cls_name):
    embodiment = getattr(isaac.so101, cls_name)(initial_joint_pose={"Jaw": 0.5}, spawn_cfg_addon={"robot": {}})
    assert embodiment.base_kwargs == {"spawn_cfg_addon": {"robot": {}}}
    assert embodiment.joint_initial_pos == {"Jaw": 0.5}
    assert isinstance(embodiment.gripper, isaac.so101.SO101Gripper)
