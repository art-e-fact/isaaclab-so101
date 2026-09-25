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
