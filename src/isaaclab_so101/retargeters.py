"""Retargeter entries pairing teleop devices with SO-101 embodiments.

The ``isaaclab_arena.plugins`` entry point imports this module on Arena's first registry lookup, which can come before
Kit starts, so it must not import ``carb``, ``omni`` or ``isaacsim``.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab_arena.assets.register import register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase


@register_retargeter
class SO101LeaderAbsJointRetargeter(RetargetterBase):
    device = "so101_leader"
    embodiment = "so101_abs_joint"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class SO101IKKeyboardRetargeter(RetargetterBase):
    device = "keyboard"
    embodiment = "so101_ik"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class SO101IKSpaceMouseRetargeter(RetargetterBase):
    device = "spacemouse"
    embodiment = "so101_ik"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class SO101IKGamepadRetargeter(RetargetterBase):
    device = "so101_gamepad"
    embodiment = "so101_ik"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class SO101AbsIKGamepadRetargeter(RetargetterBase):
    device = "so101_gamepad"
    embodiment = "so101_abs_ik"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None
