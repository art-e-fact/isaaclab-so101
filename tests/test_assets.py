from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


def test_assets_import_without_arena(isaac, monkeypatch):
    # Plain Isaac Lab users import the robot configs without Arena installed.
    for name in list(sys.modules):
        if name.split(".")[0] == "isaaclab_arena":
            monkeypatch.setitem(sys.modules, name, None)  # a None entry makes importing it raise ImportError
        elif name.startswith("isaaclab_so101."):
            monkeypatch.delitem(sys.modules, name)  # re-import, so an Arena import in any of them fails too

    assets = importlib.import_module("isaaclab_so101.assets")

    assert assets.SO101_CFG is not assets.SO101_HIGH_PD_CFG


def test_linear_rgb_reads_hex_as_srgb_and_triples_as_linear(isaac):
    linear_rgb = isaac.assets.linear_rgb
    assert linear_rgb("#ffffff") == (1.0, 1.0, 1.0)
    assert linear_rgb("000000") == (0.0, 0.0, 0.0)
    assert linear_rgb("#808080") == pytest.approx((0.2158605,) * 3)  # sRGB mid-grey is 22% linear
    assert linear_rgb([1, 0.82, 0.12]) == (1.0, 0.82, 0.12)  # the USD's own yellow, as the material holds it
    for bad in ("#fff", "#gggggg", (1.0, 0.0), (2.0, 0.0, 0.0)):
        with pytest.raises((AssertionError, ValueError)):
            linear_rgb(bad)


def test_color_event_draws_each_env_a_listed_colour_in_linear_rgb(isaac):
    term = isaac.assets.color_event(["#ff0000", (0.0, 0.0, 0.5)], mode="startup")
    assert term.func is isaac.assets.mdp_isaac_lab.randomize_visual_material
    assert term.mode == "startup"
    assert term.params["materials"].name == "robot_color"
    assert term.params["channels"] == {"color": {"choices": [(1.0, 0.0, 0.0), (0.0, 0.0, 0.5)]}}


def test_write_color_writes_one_linear_colour_to_every_env_or_the_given_ones(isaac):
    import torch

    writes = []
    material = SimpleNamespace(num_instances=3, device="cpu")
    material.write_channels = lambda materials, channels, env_ids: writes.append((materials, channels, env_ids))

    isaac.assets.write_color(material, "#ffffff")
    isaac.assets.write_color(material, (0.5, 0.0, 0.0), env_ids=torch.tensor([2]))

    (materials, every, all_envs), (_, one, env_ids) = writes
    assert materials == [material] and all_envs is None
    assert torch.equal(every["color"], torch.ones(1, 3, 3))  # (materials, envs, rgb)
    assert torch.equal(one["color"], torch.tensor([[[0.5, 0.0, 0.0]]]))
    assert env_ids.tolist() == [2]
