from __future__ import annotations

import importlib
import sys


def test_assets_import_without_arena(isaac, monkeypatch):
    # Plain Isaac Lab users import the robot configs without Arena installed.
    for name in list(sys.modules):
        if name.split(".")[0] == "isaaclab_arena":
            monkeypatch.setitem(sys.modules, name, None)  # a None entry makes importing it raise ImportError
        elif name.startswith("arena_so101."):
            monkeypatch.delitem(sys.modules, name)  # re-import, so an Arena import in any of them fails too

    assets = importlib.import_module("arena_so101.assets")

    assert assets.SO101_CFG is not assets.SO101_HIGH_PD_CFG
