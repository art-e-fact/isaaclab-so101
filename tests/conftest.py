from __future__ import annotations

import dataclasses
import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

# arena_so101 modules that import Isaac Sim / Isaac Lab / Arena at module level.
_ISAAC_DEPENDENT = ("joint_gamepad_device", "leader_device", "devices")


def _module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


@pytest.fixture
def isaac(monkeypatch):
    """Import the Isaac-dependent arena_so101 modules against stubbed Isaac Sim / Lab / Arena.

    Returns a namespace of the freshly imported modules. Everything, including the
    ``arena_so101.<module>`` entries in ``sys.modules`` and on the package, is restored afterwards.
    """
    pytest.importorskip("torch")
    import arena_so101

    class DeviceBase:
        def __init__(self, retargeters=None):
            pass

    @dataclasses.dataclass
    class DeviceCfg:
        sim_device: str = "cpu"

    class TeleopDeviceBase:
        def __init__(self, sim_device=None):
            self.sim_device = sim_device

    stubs = {
        "carb": MagicMock(),
        "omni": MagicMock(),
        "isaaclab": _module("isaaclab"),
        "isaaclab.devices": _module("isaaclab.devices", Se3GamepadCfg=MagicMock()),
        "isaaclab.devices.device_base": _module(
            "isaaclab.devices.device_base", DeviceBase=DeviceBase, DeviceCfg=DeviceCfg
        ),
        "isaaclab.utils": _module("isaaclab.utils", configclass=dataclasses.dataclass),
        "isaaclab_arena": _module("isaaclab_arena"),
        "isaaclab_arena.assets": _module("isaaclab_arena.assets"),
        "isaaclab_arena.assets.device_library": _module(
            "isaaclab_arena.assets.device_library", TeleopDeviceBase=TeleopDeviceBase
        ),
        "isaaclab_arena.assets.register": _module("isaaclab_arena.assets.register", register_device=lambda cls: cls),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for short in _ISAAC_DEPENDENT:
        # setitem/setattr first so monkeypatch restores the pre-test state (absent or not), then clear.
        monkeypatch.setitem(sys.modules, f"arena_so101.{short}", None)
        del sys.modules[f"arena_so101.{short}"]
        monkeypatch.setattr(arena_so101, short, None, raising=False)
        delattr(arena_so101, short)

    return SimpleNamespace(**{short: importlib.import_module(f"arena_so101.{short}") for short in _ISAAC_DEPENDENT})
