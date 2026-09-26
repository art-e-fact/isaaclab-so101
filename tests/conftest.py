from __future__ import annotations

import copy
import dataclasses
import importlib
import inspect
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

# arena_so101 modules that import Isaac Sim / Isaac Lab / Arena at module level.
# ``embodiments`` is one because its __init__ imports ``embodiments.so101``.
_ISAAC_DEPENDENT = (
    "assets",
    "cameras",
    "joint_gamepad_device",
    "leader_device",
    "devices",
    "embodiments",
    "embodiments.so101",
)


def _module(name: str) -> ModuleType:
    """Stub module that returns a (cached) MagicMock for any attribute not set on it."""
    mod = ModuleType(name)
    mocks: dict[str, MagicMock] = {}

    def __getattr__(attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        return mocks.setdefault(attr, MagicMock(name=f"{name}.{attr}"))

    mod.__getattr__ = __getattr__
    return mod


def _configclass(cls):
    """Isaac Lab's ``configclass`` as far as the tests need it: a dataclass that deep-copies
    its non-class defaults for every instance, so instances never share a default."""
    for key in inspect.get_annotations(cls):
        value = cls.__dict__.get(key, dataclasses.MISSING)
        if value is not dataclasses.MISSING and not isinstance(value, (dataclasses.Field, type)):
            setattr(cls, key, dataclasses.field(default_factory=lambda value=value: copy.deepcopy(value)))
    return dataclasses.dataclass(cls)


def _identity(cls):
    return cls


def _reset_joints_by_offset(env, env_ids, position_range, velocity_range, asset_cfg=None):
    """Plain function, so config deep copies keep it (as they keep Isaac Lab's)."""


@pytest.fixture
def isaac(monkeypatch):
    """Import the Isaac-dependent arena_so101 modules against stubbed Isaac Sim / Lab / Arena.

    Returns a namespace of the freshly imported modules, by their last name (``isaac.so101``).
    Everything, including the ``arena_so101.<module>`` entries in ``sys.modules`` and on the
    package, is restored afterwards.
    """
    pytest.importorskip("torch")
    import arena_so101  # noqa: F401  (pure; loaded first so the submodule attributes set on it get restored)

    class DeviceBase:
        def __init__(self, retargeters=None):
            pass

    @dataclasses.dataclass
    class DeviceCfg:
        sim_device: str = "cpu"

    class TeleopDeviceBase:
        def __init__(self, sim_device=None):
            self.sim_device = sim_device

    class EmbodimentBase:
        def __init__(self, *args, **kwargs):
            self.event_config = None
            self.base_kwargs = kwargs  # what the SO-101 classes pass through

        def get_events_cfg(self):
            return self.event_config

        def add_camera_variations(self, camera_rig):
            pass

        def set_joint_initial_pos(self, joint_pos):
            self.joint_initial_pos = dict(joint_pos)

    class ParallelJawGripper:
        def get_opening_width_m(self, world):
            return self.get_jaw_gap_m(world)

    class ArenaCameraCfg:
        def set_use_tiled_camera(self, use_tiled_camera):
            pass

    class ObservationGroupCfg:
        pass

    # Attributes the tests rely on. Anything else these modules import is a MagicMock.
    real_attrs = {
        "isaaclab.devices.device_base": {"DeviceBase": DeviceBase, "DeviceCfg": DeviceCfg},
        "isaaclab.envs.mdp": {"reset_joints_by_offset": _reset_joints_by_offset},
        "isaaclab.managers": {
            "EventTermCfg": SimpleNamespace,
            "ObservationGroupCfg": ObservationGroupCfg,
            "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
        },
        # Imported as ``from isaaclab.utils.configclass import configclass``: ``isaaclab.utils`` is a lazy
        # package, and its ``configclass`` attribute becomes the submodule once anything imports that directly.
        "isaaclab.utils.configclass": {"configclass": _configclass},
        "isaaclab_arena.assets.device_library": {"TeleopDeviceBase": TeleopDeviceBase},
        "isaaclab_arena.assets.register": {"register_asset": _identity, "register_device": _identity},
        "isaaclab_arena.embodiments.embodiment_base": {"EmbodimentBase": EmbodimentBase},
        "isaaclab_arena.embodiments.gripper": {"ParallelJawGripper": ParallelJawGripper},
        "isaaclab_arena.utils.cameras": {"ArenaCameraCfg": ArenaCameraCfg},
        "isaaclab.actuators": {},
        "isaaclab.assets.articulation": {},
        "isaaclab.controllers.differential_ik_cfg": {},
        "isaaclab.devices": {},
        "isaaclab.envs.mdp.actions.actions_cfg": {},
        "isaaclab.sensors": {},
        "isaaclab.sensors.frame_transformer.frame_transformer_cfg": {},
        "isaaclab.sim": {},
        "isaaclab.utils.math": {},
        "isaaclab_arena.embodiments.common.arm_mode": {},
        "isaaclab_arena.utils.pose": {},
    }
    stubs: dict[str, ModuleType | MagicMock] = {"carb": MagicMock(), "omni": MagicMock()}
    for name, attrs in real_attrs.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):  # the module and its parent packages
            stubs.setdefault(".".join(parts[:i]), _module(".".join(parts[:i])))
        stubs[name].__dict__.update(attrs)
    for name, mod in stubs.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(stubs[parent], child, mod)  # ``import a.b as x`` reads the attribute
        monkeypatch.setitem(sys.modules, name, mod)

    for short in _ISAAC_DEPENDENT:
        full = f"arena_so101.{short}"
        # setitem/setattr first so monkeypatch restores the pre-test state (absent or not), then clear.
        monkeypatch.setitem(sys.modules, full, None)
        del sys.modules[full]
        parent_name, _, attr = full.rpartition(".")
        parent = sys.modules.get(parent_name)  # already cleared for ``embodiments.so101``
        if parent is not None:
            monkeypatch.setattr(parent, attr, None, raising=False)
            delattr(parent, attr)

    return SimpleNamespace(
        **{short.rpartition(".")[2]: importlib.import_module(f"arena_so101.{short}") for short in _ISAAC_DEPENDENT}
    )
