from __future__ import annotations

import math
import subprocess
import sys

import pytest

import arena_so101
from arena_so101 import (
    HOME_JOINT_POS,
    JAW_CLOSE_RAD,
    JAW_OPEN_RAD,
    JOINT_LIMITS_RAD,
    SIM_JOINT_NAMES,
)


def test_import_is_pure_python():
    code = "import sys, arena_so101; assert not {'torch', 'isaaclab'} & set(sys.modules), sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_jaw_targets_are_usd_limits():
    assert JAW_OPEN_RAD == pytest.approx(math.radians(100.0))
    assert JAW_CLOSE_RAD == pytest.approx(math.radians(-10.0))


def test_home_pose_covers_all_joints_within_limits():
    assert tuple(HOME_JOINT_POS) == SIM_JOINT_NAMES
    for name, (lo, hi) in zip(SIM_JOINT_NAMES, JOINT_LIMITS_RAD, strict=True):
        assert lo <= HOME_JOINT_POS[name] <= hi, name


def test_home_pose_is_read_only():
    with pytest.raises(TypeError):
        HOME_JOINT_POS["Jaw"] = 0.0  # type: ignore[index]


def test_asset_paths_exist():
    assert arena_so101.USD_PATH.is_file()


def test_mapping_round_trip_and_endpoints():
    torch = pytest.importorskip("torch")
    from arena_so101.mapping import motor_norm_to_sim_radians, sim_radians_to_motor_norm

    lows = torch.tensor([-100.0] * 5 + [0.0])
    highs = torch.tensor([100.0] * 5 + [100.0])
    assert torch.allclose(motor_norm_to_sim_radians(lows), torch.tensor([lo for lo, _ in JOINT_LIMITS_RAD]))
    assert torch.allclose(motor_norm_to_sim_radians(highs), torch.tensor([hi for _, hi in JOINT_LIMITS_RAD]))

    raw = torch.tensor([-37.0, 0.0, 12.5, 99.0, -80.0, 42.0])
    assert torch.allclose(sim_radians_to_motor_norm(motor_norm_to_sim_radians(raw)), raw, atol=1e-4)
