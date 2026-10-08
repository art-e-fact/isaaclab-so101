"""Each joint of the SO-101 in plain Isaac Lab, no Arena: examples/isaaclab/sweep.py moves the joints one at a
time and, with --out, records a camera and the joint angles. One case per joint checks it followed its target,
one that the arm settled back home.

Artifacts in $SO101_SHOWCASE_DIR: sweep.mp4, joint_tracking.png and sweep.json. Needs the uv project in
examples/isaaclab (uv run syncs it on a first run, Isaac Sim included).
"""

from __future__ import annotations

import json

import pytest
from conftest import OUT, run_child

from isaaclab_so101 import SIM_JOINT_NAMES


@pytest.fixture(scope="session")
def sweep() -> dict:
    results = OUT / "sweep.json"
    results.unlink(missing_ok=True)
    # --frozen: take examples/isaaclab's lock as it is.
    command = ["uv", "run", "--frozen", "--project", "examples/isaaclab", "python", "examples/isaaclab/sweep.py", "--out", str(OUT)]
    run_child(command, timeout_min=15)
    return json.loads(results.read_text())


@pytest.mark.parametrize("joint", SIM_JOINT_NAMES)
def test_joint_follows_its_target(sweep, joint, record_property):
    result = sweep["joints"][joint]
    record_property("peak_rad", round(result["peak_rad"], 3))
    record_property("max_error_rad", round(result["max_error_rad"], 3))
    assert result["peak_rad"] > 0.8 * sweep["amplitude_rad"], (
        f"{joint} got {result['peak_rad']:.2f} rad from home on a {sweep['amplitude_rad']} rad sweep"
    )


def test_arm_settles_home(sweep, record_property):
    record_property("settle_error_rad", round(sweep["settle_error_rad"], 4))
    assert sweep["settle_error_rad"] < sweep["tolerance_rad"], f"{sweep['settle_error_rad']:.3f} rad from home after settling"


def test_video_and_plot_written(sweep, record_property):
    for name in ("sweep.mp4", "joint_tracking.png"):
        record_property(name.split(".")[0], name)
        assert (OUT / name).stat().st_size > 10_000, f"{name} is {(OUT / name).stat().st_size} bytes"
