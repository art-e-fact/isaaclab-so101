"""The natural gamepad layout, one input at a time: examples/arena/natural_tour.py plays the README's table on a
scripted gamepad in Arena and records the embodiment's cameras. One case per input checks that the held stick
moved its own axis, that nothing else moved by itself but the lean the layout allows, and that the arm reached
the target; the last input is held past the arm's reach, where the target has to stop.

Artifacts in $SO101_SHOWCASE_DIR: external_camera.mp4, wrist_camera.mp4, a still of each, tcp_tracking.png and
natural_tour.json. Needs the Arena environment examples/arena/setup.sh builds.
"""

from __future__ import annotations

import json
import shlex
import sys

import pytest
from conftest import OUT, ROOT, run_child

sys.path.insert(0, str(ROOT / "examples" / "arena"))
from natural_tour import AXES, PHASES  # noqa: E402  (the phase table; the script imports Isaac Sim only in main)

from isaaclab_so101 import JAW_CLOSE_RAD, JAW_OPEN_RAD  # noqa: E402

POSITION_TOLERANCE_M = 0.015  # after settling; the smoke test holds a target within 1 cm and 3 degrees
ROTATION_TOLERANCE_DEG = 5.0
POSITION_AXES = ("pan", "reach", "height")


@pytest.fixture(scope="session")
def tour() -> dict:
    results = OUT / "natural_tour.json"
    results.unlink(missing_ok=True)
    script = ROOT / "examples" / "arena" / "natural_tour.py"
    shell = f"source examples/arena/setup.sh > /dev/null && exec python {shlex.quote(str(script))} --out {shlex.quote(str(OUT))}"
    run_child(["bash", "-c", shell], timeout_min=20)
    return json.loads(results.read_text())


@pytest.mark.parametrize("phase", [phase for phase in PHASES if phase.axis], ids=lambda phase: phase.name)
def test_input_moves_its_axis_and_the_arm_follows(tour, phase, record_property):
    result = tour["phases"][phase.name]
    delta = result["target_delta"]
    record_property("target_delta", json.dumps({axis: round(value, 4) for axis, value in delta.items()}))
    record_property("pos_error_mm", round(result["pos_error_m"] * 1e3, 1))
    record_property("rot_error_deg", round(result["rot_error_deg"], 2))
    if phase.axis == "jaw":
        expected = JAW_CLOSE_RAD if phase.direction < 0 else JAW_OPEN_RAD
        assert abs(result["jaw_rad"] - expected) < 0.1, f"Jaw at {result['jaw_rad']:.2f} rad, not {expected:.2f}"
    else:
        moved = delta[phase.axis] * phase.direction
        if phase.name == "edge":
            # The target goes out to the edge of the arm's reach and stops there instead of running away.
            assert moved > 0 and result["stalled_steps"] >= 10, (
                f"reach moved {moved:.3f} m and stalled for {result['stalled_steps']} of {phase.steps} held steps"
            )
        else:
            assert moved >= 0.8 * result["nominal_delta"], (
                f"{phase.axis} moved {moved:.3f} while the stick asked for {result['nominal_delta']:.3f}"
            )
        for other in AXES:
            # A position step the tilt forbids is retried with the gripper leaning a step: the one axis that
            # moves by itself.
            if other != phase.axis and not (other == "tilt" and phase.axis in POSITION_AXES):
                assert abs(delta[other]) < 1e-6, f"{other} moved {delta[other]:.4f} while {phase.input} was held"
    assert result["pos_error_m"] < POSITION_TOLERANCE_M, f"TCP {result['pos_error_m'] * 1e3:.1f} mm off after settling"
    assert result["rot_error_deg"] < ROTATION_TOLERANCE_DEG, f"TCP {result['rot_error_deg']:.1f} deg off after settling"


def test_cameras_recorded(tour, record_property):
    for name in ("external_camera", "wrist_camera"):
        video = OUT / f"{name}.mp4"
        record_property(name, video.name)
        assert video.stat().st_size > 50_000, f"{video.name} is {video.stat().st_size} bytes"
        assert tour["camera_std"][name] > 10, f"{name}: blank frames (pixel std {tour['camera_std'][name]:.1f})"
