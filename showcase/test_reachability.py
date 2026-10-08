"""Where the natural gamepad lets its target go: ``natural_ik`` with the device's margins, over a grid of
(reach, height) in the arm's vertical plane, at a few tilts. No Isaac Sim: seconds, anywhere.

Artifacts in $SO101_SHOWCASE_DIR: reachability.png, one panel per tilt, and metrics.json. The cases check the
README's claims: the home pose is inside, with the jaws pointing down the fingertips cannot rise above about
12 cm, and the device stops 2 cm short of full stretch.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import OUT

from isaaclab_so101 import PAN_AXIS_XY
from isaaclab_so101.ee_pose import HOME_NATURAL_POSE, natural_ik, pan_reach_from_xy

# SO101NaturalGamepadCfg's defaults (that module imports Isaac Lab): how far inside each joint limit, full stretch
# and the table a step must stay, and the box the target may move in before natural_ik has its say.
MARGINS = {"margin": 0.02, "stretch_margin": 0.02, "wrist_floor": 0.06}
REACH_LIMITS, HEIGHT_LIMITS = (0.10, 0.5), (0.03, 0.5)
TABLE_TOP = 0.03  # the surface the robot stands on, above the base origin

HOME_TILT = HOME_NATURAL_POSE[3]
TILTS = {
    "jaws straight down (tilt 0)": 0.0,
    f"the home lean ({math.degrees(HOME_TILT):.0f}°)": HOME_TILT,
    "jaws horizontal, pointing out (-90°)": -math.pi / 2,
    "leaning back (+45°)": math.pi / 4,
}
STEP = 0.0025
REACH = np.arange(0.0, 0.5, STEP)
HEIGHT = np.arange(-0.1, 0.5, STEP)


def envelope(tilt: float, **margins) -> np.ndarray:
    """(height, reach) grid of where natural_ik holds the TCP straight ahead of the base (pan -π/2) at this tilt."""
    x = PAN_AXIS_XY[0]
    return np.array([[natural_ik(x, PAN_AXIS_XY[1] - r, z, tilt, 0.0, **margins) is not None for r in REACH] for z in HEIGHT])


@pytest.fixture(scope="session")
def envelopes() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per tilt: the arm's envelope, and the device's (the margins and the box limits). Draws reachability.png."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    in_reach = (REACH_LIMITS[0] <= REACH[None, :]) & (REACH[None, :] <= REACH_LIMITS[1])
    in_height = (HEIGHT_LIMITS[0] <= HEIGHT[:, None]) & (HEIGHT[:, None] <= HEIGHT_LIMITS[1])
    box = in_reach & in_height
    result = {label: (envelope(tilt), envelope(tilt, **MARGINS) & box) for label, tilt in TILTS.items()}

    fig, axes = plt.subplots(1, len(TILTS), figsize=(4.2 * len(TILTS), 5.2), sharey=True)
    extent = (REACH[0], REACH[-1], HEIGHT[0], HEIGHT[-1])
    home_reach = pan_reach_from_xy(*HOME_NATURAL_POSE[:2])[1]
    for ax, (label, (arm, device)) in zip(axes, result.items()):
        ax.imshow(arm.astype(int) + device.astype(int), origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=2.5)
        ax.axhline(TABLE_TOP, color="tab:brown", lw=1, label="table top")
        ax.plot(home_reach, HOME_NATURAL_POSE[2], "r*", ms=11, label="home TCP")
        ax.set_title(label)
        ax.set_xlabel("reach from the pan axis [m]")
    axes[0].set_ylabel("height above the base [m]")
    axes[0].legend(loc="upper right")
    fig.suptitle("Where the natural gamepad lets the fingertips go: light, the arm can hold it; dark, and the device allows it")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "reachability.png", dpi=110)

    jaws_down, horizontal = result["jaws straight down (tilt 0)"][1], result["jaws horizontal, pointing out (-90°)"]
    metrics = {
        "max_height_jaws_down_m": float(HEIGHT[jaws_down.any(axis=1)].max()),
        "max_reach_m": float(REACH[horizontal[0].any(axis=0)].max()),
        "device_max_reach_m": float(REACH[horizontal[1].any(axis=0)].max()),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return result


def test_home_pose_is_inside_the_envelope(record_property):
    record_property("home_natural_pose", str(HOME_NATURAL_POSE))
    assert natural_ik(*HOME_NATURAL_POSE, **MARGINS) is not None


def test_jaws_down_the_fingertips_cannot_rise_above_12cm(envelopes, record_property):
    _, device = envelopes["jaws straight down (tilt 0)"]
    top = float(HEIGHT[device.any(axis=1)].max())
    record_property("max_height_m", round(top, 3))
    assert 0.10 < top < 0.14, f"with the jaws pointing down the fingertips reach {top:.3f} m, not about 0.12"


def test_the_device_stops_2cm_short_of_full_stretch(envelopes, record_property):
    arm, device = envelopes["jaws horizontal, pointing out (-90°)"]
    full, allowed = float(REACH[arm.any(axis=0)].max()), float(REACH[device.any(axis=0)].max())
    record_property("full_stretch_m", round(full, 3))
    record_property("device_max_reach_m", round(allowed, 3))
    assert math.isclose(full - allowed, MARGINS["stretch_margin"], abs_tol=2 * STEP), f"{full:.3f} vs {allowed:.3f}"


def test_map_is_drawn(envelopes, record_property):
    record_property("map", "reachability.png")
    assert (OUT / "reachability.png").stat().st_size > 20_000
