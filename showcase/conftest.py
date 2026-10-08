"""Shared by the showcase: where its videos, pictures and results go, and how a simulation child is run.

The showcase (showcase/test_*.py) is not part of `pytest`, which collects tests/ only: each module boots Isaac Sim
for minutes and leaves videos or images for a person to look over. artefacts.yaml runs each as an Artefacts job;
by hand, from the repo root in a venv with pytest (and matplotlib, for the reachability map):

    SO101_SHOWCASE_DIR=/tmp/so101-showcase python -m pytest -s showcase/test_natural_gamepad.py

A module's simulation runs in a child process in the environment it needs (examples/arena/setup.sh, or the uv
project in examples/isaaclab), writes what it measured as JSON next to its videos, and the tests judge that. Kit's
exit, which can hang or skip Python's own shutdown, then never touches pytest's verdict.
"""

from __future__ import annotations

import os
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SO101_SHOWCASE_DIR", "/tmp/so101-showcase"))


def run_child(command: list[str], timeout_min: float) -> None:
    """Run a simulation script from the repo root in its own process group, and fail the test when it exits
    non-zero or hangs. The group is killed either way: Kit's helper processes outlive a clean exit."""
    OUT.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMNI_KIT_ACCEPT_EULA": "YES",  # the first-start prompt would hang a job with no terminal
        "ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE": "1",  # exit once the results are written, not in Kit's app.close()
        "PYTHONUNBUFFERED": "1",
    }
    proc = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        returncode = proc.wait(timeout=timeout_min * 60)
    except subprocess.TimeoutExpired:
        returncode = None
    finally:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # SIGKILL: Kit installs its own SIGTERM handler
        proc.wait()
    shown = " ".join(command)
    if returncode is None:
        pytest.fail(f"hung, killed after {timeout_min:g} min: {shown}")
    if returncode:
        pytest.fail(f"exited with {returncode} (its log is above): {shown}")
