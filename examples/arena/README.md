# SO-101 in Isaac Lab Arena

`so101_table.py` puts the SO-101 on a table with a cube to lift, using Arena's lift task. This directory is a
uv project: `uv sync` installs Isaac Sim, Isaac Lab, Arena and arena-so101 (from `../..`, editable). No Docker,
no clone of Arena.

Requirements: Linux x86_64, a GPU and driver that
[Isaac Sim 6.0 supports](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html),
[uv](https://docs.astral.sh/uv/getting-started/installation/), and about 20 GB of disk.

```bash
cd examples/arena
uv sync                        # the first run downloads Isaac Sim, about 10 GB
uv run python smoke_test.py    # headless: builds the env with each embodiment, steps it, checks the reset
```

Isaac Sim asks you to accept its EULA on first start. `OMNI_KIT_ACCEPT_EULA=YES` accepts it without the prompt.

## Teleoperate

Opens the Isaac Sim window. The keyboard needs no extra hardware; its key bindings print in the terminal.
The first launch compiles RTX shaders, which can take several minutes with no terminal output after
`AppLauncher initialization complete`. They are cached in the venv, so later launches take seconds.

```bash
uv run python -m isaaclab_arena.scripts.imitation_learning.teleop --viz kit \
  --external_environment_class_path so101_table:SO101TableEnvironment \
  so101_table --embodiment so101_ik --teleop_device keyboard
```

Flags before `so101_table` go to Isaac Lab and Arena, flags after it to the environment
(`SO101TableEnvironmentCfg` in `so101_table.py`). Other pairings:

| `--embodiment` | `--teleop_device` |
|---|---|
| `so101_ik` | `keyboard`, `spacemouse`, `so101_gamepad` |
| `so101_abs_joint` | `so101_gamepad`, `so101_leader` (a physical leader arm; add `--leader_port /dev/ttyACM0`) |

## Record demonstrations

Same flags, plus where to write the HDF5 file:

```bash
uv run python -m isaaclab_arena.scripts.imitation_learning.record_demos --viz kit \
  --dataset_file datasets/so101_lift.hdf5 --num_demos 5 \
  --external_environment_class_path so101_table:SO101TableEnvironment \
  so101_table --embodiment so101_abs_joint --teleop_device so101_gamepad
```

## Your own environment

Copy `so101_table.py`. The one SO-101-specific step is calling `arena_so101.register()` inside `build()`:
it imports Isaac Lab, so it has to run after the simulation app starts.

## Arena version

`pyproject.toml` pins the Arena commit that arena-shape-sorting runs, and copies that commit's `[tool.uv]`
settings (a dependency's `[tool.uv]` doesn't apply to the project that depends on it). Arena dropped its
Isaac Lab wheel install on 2026-09-03, so moving to a newer Arena means taking Isaac Lab from git as well.
Isaac Lab's `tools/wheel_builder` builds it from a commit.

Known gaps at this commit:

- Arena's `policy_runner` can't be imported from a git install: `isaaclab_arena/evaluation` has no `__init__.py`,
  so the wheel leaves it out. Teleop, record and replay work.
- `record_demos` and teleop need the window (`--viz kit`).
