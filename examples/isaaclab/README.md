# SO-101 in Isaac Lab (no Arena)

Two examples that use the robot configs from `isaaclab_so101.assets` with plain Isaac Lab:

- `sweep.py` spawns the arm with Isaac Lab's scene API and moves each joint in turn.
- `so101_reach.py` puts the arm in Isaac Lab's reach task, which Isaac Lab's own `isaaclab train` and
  `isaaclab play` run.

This directory is a uv project: `uv sync` installs Isaac Sim 6.2, Isaac Lab 3.0.0 and isaaclab-so101 (from `../..`,
editable). Arena is not installed.

Requirements: Linux x86_64, a GPU and driver that
[Isaac Sim 6.2 supports](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html),
[uv](https://docs.astral.sh/uv/getting-started/installation/), and about 30 GB of disk.

```bash
cd examples/isaaclab
uv sync                          # the first run downloads Isaac Sim, about 10 GB
uv run python sweep.py           # headless: checks every joint follows its target and the arm settles home
uv run python sweep.py --viz kit # in the Isaac Sim window, until you close it
uv run python sweep.py --out DIR # headless, and leaves a video, a joint-angle plot and sweep.json in DIR
```

Isaac Sim asks you to accept its EULA on first start. `OMNI_KIT_ACCEPT_EULA=YES` accepts it without the prompt.
The first `--viz kit` launch compiles shaders for several minutes with no terminal output.

## Train a reach policy

The arm learns to hold its gripper link on a goal above the table, which jumps to a new random point every 4 s.
`so101_reach.py` swaps the SO-101 into Isaac Lab's reach task and changes its rewards back to plain goal
tracking, which Isaac Lab 3.0 replaced with an episode that ends on the first touch.

```bash
uv run isaaclab train --task SO101-Reach           # headless: 4096 envs, 200 iterations, ~2 min on an RTX 5070 Ti
uv run isaaclab play --task SO101-Reach --viz kit  # the latest checkpoint, 50 envs
uv run isaaclab play --task SO101-Reach --video --video_length 360  # or record 12 s headless
```

The trained gripper ends each episode about 4 mm from its goal, 4 s after the goal last jumped
(`Metrics/ee_pose/position_error` in the training log).
Checkpoints and clips go to `logs/rsl_rl/reach_so101/<date>/`. `uv run isaaclab train --task SO101-Reach --help` lists the
options, such as `--num_envs` and `--max_iterations`.

Isaac Lab's CLI finds the task through the `isaaclab.tasks` entry point in `pyproject.toml`: it imports
`so101_reach`, which registers `SO101-Reach` with gym. A project of your own registers its tasks the same way.

Physics is PhysX. Training runs without Kit (OvPhysX), in about 4 GB of host memory plus 0.2 MB per env. `--viz kit`
starts Kit and its renderer, about 6 GB more but little per env: Isaac Lab 3.0.0rc1 kept a copy of every robot's
meshes under Kit, about 32 MB each (the USD has 3.4 M vertices), and 3.0.0 doesn't. `play` caps itself at 50 envs.
Newton, the reach task's default backend, can't load the SO-101 USD yet.

## Isaac Lab version

`pyproject.toml` pins Isaac Lab 3.0.0. The Arena example installs Isaac Lab from Arena's submodule instead, a
commit from before 3.0.0 (see `../arena/`). The `[tool.uv]` overrides are Isaac Lab's own, from its
`tools/wheel_builder/uv-overrides.txt`, and the constraints cover what the Isaac Lab wheel's own dependencies leave out.
