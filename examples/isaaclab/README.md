# SO-101 in Isaac Lab (no Arena)

Two examples that use the robot configs from `arena_so101.assets` with plain Isaac Lab:

- `sweep.py` spawns the arm with Isaac Lab's scene API and moves each joint in turn.
- `so101_reach.py` puts the arm in Isaac Lab's reach task, which Isaac Lab's own `isaaclab train` and
  `isaaclab play` run.

This directory is a uv project: `uv sync` installs Isaac Sim 6.1, Isaac Lab 3.0.0rc1 and arena-so101 (from `../..`,
editable). Arena is not installed.

Requirements: Linux x86_64, a GPU and driver that
[Isaac Sim 6.1 supports](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html),
[uv](https://docs.astral.sh/uv/getting-started/installation/), and about 20 GB of disk.

```bash
cd examples/isaaclab
uv sync                          # the first run downloads Isaac Sim, about 10 GB
uv run python sweep.py           # headless: checks every joint follows its target and the arm settles home
uv run python sweep.py --viz kit # in the Isaac Sim window, until you close it
```

Isaac Sim asks you to accept its EULA on first start. `OMNI_KIT_ACCEPT_EULA=YES` accepts it without the prompt.
The first `--viz kit` launch compiles shaders for several minutes with no terminal output.

## Train a reach policy

The arm learns to hold its gripper link on a goal above the table, which jumps to a new random point every 4 s.
`so101_reach.py` swaps the SO-101 into Isaac Lab's reach task and changes its rewards back to plain goal
tracking, which the rc1 task replaced with an episode that ends on the first touch.

```bash
uv run isaaclab train --task SO101-Reach           # headless: 4096 envs, 200 iterations, ~10 min on an RTX 5070 Ti
uv run isaaclab play --task SO101-Reach --viz kit  # the latest checkpoint, 50 envs
uv run isaaclab play --task SO101-Reach --video --video_length 360  # or record 12 s headless
```

The trained gripper stays about 1.4 cm from its goal on average, counting the moves after each jump.
Checkpoints and clips go to `logs/rsl_rl/reach_so101/<date>/`. `uv run isaaclab train --task SO101-Reach --help` lists the
options, such as `--num_envs` and `--max_iterations`.

Isaac Lab's CLI finds the task through the `isaaclab.tasks` entry point in `pyproject.toml`: it imports
`so101_reach`, which registers `SO101-Reach` with gym. A project of your own registers its tasks the same way.

Physics is PhysX. Training runs without Kit (OvPhysX), at about 0.5 MB of host memory per env. `--viz kit` starts
Kit, which keeps a copy of every robot's meshes, about 32 MB each (the USD has 3.4 M vertices). So use the window
with `play`, which caps itself at 50 envs, or pass a small `--num_envs`. Newton, the reach task's default
backend, can't load the SO-101 USD yet.

## Isaac Lab version

`pyproject.toml` pins Isaac Lab 3.0.0rc1, the newest wheel on pypi.nvidia.com. The Arena example stays on
3.0.0b2 because its Arena commit needs it. The `[tool.uv]` constraints and overrides cover what the Isaac Lab
wheel's own dependencies leave out.
