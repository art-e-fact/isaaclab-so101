# SO-101 in Isaac Lab Arena

`so101_table.py` puts the SO-101 on a table with a cube to lift, using Arena's lift task. `setup.sh` builds the
environment that runs it: it clones [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) at a pinned
commit into `IsaacLab-Arena/`, runs `uv sync` there (Isaac Sim, Isaac Lab from Arena's submodule and Arena, from
Arena's own lock), installs arena-so101 from `../..` editable into that environment, and activates it. No Docker.

Requirements: Linux x86_64, a GPU and driver that
[Isaac Sim 6.1 supports](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html),
[uv](https://docs.astral.sh/uv/getting-started/installation/), git, and about 40 GB of disk.

```bash
cd examples/arena
source ./setup.sh          # the first run downloads Isaac Sim, about 10 GB; later runs only activate
python smoke_test.py       # headless: builds the env with each embodiment, steps it, checks the reset
```

Isaac Sim asks you to accept its EULA on first start. `OMNI_KIT_ACCEPT_EULA=YES` accepts it without the prompt.
`source ./setup.sh --leader` adds the `leader` extra (lerobot) for a physical leader arm; `--force` re-syncs.

## Teleoperate

Arena environments run through Isaac Lab's own scripts: `--external_callback` names a function that registers the
Arena environment given by `--task`, and `--external_environment_class_path` tells that function where ours is
(`setup.sh` puts this directory on `PYTHONPATH` for it). The flags the environment defines
(`SO101TableEnvironmentCfg` in `so101_table.py`) are accepted next to Isaac Lab's own. This opens the Isaac Sim
window; the keyboard bindings print in the terminal.

```bash
python IsaacLab-Arena/submodules/IsaacLab/scripts/environments/teleoperation/teleop_se3_agent.py --viz kit \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task so101_table --external_environment_class_path so101_table:SO101TableEnvironment \
  --embodiment so101_ik --teleop_device keyboard
```

The first launch compiles RTX shaders, which can take several minutes with no terminal output after
`AppLauncher initialization complete`. Pairings:

| `--embodiment` | `--teleop_device` |
|---|---|
| `so101_ik` | `keyboard`, `spacemouse`, `so101_gamepad` |
| `so101_abs_joint` | `so101_gamepad`, `so101_leader` (a physical leader arm; add `--leader_port /dev/ttyACM0`) |

`--teleop_device` is read twice: Arena's callback builds that device for the chosen embodiment into the environment
config, and Isaac Lab's script then picks it up from there by the same name.

## Record demonstrations

Same flags, with Isaac Lab's recorder and where to write the HDF5 file:

```bash
python IsaacLab-Arena/submodules/IsaacLab/scripts/tools/record_demos.py --viz kit \
  --external_callback isaaclab_arena.environments.isaaclab_interop.environment_registration_callback \
  --task so101_table --external_environment_class_path so101_table:SO101TableEnvironment \
  --embodiment so101_abs_joint --teleop_device so101_gamepad \
  --dataset_file datasets/so101_lift.hdf5 --num_demos 5
```

For a LeRobot dataset instead, see [Recording LeRobot datasets](../../README.md#recording-lerobot-datasets)
in the main README.

## Your own environment

Copy `so101_table.py`. The one SO-101-specific step is calling `arena_so101.register()` inside `build()`:
it imports Isaac Lab, so it has to run after the simulation app starts.

## Arena version

`ARENA_REV` in `setup.sh` pins the IsaacLab-Arena commit (`main` of 2026-09-23). Arena pins Isaac Lab through its
`submodules/IsaacLab` and Isaac Sim through its `uv.lock`, so that one SHA fixes the whole stack. The clone is
blobless and leaves Arena's LFS files (docs images, test data) as pointers: about 200 MB with the Isaac Lab
submodule. To move the pin, change `ARENA_REV` and source the script again; it checks the managed clone out at the
new commit and re-syncs. `ARENA_DIR=/path/to/IsaacLab-Arena` uses a checkout of your own instead, at whatever
commit it is on.

Until 2026-09-25 this directory was a uv project of its own, stuck on an older Arena commit: Arena dropped its
Isaac Lab wheel install on 2026-09-03, and Arena `main` installs Isaac Lab editable from its submodule, which only
Arena's own environment can do.
