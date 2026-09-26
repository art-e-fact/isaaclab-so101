# SO-101 for Isaac Lab and IsaacLab-Arena

:construction: Under development. Tested against [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) `main`
at aa36f19 (2026-09-23), pinned in [`examples/arena/setup.sh`](examples/arena/setup.sh).

SO-101 follower embodiment (and optional leader-arm helpers) for
[IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena), and the robot configs on their own for plain
[Isaac Lab](#isaac-lab-without-arena).

Environments using this embodiment:
- [Arena Shape Sorting](https://github.com/art-e-fact/arena-shape-sorting/)

Planned features:
 - More natural teleop setup with gamepad and keyboard.

## Install

First, install [IsaacLab-Arena](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html),
or only [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) for the [Isaac Lab configs](#isaac-lab-without-arena).

Requires Python 3.12 (same as Arena). The repository is `isaaclab-so101`; the package you install is
`arena-so101`, and you import it as `arena_so101`. It is not on PyPI, so install from git:

```bash
uv add "arena-so101 @ git+https://github.com/art-e-fact/isaaclab-so101.git"
# optional extras: `lerobot` (LeRobot dataset recorder), `leader` (physical leader arm teleop)
uv add "arena-so101[lerobot,leader] @ git+https://github.com/art-e-fact/isaaclab-so101.git"
# or with pip, e.g. inside the Isaac Sim Python
python -m pip install "arena-so101[lerobot,leader] @ git+https://github.com/art-e-fact/isaaclab-so101.git"
```

The base package has no Python dependencies; Isaac Sim, Isaac Lab and Arena come from your
environment. The extras pin `lerobot>=0.6.1,<0.7`. If a lerobot upgrade would replace Isaac Sim's
`torch`, install with a constraints file that pins your Isaac Sim `torch`/`torchvision`.

After `SimulationApp` is running, register once:

```python
import arena_so101
arena_so101.register()  # so101_abs_joint, so101_rel_joint, so101_ik, so101_leader, so101_gamepad
```

Joint names, limits, the home pose, Jaw open/close targets, the TCP offset and asset paths are exported as
plain constants (no Isaac Sim needed): `from arena_so101 import SIM_JOINT_NAMES, HOME_JOINT_POS, JAW_OPEN_RAD, TCP_OFFSET, USD_PATH`.
`HOME_JOINT_POS` is read-only; pass `dict(HOME_JOINT_POS)` to configs. `CUROBO_ROBOT_YML` exists only
after running the cuRobo generator (see below).

Then use like any Arena embodiment:

```python
from isaaclab_arena.assets.registries import AssetRegistry

embodiment = AssetRegistry().get_asset_by_name("so101_abs_joint")(enable_cameras=True)
```

Arena finds external environments by path (`--external_environment_class_path module:Class`), and the
environment registers the SO-101 inside its `build()`. [`examples/arena/`](examples/arena/) has a minimal one that
teleoperates the arm in Arena's lift task; its `setup.sh` clones Arena at the tested commit and installs Isaac Sim,
Isaac Lab and Arena from Arena's own uv lock. [arena-shape-sorting](https://github.com/art-e-fact/arena-shape-sorting/blob/25ea6bfea43a5134570e924fb3cdacb663f59472/arena_envs/src/shape_sorting/shape_sorting_env.py#L131)
is a full one.

See the [IsaacLab-Arena documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/concepts/embodiment/index.html) for more details.

### Cameras

`enable_cameras=True` (the example's `--enable_cameras`) adds two 640×480 RGB cameras, observed in the `camera_obs`
group as `camera_ego_rgb` and `external_camera_rgb`:

- `camera_ego`: the workshop's wrist camera on the gripper (`SO101_WRIST_CAMERA_CFG`).
- `external_camera`: a third-person view attached to the base link, so it moves with the robot. Aim it with
  `embodiment.set_external_camera_view(eye, target)`, both relative to the robot base with the arm facing +X.
  The default is eye `(0.55, -0.6, 0.45)`, target `(0.12, 0, 0.1)`.

Both carry Arena's camera extrinsics and intrinsics variations. `arena_so101.cameras.look_at_offset(eye, target)`
builds the `CameraCfg.OffsetCfg` for a camera of your own (points in the camera's parent frame).

## Isaac Lab (without Arena)

`arena_so101.assets` imports only Isaac Lab. Like `isaaclab_assets`, import it after the simulation app starts:

```python
from arena_so101.assets import SO101_CFG  # also SO101_HIGH_PD_CFG (for IK), SO101_WRIST_CAMERA_CFG

robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

The arm faces +X: `init_state` yaws its base 90°, so commands sampled in the base frame reach forward along -Y.
[`examples/isaaclab/`](examples/isaaclab/) sweeps the joints with the scene API and trains the arm in Isaac Lab's
reach task with Isaac Lab's own `isaaclab train`. Tested with Isaac Lab 3.0.0rc1 on PhysX. Newton can't load the
USD yet: it carries its own world joint, and Newton won't merge it with the one `fix_root_link` adds.

## Embodiments

| Name | Actions |
|------|---------|
| `so101_abs_joint` | Absolute joint positions (leader + joint-space gamepad) |
| `so101_rel_joint` | Relative joint positions (for policies; no teleop device pairing) |
| `so101_ik` | Relative SE(3) differential IK + binary Jaw (keyboard / gamepad / spacemouse) |

Every episode reset returns the arm to `init_state.joint_pos`: the home pose, or whatever you set with
`set_joint_initial_pos`. Pass `reset_joint_noise=<rad>` to the constructor to add uniform noise to each joint.

Observations (`policy` group): `actions`, `joint_pos`, `joint_vel`, and for Franka parity `eef_pos` / `eef_quat`
(the TCP in the robot base frame, what `FrankaMimicEnv`-style code reads) and `gripper_pos` (the Jaw angle).
`initial_joint_pose={"Jaw": 0.5, ...}` in the constructor overrides the home pose for spawning and resets.
`get_gripper()` returns an Arena `ParallelJawGripper`: `get_jaw_gap_m` from the Jaw angle and the jaw geometry,
`get_position_w` at the TCP. Other constructor keywords (`collision_mode`, `spawn_cfg_addon`) go to Arena's
`EmbodimentBase`.

The arm faces +X. `set_initial_pose(Pose(position_xyz=...))` moves it and keeps it facing +X, and so do
placement relations (`embodiment.add_relation(On(table))`); a `rotation_xyzw` turns it from there. (The USD's
base link is yawed 90° to face +X; the embodiment composes that yaw into every pose Arena writes, so you never
pass it yourself.)

USD joints: `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.
The robot USD comes from the [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop).


`so101_ik` is a 5-DOF arm: DLS tracks EE position and does best-effort orientation on the 6D pose command.
The IK command and `ee_frame` target the TCP between the jaw tips (`TCP_OFFSET`: 10.2 cm along the jaws from the
wrist-roll axis, in the `gripper` link frame), so rotations pivot about the tips and reach rewards measure to them.

## cuRobo planning assets

Generate a URDF (from the workshop USD) plus a cuRobo robot YAML (collision spheres,
self-collision ignore matrix, locked Jaw, home pose) under the package's
`embodiments/data/curobo/` (`src/arena_so101/...` in a checkout; `--output-dir` writes elsewhere):

```bash
# Inside the Isaac Sim / Arena env (needs CUDA + nvidia-curobo)
python -m arena_so101.generate_curobo_config --headless
```

> TODO: Document manually authoring collision spheres.

Outputs:

| File | Purpose |
|------|---------|
| `embodiments/data/curobo/urdf/SO-ARM101-USD.urdf` | Kinematics matching sim joint names, plus a fixed `tcp` link at `TCP_OFFSET` |
| `embodiments/data/curobo/meshes/` | Link meshes referenced by the URDF |
| `embodiments/data/curobo/so101.yml` | cuRobo `robot_cfg` for `MotionPlanner`: plans target `tcp` (between the jaw tips); objects attach at `gripper` |

Rebuild the spheres from the URDF generated above, without Isaac Sim (still needs CUDA +
nvidia-curobo, and `usd-core` to read the authored spheres):

```bash
python -m arena_so101.generate_curobo_config --skip-usd-convert
```

It reads `urdf/` and `meshes/` from the output directory. For a URDF elsewhere, pass `--urdf <file>`
and `--asset-path <mesh dir>`.

Add `--visualize` to inspect fitted spheres in Viser.

### Joint-space gamepad layout (`so101_abs_joint` + `so101_gamepad`)

Sticks/triggers integrate into a held absolute joint target. Releasing sticks holds pose.

| Input | Joint |
|-------|-------|
| RT (+) / LT (−) | `Rotation` (base yaw) |
| Left stick up/down | `Pitch` |
| Left stick left/right | `Elbow` |
| Right stick up/down | `Wrist_Pitch` |
| Right stick right/left | `Wrist_Roll` |
| X | `Jaw` toggle open / close (absolute limits). After a reset the jaw keeps its reset pose until the first press. |

Speed is `delta_scale` on `SO101GamepadCfg` (default `0.03` rad/step at full deflection).

`so101_gamepad` with `so101_ik` uses Isaac Lab's SE(3) gamepad layout. The device was named
`gamepad` before; pass `--teleop_device so101_gamepad` now.

`so101_leader` emits a (6,) absolute joint vector, so it pairs only with `so101_abs_joint`. Also works with
Arena's `record_demos.py` when the env wires `--teleop_device so101_leader`.

`SO101LeaderCfg` options: `port`, `leader_id`, `leader_recalibrate`, `calibration_dir`
(default `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/`, file `<leader_id>.json`)
`num_read_retries` and `max_consecutive_read_failures`. A failed bus read holds the last pose
instead of ending the session; it raises after `max_consecutive_read_failures` (default 10) in a row.

## Development

The tests run without Isaac Sim: Isaac-dependent modules are imported against stubs (the `isaac`
fixture in `tests/conftest.py`), with real `torch` and `numpy`.

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu -e ".[dev]"  # ".[dev,lerobot,leader]" also runs the recorder test
.venv/bin/pytest && .venv/bin/ruff check
```

CI runs both variants on every pull request. Nothing in CI starts Isaac Sim.

## Acknowledgments

We used the SO-101 USD model from the [Sim-to-Real-SO-101-Workshop](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Exception: `src/arena_so101/embodiments/data/SO-ARM101-USD.usd` is Copyright NVIDIA
Corporation & Affiliates, from the
[Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop),
and is licensed under Apache-2.0 only.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.