# SO-101 for Isaac Lab and IsaacLab-Arena

🚧 Under development. Tested against [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) `main` at aa36f19
(2026-09-23), pinned in
[`examples/arena/setup.sh`](https://github.com/art-e-fact/isaaclab-so101/blob/main/examples/arena/setup.sh). All tested
combinations are under [Compatibility](#compatibility).

SO-101 follower embodiment (and optional leader-arm helpers) for
[IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena), and the robot configs on their own for plain
[Isaac Lab](#isaac-lab-without-arena).

A community package, not affiliated with or endorsed by NVIDIA.

Environments using this embodiment:
- [Arena Shape Sorting](https://github.com/art-e-fact/arena-shape-sorting/)

Planned features:
 - A natural teleop layout for the keyboard (the gamepad already has one: [Natural gamepad layout](#natural-gamepad-layout-so101_abs_ik--so101_gamepad)).

<img width="1740" height="988" alt="image" src="https://github.com/user-attachments/assets/34f830f5-8955-47f8-ba60-1b45eedd32e9" />


## Install

First, install [IsaacLab-Arena](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html),
or only [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) for the [Isaac Lab configs](#isaac-lab-without-arena).

Requires Python 3.12 (same as Arena).

```bash
uv add isaaclab-so101
# optional extras: `lerobot` (LeRobot dataset recorder), `leader` (physical leader arm teleop)
uv add "isaaclab-so101[lerobot,leader]"
# or with pip, e.g. inside the Isaac Sim Python
python -m pip install "isaaclab-so101[lerobot,leader]"
# unreleased changes, from git
uv add "isaaclab-so101 @ git+https://github.com/art-e-fact/isaaclab-so101.git"
```

The base package has no Python dependencies; Isaac Sim, Isaac Lab and Arena come from your
environment. The extras pin `lerobot>=0.6.1,<0.7`. If a lerobot upgrade would replace Isaac Sim's
`torch`, install with a constraints file that pins your Isaac Sim `torch`/`torchvision`.

After `SimulationApp` is running, register once:

```python
import isaaclab_so101
isaaclab_so101.register()  # so101_abs_joint, so101_rel_joint, so101_ik, so101_abs_ik, so101_leader, so101_gamepad
```

Joint names, limits, the home pose, Jaw open/close targets, the TCP offset, the pan axis and asset paths are exported as
plain constants (no Isaac Sim needed): `from isaaclab_so101 import SIM_JOINT_NAMES, HOME_JOINT_POS, JAW_OPEN_RAD, TCP_OFFSET, PAN_AXIS_XY, USD_PATH`.
`HOME_JOINT_POS` is read-only; pass `dict(HOME_JOINT_POS)` to configs. `CUROBO_ROBOT_YML` is the shipped cuRobo
config (see below).

Then use like any Arena embodiment:

```python
from isaaclab_arena.assets.registries import AssetRegistry

embodiment = AssetRegistry().get_asset_by_name("so101_abs_joint")(enable_cameras=True)
```

Arena finds external environments by path (`--external_environment_class_path module:Class`), and the environment
registers the SO-101 inside its `build()`.
[`examples/arena/`](https://github.com/art-e-fact/isaaclab-so101/tree/main/examples/arena) has a minimal one that
teleoperates the arm in Arena's lift task; its `setup.sh` clones Arena at the tested commit and installs Isaac Sim,
Isaac Lab and Arena from Arena's own uv lock.
[arena-shape-sorting](https://github.com/art-e-fact/arena-shape-sorting/blob/25ea6bfea43a5134570e924fb3cdacb663f59472/arena_envs/src/shape_sorting/shape_sorting_env.py#L131)
is a full one.

See the [IsaacLab-Arena documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/concepts/embodiment/index.html) for more details.

### Cameras

`enable_cameras=True` (the example's `--enable_cameras`) adds two 640×480 RGB cameras, observed in the `camera_obs`
group as `camera_ego_rgb` and `external_camera_rgb`:

- `camera_ego`: the workshop's wrist camera on the gripper (`SO101_WRIST_CAMERA_CFG`).
- `external_camera`: a third-person view attached to the base link, so it moves with the robot. Aim it with
  `embodiment.set_external_camera_view(eye, target)`, both relative to the robot base with the arm facing +X.
  The default is eye `(0.55, -0.6, 0.45)`, target `(0.12, 0, 0.1)`.

Both carry Arena's camera extrinsics and intrinsics variations. `isaaclab_so101.cameras.look_at_offset(eye, target)`
builds the `CameraCfg.OffsetCfg` for a camera of your own (points in the camera's parent frame).

### Recording LeRobot datasets

`isaaclab_so101.lerobot` (the `lerobot` extra) writes rollouts straight into a LeRobot v3 dataset. `observation.state`
is `policy.joint_pos`; `action` is the absolute joint targets the sim received for that step, so all three
embodiments record the same joint-space action (relative and IK actions end as position targets too); and there is
one video per camera in `cameras` (sim observation term → dataset key; the default is the two embodiment cameras as
`observation.images.ego_view` / `exterior_image`). Frames are uint8 or float in [0, 1], as `camera_obs` gives them.

```python
from isaaclab_so101.lerobot import SO101LeRobotRecorder, camera_shapes, joint_targets

with SO101LeRobotRecorder(
    root="datasets/lift", repo_id="me/so101_lift", fps=round(1 / env.unwrapped.step_dt),
    cameras={"camera_ego_rgb": "observation.images.wrist"}, image_shapes=camera_shapes(env),
) as recorder:
    obs, _ = env.reset()
    for _ in range(200):
        snapshot = recorder.snapshot_observation(obs)  # the env reuses its buffers
        obs, *_ = env.step(policy(obs))
        recorder.add_transition(snapshot, joint_targets(env), task="Lift the cube.")
    recorder.save_episode()  # or discard_episode(); close() drops whatever is pending
```

`resume=True` appends to an existing dataset (same fps and features), `overwrite=True` replaces it.

## Isaac Lab (without Arena)

`isaaclab_so101.assets` imports only Isaac Lab. Like `isaaclab_assets`, import it after the simulation app starts:

```python
from isaaclab_so101.assets import SO101_CFG  # also SO101_HIGH_PD_CFG (for IK), SO101_WRIST_CAMERA_CFG

robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

The arm faces +X: `init_state` yaws its base 90°, so commands sampled in the base frame reach forward along -Y.
[`examples/isaaclab/`](https://github.com/art-e-fact/isaaclab-so101/tree/main/examples/isaaclab) sweeps the joints with
the scene API and trains the arm in Isaac Lab's reach task with Isaac Lab's own `isaaclab train`. Tested with Isaac Lab
3.0.0rc1 on PhysX. Newton can't load the USD yet: it carries its own world joint, and Newton won't merge it with the one
`fix_root_link` adds.

## Embodiments

| Name | Actions |
|------|---------|
| `so101_abs_joint` | Absolute joint positions (leader) |
| `so101_rel_joint` | Relative joint positions (for policies; no teleop device pairing) |
| `so101_ik` | Relative SE(3) differential IK + binary Jaw (keyboard / gamepad / spacemouse) |
| `so101_abs_ik` | Absolute TCP pose (position + quaternion, base frame) through the same IK + binary Jaw ([natural gamepad](#natural-gamepad-layout-so101_abs_ik--so101_gamepad)) |

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
`so101_abs_ik` is the same IK fed absolute poses, `(x, y, z, qx, qy, qz, qw)` in the base frame, for devices that
hold a target instead of streaming deltas.

## cuRobo planning assets

The package ships a cuRobo v0.8 robot config, `so101.yml`, and the URDF it refers to (exported from the workshop
USD), under `embodiments/data/curobo/`. Load it with `robot_cfg()`, which turns the YAML's relative `urdf_path`
into an absolute one (cuRobo would look for a relative path under its own assets):

```python
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

from isaaclab_so101.curobo import robot_cfg

planner = MotionPlanner(MotionPlannerCfg.create(robot=robot_cfg()))  # plans target `tcp`, the point between the jaw tips
```

| File | Purpose |
|------|---------|
| `embodiments/data/curobo/so101.yml` | cuRobo `robot_cfg`: collision spheres, self-collision ignore matrix, locked Jaw, home pose; plans target `tcp`, objects attach at `tcp` |
| `embodiments/data/curobo/urdf/SO-ARM101-USD.urdf` | Kinematics matching the sim joint names, plus the fixed `tcp` link at `TCP_OFFSET` |
| `embodiments/data/curobo/meshes/` | Link meshes (41 MB), not shipped: only the generator needs them |

**Supported cuRobo: 0.8.** `so101.yml` and `robot_cfg()` target the 0.8 API (`curobo.motion_planner`), which is
what the shape-sorting demo plans with. Isaac Lab's `isaaclab_mimic` planner and Arena's `isaaclab_arena_curobo`
placement-reachability check still pin cuRobo 0.7.7 (`ebb7170`): 0.8.0 removed every module they import, and
0.7.7's loader rejects this YAML (`tool_frames` instead of `ee_link`, `format_version`), so no environment can hold
both. SO-101 therefore registers no `CuroboEmbodimentCfg`, and Arena's `ik_reachable` placement check is unsupported
until Arena moves to 0.8. The YAML's `isaaclab_so101:` block already carries what that registration needs
(`ee_link_name`, the gripper joint with its open and closed positions, `hand_link_names`).

### Regenerating (maintainers)

`generate_curobo_config` converts the USD to URDF (Isaac Sim), fits collision spheres (cuRobo, CUDA) and writes the
YAML. It never writes into the installed package: the default output is `~/.cache/isaaclab_so101/curobo`, and a
checkout refreshes the shipped files with `--output-dir` (`meshes/` stays untracked):

```bash
# Inside the Isaac Sim / Arena env (needs CUDA + nvidia-curobo)
python -m isaaclab_so101.generate_curobo_config --headless --output-dir src/isaaclab_so101/embodiments/data/curobo
```

Rebuild only the spheres from the URDF and meshes already in the output directory, without Isaac Sim (still needs
CUDA + nvidia-curobo, and `usd-core` to read the authored spheres):

```bash
python -m isaaclab_so101.generate_curobo_config --skip-usd-convert --output-dir src/isaaclab_so101/embodiments/data/curobo
```

For a URDF elsewhere, pass `--urdf <file>` and `--asset-path <mesh dir>`. Add `--visualize` to inspect the fitted
spheres in Viser.

### Authoring collision spheres

The auto-fit covers the arm well but not the thin jaws, so `embodiments/data/curobo_sphere_colliders.usda` carries
hand-placed spheres for the `gripper` and `jaw` links, and the generator uses them instead of the fit for those
links. To edit them, open the USDA in Isaac Sim (it references the robot USD next to it), and under the link's prim
(`over "jaw"`, `over "gripper"`) add or move `Sphere` prims whose name contains `curobo_collider_sphere`; the
generator reads each sphere's `radius` and `xformOp:translate` (link-local, metres) with `pxr` alone. A link with
at least one authored sphere keeps only the authored ones. Then regenerate as above (`--skip-usd-convert` is enough).

## Teleop devices

### Natural gamepad layout (`so101_abs_ik` + `so101_gamepad`)

The sticks drive the arm in its own coordinates; the gripper's heading, and its lean when it must, take care of
themselves.

| Input | Moves |
|-------|-------|
| Left stick up/down | `reach`: fingertips out along the arm / back in |
| Left stick left/right | `pan`: swing the arm left / right (the fingertips move sideways at the same speed at any reach) |
| RT / LT | fingertips up / down |
| Right stick up/down | `tilt`: lean the gripper along the arm (up: fingertips swing away from the base) |
| Right stick left/right | `roll`: spin the gripper about its own axis (right: clockwise seen from above, jaws pointing down) |
| X | `Jaw` toggle open / close |
| LB (held) | slow: a quarter of the speed |
| D-pad up / down | speed ×1.25 / ÷1.25, between ×0.25 and ×2 |
| Back, or keyboard R | reset the episode: the target returns to the home TCP and the jaw opens |

Speeds are `pos_delta_scale` (m/step at the fingertips, default `0.004`) and `delta_scale` (rad/step, default `0.03`)
on `SO101GamepadCfg`; `SO101NaturalGamepadCfg.signs` flips an axis.

**Only what the arm can do.** The device holds a target `(pan, reach, z, tilt, roll)` and moves it one axis at a
time, keeping a step only if `isaaclab_so101.ee_pose.natural_ik`, a closed-form solution of the five joints checked
against their limits, can hold the result. A stick held past the arm's reach stops the target at the edge while the
other axes keep moving, instead of running it away into poses the IK can only fight over (which is what made the arm
oscillate). The edge sits 2 cm inside full stretch (the last centimetres of reach take the last 46° of elbow bend,
where the differential IK is sluggish and can fold the elbow the wrong way, after which it does not recover) and
keeps the wrist axis 3 cm above the table, so the wrist and gripper housings clear it with the jaws horizontal. One more thing takes care of itself: the SO-101's wrist pitch runs out early (with the jaws pointing
straight down the fingertips cannot rise above about 12 cm), so when a position step is impossible at the current
tilt the device leans the gripper a step forward, or back, to make it possible. Lift from a vertical grasp and the
gripper leans as it must; while a position stick pushes past what the tilt allows, that lean wins over the tilt
stick. When an axis has nowhere left to go the terminal says so, once per push.

**Why "natural".** A six-joint arm can put its tool anywhere in its workspace, pointing any way. The SO-101 has five
joints, so one of those six freedoms is missing, and it is a specific one. The shoulder pan swings a vertical plane
and the rest of the arm folds within it, the gripper's own axis included. Seen from above, the gripper therefore
always faces away from the pan axis, out along the arm, wherever the fingertips are: its heading is decided by
*where* it is. Ask for any other heading and the IK can only compromise.

So this layout never asks. You pan the arm, reach out along it and lift; the device works out the one heading the
arm can hold there, and two things stay free: `tilt`, how far the gripper leans up or down within the arm plane (0 is
pointing straight down, negative swings the fingertips away from the base), and `roll`, how it spins about its own
axis. Move the fingertips sideways and the gripper turns with the arm by itself, like a desk lamp whose head
follows the arm: you only lean it and turn it. Every target is one the arm can take, so the IK settles instead of
fighting an impossible orientation.

![Top view: the gripper faces out along the arm and turns by itself when moved sideways. Side view: z, tilt and roll.](https://raw.githubusercontent.com/art-e-fact/isaaclab-so101/main/docs/natural_control.svg)

In numbers: `R = Rz(azimuth) · Ry(tilt) · Rz(roll)`, where the azimuth is the direction of the fingertips seen from
the pan axis (`PAN_AXIS_XY`). `isaaclab_so101.ee_pose.natural_ee_quat_xyzw(x, y, tilt, roll)` returns it as a
quaternion without Isaac Sim, and the [arena-shape-sorting](https://github.com/art-e-fact/arena-shape-sorting/)
cuRobo policy builds its grasp and insertion poses the same way. One approximation: the TCP sits 7 mm off the roll
axis, which skews the heading by under 3° beyond 15 cm from the pan axis. The IK absorbs it.

The device integrates the sticks into a held target `(pan, reach, z, tilt, roll)`, starting at the home TCP
(`HOME_NATURAL_POSE`), and emits `(x, y, z, qx, qy, qz, qw, jaw)` in the base frame. It pairs with `so101_abs_ik`
because relative IK adds each delta to where the arm actually is, so a held target would drift as the arm falls
short of it; absolute commands pull it back every step. A reset returns the target to the home TCP, so an arm
started with `initial_joint_pose` is pulled there on the first step. The embodiment's reset event re-seeds the
target as well: when the environment resets itself on task success, the arm does not snap home and lunge back to a
stale target.

`so101_gamepad` with `so101_ik` uses Isaac Lab's SE(3) gamepad layout, which asks for a yaw the arm cannot give.
The device was named `gamepad` before; pass `--teleop_device so101_gamepad` now. The joint-space gamepad layout
(`so101_abs_joint` + `so101_gamepad`) is gone; the natural layout replaces it.

### Leader arm (`so101_abs_joint` + `so101_leader`)

`so101_leader` emits a (6,) absolute joint vector, so it pairs only with `so101_abs_joint`. Also works with
Arena's `record_demos.py` when the env wires `--teleop_device so101_leader`.

`SO101LeaderCfg` options: `port`, `leader_id`, `leader_recalibrate`, `calibration_dir`
(default `~/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/`, file `<leader_id>.json`)
`num_read_retries` and `max_consecutive_read_failures`. A failed bus read holds the last pose
instead of ending the session; it raises after `max_consecutive_read_failures` (default 10) in a row.

## Compatibility

Tested combinations. Arena has no releases yet, so its rows name commits.

| Arena | Isaac Lab | Isaac Sim | lerobot | cuRobo | Checked by |
|-------|-----------|-----------|---------|--------|------------|
| none | 3.0.0rc1 | 6.1.0.0 | | | [`examples/isaaclab`](https://github.com/art-e-fact/isaaclab-so101/tree/main/examples/isaaclab): joint sweep and SO101-Reach training, PhysX |
| `main` aa36f19 | bb0c8e1, Arena's submodule | 6.1.0.0 | | | [`examples/arena`](https://github.com/art-e-fact/isaaclab-so101/tree/main/examples/arena) smoke test |
| aa36f19 | bb0c8e1 | 6.1.0.0 | 0.6.1 | 0.8, at 8e734f3 | [arena-shape-sorting](https://github.com/art-e-fact/arena-shape-sorting/): headless policy runs and dataset generation |

cuRobo 0.7.7, which Isaac Lab's `isaaclab_mimic` and Arena's reachability check pin, cannot load the shipped config
(see [cuRobo planning assets](#curobo-planning-assets)). Newton cannot load the USD yet
(see [Isaac Lab](#isaac-lab-without-arena)).

## Development

The tests run without Isaac Sim: Isaac-dependent modules are imported against stubs (the `isaac`
fixture in `tests/conftest.py`), with real `torch` and `numpy`.

```bash
uv venv --python 3.12
uv pip install --torch-backend cpu --group dev -e .  # -e ".[lerobot,leader]" also runs the recorder test
.venv/bin/pytest && .venv/bin/ruff check
```

CI runs both variants on every pull request. Nothing in CI starts Isaac Sim.

### Releasing

[`publish.yml`](https://github.com/art-e-fact/isaaclab-so101/blob/main/.github/workflows/publish.yml) builds the wheel
and sdist on every pull request, checks them with `twine check --strict`, and imports the wheel in a clean venv.
Publishing a GitHub release uploads them to PyPI; running the workflow by hand uploads them to TestPyPI. Both use
Trusted Publishing, so no token is stored. To release:

1. Set `version` in `pyproject.toml` and move the changes under a new heading in `CHANGELOG.md`.
2. Merge, then publish a GitHub release tagged `v<version>`. The workflow refuses a tag that doesn't match.

## Acknowledgments

We used the SO-101 USD model from the [Sim-to-Real-SO-101-Workshop](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/art-e-fact/isaaclab-so101/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/art-e-fact/isaaclab-so101/blob/main/LICENSE-MIT))

at your option.

Exception: `src/isaaclab_so101/embodiments/data/SO-ARM101-USD.usd` is Copyright NVIDIA Corporation & Affiliates, from
the [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop), and is licensed under
Apache-2.0 only. [NOTICE](https://github.com/art-e-fact/isaaclab-so101/blob/main/NOTICE) lists it with the files
generated from it, and the wheel carries NOTICE and both licenses.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
