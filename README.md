# SO-101 Embodiment for IsaacLab-Arena

:construction: Under development. `main` is tracking the `main` branch of [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena).

SO-101 follower embodiment (and optional leader-arm helpers) for
[IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena).

Environments using this embodiment:
- [Arena Shape Sorting](https://github.com/art-e-fact/arena-shape-sorting/)

Planned features:
 - More natural teleop setup with gamepad and keyboard.
 - Isaac Lab support.

## Install

First, install [IsaacLab-Arena](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/quickstart/installation.html)

```bash
uv add git+https://github.com/art-e-fact/isaaclab-so101.git
# optional: for leader arm teleop
uv add git+https://github.com/art-e-fact/isaaclab-so101.git#egg=arena-so101[leader]
```

After `SimulationApp` is running, register once:

```python
import arena_so101
arena_so101.register()  # so101_abs_joint, so101_rel_joint, so101_ik, so101_leader, gamepad
```

Then use like any Arena embodiment:

```python
embodiment = asset_registry.get_asset_by_name("so101_abs_joint")(enable_cameras=True)
```

See [example usage in an IsaacLab-Arena environment](https://github.com/art-e-fact/arena-shape-sorting/blob/25ea6bfea43a5134570e924fb3cdacb663f59472/arena_envs/src/shape_sorting/shape_sorting_env.py#L131).

See the [IsaacLab-Arena documentation](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/concepts/embodiment/index.html) for more details.

> TODO: Add a simple example environment that uses this embodiment.
> TODO: Document the camera configuration.

## Embodiments

| Name | Actions |
|------|---------|
| `so101_abs_joint` | Absolute joint positions (leader + joint-space gamepad) |
| `so101_rel_joint` | Relative joint positions |
| `so101_ik` | Relative SE(3) differential IK + binary Jaw (keyboard / gamepad / spacemouse) |

USD joints: `Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`.
The robot USD comes from the [Sim-to-Real-SO-101-Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop).


`so101_ik` is a 5-DOF arm: DLS tracks EE position and does best-effort orientation on the 6D pose command.

## cuRobo planning assets

Generate a URDF (from the workshop USD) plus a cuRobo robot YAML (collision spheres,
self-collision ignore matrix, locked Jaw, home pose) under
`embodiments/data/curobo/`:

```bash
# Inside the Isaac Sim / Arena env (needs CUDA + nvidia-curobo)
python -m arena_so101.generate_curobo_config --headless
```

> TODO: Document manually authoring collision spheres.

Outputs:

| File | Purpose |
|------|---------|
| `embodiments/data/curobo/urdf/SO-ARM101-USD.urdf` | Kinematics matching sim joint names |
| `embodiments/data/curobo/meshes/` | Link meshes referenced by the URDF |
| `embodiments/data/curobo/so101.yml` | cuRobo `robot_cfg` for `MotionPlanner` |

Rebuild spheres from an existing URDF (no Isaac Sim):

```bash
python -m arena_so101.generate_curobo_config \
  --skip-usd-convert \
  --urdf arena_so101/src/arena_so101/embodiments/data/curobo/urdf/SO-ARM101-USD.urdf \
  --asset-path arena_so101/src/arena_so101/embodiments/data/curobo/meshes
```

Add `--visualize` to inspect fitted spheres in Viser.

### Joint-space gamepad layout (`so101_abs_joint` + `gamepad`)

Sticks/triggers integrate into a held absolute joint target. Releasing sticks holds pose.

| Input | Joint |
|-------|-------|
| RT (+) / LT (−) | `Rotation` (base yaw) |
| Left stick up/down | `Pitch` |
| Left stick left/right | `Elbow` |
| Right stick up/down | `Wrist_Pitch` |
| Right stick right/left | `Wrist_Roll` |
| X | `Jaw` toggle open / close (absolute limits) |

Speed is `delta_scale` on `GamepadCfg` (default `0.03` rad/step at full deflection).

`so101_leader` emits a (6,) absolute joint vector for `so101_abs_joint`. Also works with
Arena's `record_demos.py` when the env wires `--teleop_device so101_leader`.

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