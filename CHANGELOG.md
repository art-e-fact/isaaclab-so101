# Changelog

Notable changes to `isaaclab-so101`. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html): before 1.0, a minor release may break the API.

## [0.1.0] - Unreleased

First release on PyPI.

### Added

- Arena embodiments `so101_abs_joint`, `so101_rel_joint`, `so101_ik` and `so101_abs_ik`, and the teleop devices
  `so101_leader` and `so101_gamepad`, all registered by `isaaclab_so101.register()`.
- The natural gamepad layout: with `so101_abs_ik`, the sticks drive the TCP in the arm's own coordinates, and a
  closed-form IK keeps every target reachable.
- `isaaclab_so101.assets` for plain Isaac Lab without Arena: `SO101_CFG`, `SO101_HIGH_PD_CFG` and
  `SO101_WRIST_CAMERA_CFG`.
- A wrist camera and an external camera attached to the base link.
- Franka-parity observations (`eef_pos`, `eef_quat`, `gripper_pos`), an Arena `ParallelJawGripper`, and the
  `initial_joint_pose` and `reset_joint_noise` options.
- `isaaclab_so101.lerobot`, a LeRobot v3 dataset recorder for any embodiment and camera set (the `lerobot` extra).
- A cuRobo 0.8 robot config and URDF that plan for the TCP between the jaw tips, and the generator that rebuilds them.
- Constants that import without Isaac Sim: joint names and limits, the home pose, jaw targets, the TCP offset, the pan
  axis and the asset paths.
- Examples for Isaac Lab (a joint sweep and SO101-Reach training) and for Arena (a minimal environment with a smoke
  test).

### Changed, for installs from earlier git commits

- The distribution is now `isaaclab-so101` and the module `isaaclab_so101`, renamed from `arena-so101` and
  `arena_so101`. Reinstall editable checkouts.
- The cuRobo YAML's package section is now `isaaclab_so101:`, and the generator writes to
  `~/.cache/isaaclab_so101/curobo` by default.
- The `gamepad` device is now `so101_gamepad`. Its joint-space layout is gone: pair it with `so101_abs_ik`.
- The leader arm no longer pairs with `so101_rel_joint`, where the arm drifted to its limits.
- The `dev` extra is now a `dev` dependency group: `uv pip install --group dev`.
- The license metadata is `(MIT OR Apache-2.0) AND Apache-2.0`, because the shipped NVIDIA SO-101 USD is
  Apache-2.0 only. `NOTICE` carries its attribution.

[0.1.0]: https://github.com/art-e-fact/isaaclab-so101/releases/tag/v0.1.0
