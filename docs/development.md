# Development notes

For maintainers. The tests and the showcase are in the [README](../README.md#development).

## cuRobo assets

### Regenerating

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

## Releasing

[`publish.yml`](../.github/workflows/publish.yml) builds the wheel and sdist on every pull request, checks them with
`twine check --strict`, and imports the wheel in a clean venv. Publishing a GitHub release uploads them to PyPI with
Trusted Publishing, so no token is stored. PyPI never takes the same filename twice, so a broken release is fixed by
the next version, not by a re-upload.

One-time setup:

1. On [PyPI](https://pypi.org/manage/account/publishing/), add a pending publisher: project `isaaclab-so101`, owner
   `art-e-fact`, repository `isaaclab-so101`, workflow `publish.yml`, environment `pypi`.
2. In the repository settings, create the `pypi` environment with required reviewers, and limit its deployments to
   `v*` tags. Otherwise anyone who can push a branch can run the upload.

To release:

1. Set `version` in `pyproject.toml`. In `CHANGELOG.md`, move the changes under a heading for that version and
   replace `Unreleased` with the date.
2. Merge, then publish a GitHub release tagged `v<version>`. The workflow refuses a tag that doesn't match.
