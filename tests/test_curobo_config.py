from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from arena_so101 import JAW_OPEN_RAD, TCP_OFFSET
from arena_so101.generate_curobo_config import add_tcp_link, patch_so101_robot_yaml

URDF = """<robot name="so101"><link name="base"/><link name="gripper"/>
<joint name="Wrist_Roll" type="revolute"><parent link="base"/><child link="gripper"/></joint></robot>"""


def test_add_tcp_link_hangs_off_the_ee_link_once(tmp_path):
    urdf = tmp_path / "so101.urdf"
    urdf.write_text(URDF)

    assert add_tcp_link(urdf, parent="gripper")
    assert not add_tcp_link(urdf, parent="gripper")  # already there

    root = ET.parse(urdf).getroot()
    assert root.find("link[@name='tcp']") is not None
    joint = root.find("joint[@name='tcp_joint']")
    assert joint.get("type") == "fixed"
    assert (joint.find("parent").get("link"), joint.find("child").get("link")) == ("gripper", "tcp")
    assert [float(v) for v in joint.find("origin").get("xyz").split()] == list(TCP_OFFSET)


def test_patch_plans_for_the_tcp_and_attaches_at_the_ee_link(tmp_path):
    yaml = pytest.importorskip("yaml")
    raw = tmp_path / "raw.yml"
    raw.write_text(yaml.safe_dump({"kinematics": {"cspace": {"joint_names": ["Rotation", "Jaw"]}}}))

    out = patch_so101_robot_yaml(
        raw, tmp_path / "so101.yml", urdf_path=tmp_path / "so101.urdf", asset_path=tmp_path,
        tool_frame="tcp", ee_link="gripper", jaw_joint="Jaw",
    )

    data = yaml.safe_load(out.read_text())
    kin = data["robot_cfg"]["kinematics"]
    assert kin["tool_frames"] == ["tcp"]
    assert kin["extra_links"]["attached_object"]["parent_link_name"] == "tcp"  # cuRobo attaches through tool_frames[0]
    assert "attached_object" in kin["self_collision_ignore"]["gripper"]
    assert data["arena_so101"]["ee_link_name"] == "tcp"
    assert data["arena_so101"]["hand_link_names"] == ["gripper"]


def test_patch_writes_paths_relative_to_the_yaml_and_robot_cfg_resolves_them(tmp_path):
    yaml = pytest.importorskip("yaml")
    from arena_so101.curobo import robot_cfg

    raw = tmp_path / "raw.yml"
    raw.write_text(yaml.safe_dump({"kinematics": {}}))
    out = patch_so101_robot_yaml(
        raw, tmp_path / "out" / "so101.yml", urdf_path=tmp_path / "out" / "urdf" / "so101.urdf",
        asset_path=tmp_path / "out" / "meshes", tool_frame="tcp", ee_link="gripper", jaw_joint="Jaw",
    )

    kin = yaml.safe_load(out.read_text())["robot_cfg"]["kinematics"]
    assert (kin["urdf_path"], kin["asset_root_path"]) == ("urdf/so101.urdf", "meshes")  # relocatable
    resolved = robot_cfg(out)["robot_cfg"]["kinematics"]
    assert resolved["urdf_path"] == str(tmp_path / "out" / "urdf" / "so101.urdf")
    assert resolved["asset_root_path"] == str(tmp_path / "out" / "meshes")


def test_shipped_config_plans_for_the_tcp_of_the_shipped_urdf():
    pytest.importorskip("yaml")
    from arena_so101 import CUROBO_ROBOT_YML
    from arena_so101.curobo import robot_cfg

    kin = robot_cfg()["robot_cfg"]["kinematics"]
    urdf = Path(kin["urdf_path"])
    assert urdf.is_file() and urdf.parent.parent == CUROBO_ROBOT_YML.parent  # shipped next to the YAML
    assert kin["tool_frames"] == ["tcp"] and kin["lock_joints"] == {"Jaw": JAW_OPEN_RAD}
    root = ET.parse(urdf).getroot()
    assert root.find("joint[@name='tcp_joint']/parent").get("link") == "gripper"
    assert set(kin["collision_spheres"]) >= {"gripper", "jaw"}  # authored spheres made it in
