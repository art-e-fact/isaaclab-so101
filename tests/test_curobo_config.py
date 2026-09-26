from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from arena_so101 import TCP_OFFSET
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
    assert kin["extra_links"]["attached_object"]["parent_link_name"] == "gripper"
    assert "attached_object" in kin["self_collision_ignore"]["gripper"]
    assert data["arena_so101"]["ee_link_name"] == "tcp"
    assert data["arena_so101"]["hand_link_names"] == ["gripper"]
