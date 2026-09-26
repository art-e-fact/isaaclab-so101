from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from arena_so101.lerobot.recorder import (
    ACTION_KEY,
    CAMERA_FEATURES,
    STATE_KEY,
    SO101LeRobotRecorder,
    camera_shapes,
    joint_targets,
    so101_dataset_features,
)


class _FakeDataset:
    def __init__(self, *, root: Path, repo_id: str, fps: int, features: dict) -> None:
        self.root = root
        self.repo_id = repo_id
        self.fps = fps
        self.features = features
        self.num_episodes = 0
        self.frames = []
        self.finalized = False

    @classmethod
    def create(cls, repo_id, fps, *, root, features, **_kwargs):
        return cls(root=root, repo_id=repo_id, fps=fps, features=features)

    @classmethod
    def resume(cls, repo_id, *, root, **_kwargs):
        return cls(
            root=root, repo_id=repo_id, fps=50, features=so101_dataset_features()
        )

    def add_frame(self, frame):
        self.frames.append(frame)

    def has_pending_frames(self):
        return bool(self.frames)

    def save_episode(self):
        self.num_episodes += 1
        self.frames.clear()

    def clear_episode_buffer(self):
        self.frames.clear()

    def finalize(self):
        self.finalized = True


class _FakeVideoEncodingManager:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.dataset.finalize()


@pytest.fixture
def fake_lerobot(monkeypatch):
    datasets_module = ModuleType("lerobot.datasets")
    datasets_module.LeRobotDataset = _FakeDataset
    datasets_module.VideoEncodingManager = _FakeVideoEncodingManager
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_module)


def _observation(image_shape=(480, 640, 3), cameras=CAMERA_FEATURES, fill=0.5):
    """Float frames in [0, 1], as Arena's ``camera_obs`` gives them with ``normalize=True``."""
    return {
        "policy": {"joint_pos": np.arange(6, dtype=np.float64)[None]},
        "camera_obs": {key: np.full((1, *image_shape), fill, dtype=np.float32) for key in cameras},
    }


def test_feature_schema_uses_so101_joint_order():
    features = so101_dataset_features()

    assert features[STATE_KEY]["names"] == [
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
        "Jaw",
    ]
    assert features[ACTION_KEY]["shape"] == (6,)
    assert set(CAMERA_FEATURES.values()) <= features.keys()


def test_feature_schema_takes_any_camera_set():
    features = so101_dataset_features({"wrist_rgb": "observation.images.wrist"}, {"wrist_rgb": (240, 320, 3)})

    assert [key for key in features if key.startswith("observation.images")] == ["observation.images.wrist"]
    assert features["observation.images.wrist"]["shape"] == (240, 320, 3)


def test_records_and_commits_normalized_transition(tmp_path, fake_lerobot):
    recorder = SO101LeRobotRecorder(
        root=tmp_path / "dataset",
        repo_id="local/test",
        fps=50,
    )

    snapshot = recorder.snapshot_observation(_observation())
    recorder.add_transition(
        snapshot,
        np.arange(6, dtype=np.float64)[None],
        task="Sort the shapes.",
    )

    frame = recorder.dataset.frames[0]
    assert frame[STATE_KEY].dtype == np.float32
    assert frame[ACTION_KEY].shape == (6,)
    for feature_key in CAMERA_FEATURES.values():
        assert frame[feature_key].dtype == np.uint8
        assert frame[feature_key].shape == (480, 640, 3)
        assert frame[feature_key][0, 0, 0] == 128  # 0.5 of the [0, 1] range

    recorder.save_episode()
    recorder.close()

    assert recorder.num_episodes == 1
    assert recorder.dataset.finalized


def test_records_a_custom_camera_set_and_ignores_the_others(tmp_path, fake_lerobot):
    cameras = {"camera_ego_rgb": "observation.images.wrist"}
    recorder = SO101LeRobotRecorder(
        root=tmp_path / "dataset", repo_id="local/test", fps=50, cameras=cameras, image_shapes={"camera_ego_rgb": (8, 8, 3)}
    )
    observation = _observation((8, 8, 3))  # both cameras present; only one recorded
    observation["camera_obs"]["camera_ego_rgb"] = np.full((1, 8, 8, 3), 7, dtype=np.uint8)

    recorder.add_transition(recorder.snapshot_observation(observation), np.zeros(6), task="Lift.")

    frame = recorder.dataset.frames[0]
    assert set(frame) == {STATE_KEY, ACTION_KEY, "observation.images.wrist", "task"}
    assert frame["observation.images.wrist"].shape == (8, 8, 3) and frame["observation.images.wrist"][0, 0, 0] == 7
    with pytest.raises(KeyError, match="camera_ego_rgb"):
        recorder.snapshot_observation({"policy": {"joint_pos": np.zeros(6)}})


def test_rejects_frames_that_are_not_uint8_or_unit_floats(tmp_path, fake_lerobot):
    recorder = SO101LeRobotRecorder(root=tmp_path / "dataset", repo_id="local/test", fps=50)

    with pytest.raises(TypeError, match=r"\[0, 1\]"):
        recorder.snapshot_observation(_observation(fill=200.0))  # float in the 0-255 range: no guessing
    with pytest.raises(ValueError, match="image_shapes"):
        recorder.snapshot_observation(_observation((240, 320, 3)))


def test_joint_targets_and_camera_shapes_read_the_env():
    torch = pytest.importorskip("torch")
    articulation_order = ["Jaw", "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]  # not the sim order
    targets = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    robot = SimpleNamespace(
        find_joints=lambda names, preserve_order: ([articulation_order.index(n) for n in names], list(names)),
        data=SimpleNamespace(joint_pos_target=SimpleNamespace(torch=targets)),
    )
    manager = SimpleNamespace(
        active_terms={"camera_obs": ["camera_ego_rgb", "external_camera_rgb"]},
        group_obs_term_dim={"camera_obs": [(480, 640, 3), (240, 320, 3)]},
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": robot}, observation_manager=manager))

    result = joint_targets(env)

    assert result.shape == (2, 6) and result.tolist()[0] == [1, 2, 3, 4, 5, 0]  # Rotation..Wrist_Roll, then Jaw
    targets[0, 0] = -1.0
    assert result[0, 5] == 0.0  # a copy, not a view of the reused buffer
    assert camera_shapes(env) == {"camera_ego_rgb": (480, 640, 3), "external_camera_rgb": (240, 320, 3)}


def test_close_discards_uncommitted_episode(tmp_path, fake_lerobot):
    recorder = SO101LeRobotRecorder(
        root=tmp_path / "dataset",
        repo_id="local/test",
        fps=50,
    )
    recorder.add_transition(
        recorder.snapshot_observation(_observation()),
        np.zeros((1, 6), dtype=np.float32),
        task="Sort the shapes.",
    )

    recorder.close()

    assert not recorder.dataset.frames
    assert recorder.num_episodes == 0
    assert recorder.dataset.finalized


def test_missing_lerobot_names_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot.datasets", None)

    with pytest.raises(ImportError, match="lerobot"):
        SO101LeRobotRecorder(root=tmp_path / "dataset", repo_id="local/test", fps=50)


def test_existing_output_requires_explicit_mode(tmp_path, fake_lerobot):
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(FileExistsError, match="--resume.*--overwrite"):
        SO101LeRobotRecorder(root=root, repo_id="local/test", fps=50)


def test_writes_reopenable_lerobot_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_cache"))
    LeRobotDataset = pytest.importorskip("lerobot.datasets").LeRobotDataset

    root = tmp_path / "dataset"
    cameras = {"camera_ego_rgb": "observation.images.wrist"}
    image_shapes = {"camera_ego_rgb": (64, 64, 3)}
    with SO101LeRobotRecorder(
        root=root,
        repo_id="local/test",
        fps=10,
        streaming_encoding=False,
        cameras=cameras,
        image_shapes=image_shapes,
    ) as recorder:
        for _ in range(2):
            recorder.add_transition(
                recorder.snapshot_observation(_observation((64, 64, 3), cameras)),
                np.zeros((1, 6), dtype=np.float32),
                task="Sort the shapes.",
            )
        recorder.save_episode()

    dataset = LeRobotDataset("local/test", root=root)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 2
    assert "observation.images.wrist" in dataset.features
