"""Record SO-101 simulation rollouts directly as LeRobot v3 datasets."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np

from arena_so101.mapping import SIM_JOINT_NAMES

STATE_KEY = "observation.state"
ACTION_KEY = "action"
CAMERA_FEATURES = {
    "camera_ego_rgb": "observation.images.ego_view",
    "external_camera_rgb": "observation.images.exterior_image",
}


def so101_dataset_features(
    image_shape: tuple[int, int, int] = (480, 640, 3),
) -> dict[str, dict[str, Any]]:
    """Return the LeRobot v3 feature schema for SO-101 shape-sorting rollouts."""
    joint_names = list(SIM_JOINT_NAMES)
    features: dict[str, dict[str, Any]] = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": joint_names,
        },
    }
    for feature_key in CAMERA_FEATURES.values():
        features[feature_key] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
    return features


class SO101LeRobotRecorder:
    """Manage a success-filtered SO-101 LeRobot recording session."""

    def __init__(
        self,
        *,
        root: str | Path,
        repo_id: str,
        fps: int,
        resume: bool = False,
        overwrite: bool = False,
        streaming_encoding: bool = True,
        image_shape: tuple[int, int, int] = (480, 640, 3),
    ) -> None:
        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")

        from lerobot.datasets import LeRobotDataset, VideoEncodingManager

        self.root = Path(root).expanduser().resolve()
        self.features = so101_dataset_features(image_shape)
        self._closed = False

        if overwrite and self.root.exists():
            if self.root == Path(self.root.anchor):
                raise ValueError(f"Refusing to overwrite filesystem root: {self.root}")
            shutil.rmtree(self.root)

        if resume:
            if not self.root.exists():
                raise FileNotFoundError(
                    f"Cannot resume missing LeRobot dataset: {self.root}"
                )
            self.dataset = LeRobotDataset.resume(
                repo_id,
                root=self.root,
                streaming_encoding=streaming_encoding,
            )
            try:
                self._validate_resumed_dataset(repo_id=repo_id, fps=fps)
            except Exception:
                self.dataset.finalize()
                raise
        else:
            if self.root.exists():
                raise FileExistsError(
                    f"LeRobot dataset already exists: {self.root}. "
                    "Use --resume to append or --overwrite to replace it."
                )
            self.dataset = LeRobotDataset.create(
                repo_id,
                fps=fps,
                root=self.root,
                robot_type="so101",
                features=self.features,
                use_videos=True,
                streaming_encoding=streaming_encoding,
            )

        self._video_manager = VideoEncodingManager(self.dataset)
        self._video_manager.__enter__()

    @property
    def num_episodes(self) -> int:
        """Return the number of committed episodes."""
        return self.dataset.num_episodes

    def add_transition(
        self,
        observation: dict[str, Any],
        processed_action: Any,
        *,
        task: str,
    ) -> None:
        """Buffer one pre-action observation paired with its applied action."""
        if self._closed:
            raise RuntimeError("Cannot record after the recorder has been closed")
        if not task:
            raise ValueError("LeRobot frames require a non-empty task description")

        try:
            policy_obs = observation["policy"]
            camera_obs = observation["camera_obs"]
            state = policy_obs["joint_pos"]
        except KeyError as exc:
            raise KeyError(f"Missing recording observation key: {exc}") from exc

        frame: dict[str, Any] = {
            STATE_KEY: self._joint_vector(state, STATE_KEY),
            ACTION_KEY: self._joint_vector(processed_action, ACTION_KEY),
            "task": task,
        }
        for sim_key, feature_key in CAMERA_FEATURES.items():
            if sim_key not in camera_obs:
                raise KeyError(
                    f"Missing camera observation {sim_key!r}; available: {sorted(camera_obs)}"
                )
            frame[feature_key] = self._rgb_frame(camera_obs[sim_key], feature_key)

        self.dataset.add_frame(frame)

    def snapshot_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Copy the mutable pre-step state and camera buffers to CPU memory."""
        try:
            policy_obs = observation["policy"]
            camera_obs = observation["camera_obs"]
            state = policy_obs["joint_pos"]
        except KeyError as exc:
            raise KeyError(f"Missing recording observation key: {exc}") from exc

        snapshot = {
            "policy": {"joint_pos": self._joint_vector(state, STATE_KEY)},
            "camera_obs": {},
        }
        for sim_key, feature_key in CAMERA_FEATURES.items():
            if sim_key not in camera_obs:
                raise KeyError(
                    f"Missing camera observation {sim_key!r}; available: {sorted(camera_obs)}"
                )
            snapshot["camera_obs"][sim_key] = self._rgb_frame(
                camera_obs[sim_key], feature_key
            )
        return snapshot

    def save_episode(self) -> None:
        """Commit the current successful episode."""
        if not self.dataset.has_pending_frames():
            raise RuntimeError("Cannot save an episode with no recorded frames")
        self.dataset.save_episode()

    def discard_episode(self) -> None:
        """Discard the current failed or interrupted episode."""
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def close(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Discard an unfinished episode and finalize all committed data."""
        if self._closed:
            return
        try:
            self.discard_episode()
        finally:
            self._video_manager.__exit__(exc_type, exc_value, traceback)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close(exc_type, exc_value, traceback)
        return False

    def _validate_resumed_dataset(self, *, repo_id: str, fps: int) -> None:
        if self.dataset.repo_id != repo_id:
            raise ValueError(
                f"Dataset repo_id mismatch: existing={self.dataset.repo_id!r}, requested={repo_id!r}"
            )
        if self.dataset.fps != fps:
            raise ValueError(
                f"Dataset FPS mismatch: existing={self.dataset.fps}, requested={fps}"
            )

        for key, expected in self.features.items():
            actual = self.dataset.features.get(key)
            if actual is None:
                raise ValueError(f"Existing dataset is missing feature {key!r}")
            if (
                actual["dtype"] != expected["dtype"]
                or tuple(actual["shape"]) != tuple(expected["shape"])
                or actual.get("names") != expected.get("names")
            ):
                raise ValueError(
                    f"Existing dataset feature {key!r} is incompatible: "
                    f"existing={actual}, expected={expected}"
                )

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    @classmethod
    def _joint_vector(cls, value: Any, feature_key: str) -> np.ndarray:
        array = cls._as_numpy(value)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        expected_shape = (len(SIM_JOINT_NAMES),)
        if array.shape != expected_shape:
            raise ValueError(
                f"{feature_key} must have shape {expected_shape}, got {array.shape}"
            )
        return np.ascontiguousarray(array, dtype=np.float32)

    @classmethod
    def _rgb_frame(cls, value: Any, feature_key: str) -> np.ndarray:
        array = cls._as_numpy(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"{feature_key} must be an HWC RGB frame, got {array.shape}"
            )
        if array.dtype != np.uint8:
            if (
                np.issubdtype(array.dtype, np.floating)
                and array.size
                and array.max() <= 1.0
            ):
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)
