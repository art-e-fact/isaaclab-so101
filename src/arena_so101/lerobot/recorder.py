"""Record SO-101 rollouts from a manager-based env directly as LeRobot v3 datasets.

A frame pairs the pre-step ``policy.joint_pos`` (``observation.state``) with the absolute joint targets the sim
received for that step (``action``, see :func:`joint_targets`), so every embodiment records the same portable
action, plus one video per camera in ``cameras`` (sim observation term -> dataset key).
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
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
DEFAULT_IMAGE_SHAPE = (480, 640, 3)  # the embodiment's cameras


def camera_shapes(env) -> dict[str, tuple[int, ...]]:
    """``{"camera_ego_rgb": (480, 640, 3), ...}``: the shape of every ``camera_obs`` term of a manager-based env."""
    manager = env.unwrapped.observation_manager
    return dict(zip(manager.active_terms["camera_obs"], map(tuple, manager.group_obs_term_dim["camera_obs"])))


def joint_targets(env) -> np.ndarray:
    """The absolute joint position targets the sim last received, ``(num_envs, 6)`` in ``SIM_JOINT_NAMES`` order.

    Absolute, relative and IK action terms all end as position targets, so this is the portable ``action`` for
    every embodiment. Returns a CPU copy: the articulation reuses the buffer every step.
    """
    robot = env.unwrapped.scene["robot"]
    joint_ids = robot.find_joints(list(SIM_JOINT_NAMES), preserve_order=True)[0]
    return robot.data.joint_pos_target.torch[:, joint_ids].cpu().numpy()  # fancy indexing copies


def so101_dataset_features(
    cameras: Mapping[str, str] = CAMERA_FEATURES,
    image_shapes: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """LeRobot v3 feature schema: joint state and targets, plus one video per camera (default 480×640 RGB)."""
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
    for sim_key, feature_key in cameras.items():
        features[feature_key] = {
            "dtype": "video",
            "shape": tuple((image_shapes or {}).get(sim_key, DEFAULT_IMAGE_SHAPE)),
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": False},
        }
    return features


class SO101LeRobotRecorder:
    """Manage a success-filtered SO-101 LeRobot recording session.

    ``cameras`` maps sim observation terms (``camera_obs``) to dataset keys; ``image_shapes`` gives their
    ``(height, width, channels)`` per sim term (:func:`camera_shapes` reads them off the env; missing ones default
    to 480×640 RGB). Frames are uint8, or float in [0, 1], as Arena's ``camera_obs`` provides them.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        repo_id: str,
        fps: int,
        cameras: Mapping[str, str] = CAMERA_FEATURES,
        image_shapes: Mapping[str, Sequence[int]] | None = None,
        resume: bool = False,
        overwrite: bool = False,
        streaming_encoding: bool = True,
    ) -> None:
        if resume and overwrite:
            raise ValueError("resume and overwrite are mutually exclusive")

        try:
            from lerobot.datasets import LeRobotDataset, VideoEncodingManager
        except ImportError as exc:
            raise ImportError(
                "SO101LeRobotRecorder needs lerobot[dataset]>=0.6.1,<0.7 (the arena-so101 `lerobot` extra; "
                "the `leader` extra alone does not include it)."
            ) from exc

        self.root = Path(root).expanduser().resolve()
        self.cameras = dict(cameras)
        self.features = so101_dataset_features(self.cameras, image_shapes)
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
        action: Any,
        *,
        task: str,
    ) -> None:
        """Buffer one pre-step observation with the absolute joint targets applied after it (:func:`joint_targets`)."""
        if self._closed:
            raise RuntimeError("Cannot record after the recorder has been closed")
        if not task:
            raise ValueError("LeRobot frames require a non-empty task description")

        frame = self._frame(observation)
        frame[ACTION_KEY] = self._joint_vector(action, ACTION_KEY)
        frame["task"] = task
        self.dataset.add_frame(frame)

    def snapshot_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Copy the pre-step state and camera buffers (the env reuses them) to CPU memory, in the same layout."""
        frame = self._frame(observation)
        return {
            "policy": {"joint_pos": frame[STATE_KEY]},
            "camera_obs": {sim_key: frame[feature_key] for sim_key, feature_key in self.cameras.items()},
        }

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

    def _frame(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """The state and the recorded cameras of one observation, converted and copied for the dataset."""
        try:
            state = observation["policy"]["joint_pos"]
        except KeyError as exc:
            raise KeyError(f"Missing recording observation key: {exc}") from exc
        camera_obs = observation.get("camera_obs", {})
        missing = self.cameras.keys() - camera_obs.keys()
        if missing:
            raise KeyError(f"Missing camera observations {sorted(missing)}; available: {sorted(camera_obs)}")
        frame = {STATE_KEY: self._joint_vector(state, STATE_KEY)}
        for sim_key, feature_key in self.cameras.items():
            frame[feature_key] = self._rgb_frame(camera_obs[sim_key], feature_key)
        return frame

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

    def _rgb_frame(self, value: Any, feature_key: str) -> np.ndarray:
        array = self._as_numpy(value)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        expected_shape = self.features[feature_key]["shape"]
        if array.shape != expected_shape:
            raise ValueError(
                f"{feature_key} must be an HWC frame of shape {expected_shape}, got {array.shape} "
                "(pass image_shapes=camera_shapes(env))"
            )
        if array.dtype == np.uint8:
            return np.ascontiguousarray(array)
        if np.issubdtype(array.dtype, np.floating) and 0.0 <= array.min() and array.max() <= 1.0:
            return np.ascontiguousarray((array * 255.0).round().astype(np.uint8))
        raise TypeError(f"{feature_key} must be uint8 or float in [0, 1] (as camera_obs gives it), got {array.dtype}")
