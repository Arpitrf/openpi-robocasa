"""RoboCasa observation parsing helpers.

Copied verbatim from Arpitrf/semantic_corrections (semantic_corrections/hitl/obs.py, `point-set`
branch @ c5c6cbe). No local fixes needed.
"""

import numpy as np


def obs_to_policy_state(obs) -> np.ndarray:
    """Extract the 16-D proprioceptive state vector sent to pi0."""
    return np.concatenate(
        (
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ),
        axis=0,
    )


def obs_to_camera_frames(obs) -> tuple[np.ndarray, np.ndarray]:
    """Return main and wrist camera frames as contiguous uint8 arrays."""
    main = np.ascontiguousarray(obs["video.robot0_agentview_left"])
    wrist = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
    return main, wrist


def obs_to_display_camera_frames(
    obs,
    *,
    main_camera: str,
    wrist_camera: str = "robot0_eye_in_hand",
) -> tuple[np.ndarray, np.ndarray]:
    """Return custom (or named) main + wrist frames from gym ``video.*`` keys."""
    main_key = f"video.{main_camera}"
    wrist_key = f"video.{wrist_camera}"
    missing = [key for key in (main_key, wrist_key) if key not in obs]
    if missing:
        raise KeyError(
            f"obs missing camera keys {missing}; ensure cameras are enabled in the env"
        )
    main = np.ascontiguousarray(obs[main_key][..., :3], dtype=np.uint8)
    wrist = np.ascontiguousarray(obs[wrist_key][..., :3], dtype=np.uint8)
    return main, wrist