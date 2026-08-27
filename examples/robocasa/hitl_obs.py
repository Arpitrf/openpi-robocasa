"""RoboCasa observation parsing helpers.

Copied verbatim from Arpitrf/semantic_corrections (semantic_corrections/hitl/obs.py, frs branch).
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
