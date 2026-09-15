"""Convert one or more real-DROID-robot HITL demo_0.hdf5 files into a single plain LeRobot
dataset, using `save_episodes_as_lerobot` from lerobot_export.py -- same function
convert_hitl_hdf5_to_lerobot.py uses for RoboCasa's sim HITL hdf5s, just parameterized for this
dataset's shapes (see lerobot_export.py's docstring, change 3).

This is the real-world sibling of convert_hitl_hdf5_to_lerobot.py: same SIRIUS-style
acting_agent/preintv-window handling (imported from that module, not reimplemented), but reading
DROID-native observation/action fields instead of RoboCasa's sim EE-pose obs and 12-dim EE-delta
action space:

  - state: `droid/joint_position` (7) ++ `droid/gripper_position` (1), matching
    `droid_policy.DroidInputs`'s `observation/joint_position` + `observation/gripper_position`
    exactly -- these are the DROID-interface fields a real robot's control loop would feed the
    policy at inference time, not the sim/robomimic-style `obs/robot_joint_pos` /
    `obs/robot0_gripper_qpos` mirrors also present in these hdf5s (verified NOT equal to the
    `droid/*` fields -- up to 0.24 rad apart on joint_position, and robot0_gripper_qpos is a raw
    2-finger encoder pair in meters vs. droid/gripper_position's normalized [0, 1] scalar).
  - actions: the hdf5's top-level `actions` (8-dim), which already equals
    `droid/policy_action_8d` verbatim -- 7 joint-velocity dims + 1 gripper-position-target dim,
    the exact `DroidActionSpace.JOINT_VELOCITY` layout. Used as-is, no delta-action transform
    (unlike RLDSDroidDataConfig's JOINT_POSITION path): a velocity is already a "delta" in the
    sense that transform exists for.
  - images: `obs/robot0_agentview_left_image` (exterior) and `obs/robot0_eye_in_hand_image`
    (wrist) -- same field names the sim hdf5s use, so this part is identical to the RoboCasa
    converter.

These hdf5s were verified (2026-09-14, LoadCoffee) to only ever label `acting_agent` as "human" or
"robot" -- no "steered" -- so there is no --include_steered/--olaf_sidecar here; add them the same
way convert_hitl_hdf5_to_lerobot.py does if a later task's recordings do carry a "steered" label.

Usage:
    python examples/robocasa/convert_realworld_hitl_hdf5_to_lerobot.py \
        --repo_name hitl_loadcoffee_all4 \
        --raw_dataset_path \
          /mnt/hdd3/arpit/semantic_corrections/expdata/real_world/LoadCoffee/2026-09-12-20-39-54-correction-1.hdf5 \
          /mnt/hdd3/arpit/semantic_corrections/expdata/real_world/LoadCoffee/2026-09-12-20-39-54-correction-2.hdf5 \
          /mnt/hdd3/arpit/semantic_corrections/expdata/real_world/LoadCoffee/2026-09-12-20-39-54-correction-3.hdf5 \
          /mnt/hdd3/arpit/semantic_corrections/expdata/real_world/LoadCoffee/2026-09-13-00-01-15.hdf5
"""

import argparse
import json

import h5py
from convert_hitl_hdf5_to_lerobot import _preintv_drop_mask
from lerobot_export import save_episodes_as_lerobot
import numpy as np

STATE_DIM = 8  # joint_position (7) + gripper_position (1)
ACTION_DIM = 8  # joint_velocity (7) + gripper_position target (1)
FPS = 15  # matches these hdf5s' `control_hz` attr


class _RealWorldHDF5Episode:
    """Minimal EpisodeRecorder stand-in, populated by reading a saved real-DROID-robot hdf5."""

    def __init__(
        self,
        hdf5_path: str,
        demo_name: str = "demo_0",
        preintv_window: int = 10,
        robot_only: bool = False,
    ):
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_name]
            obs = demo["obs"]
            images = obs["robot0_agentview_left_image"][:]
            wrist_images = obs["robot0_eye_in_hand_image"][:]
            gripper_position = demo["droid"]["gripper_position"][:]
            states = np.concatenate(
                [demo["droid"]["joint_position"][:], gripper_position[:, None]],
                axis=1,
            ).astype(np.float64)
            actions = demo["actions"][:].astype(np.float64)
            assert actions.shape[1] == ACTION_DIM, f"expected {ACTION_DIM}-dim actions, got {actions.shape}"
            self.task_lang = json.loads(demo.attrs["ep_meta"])["lang"]

            agent = np.array([a.decode() for a in demo["acting_agent"][:]])
            unknown = set(np.unique(agent)) - {"human", "robot"}
            if unknown:
                raise ValueError(
                    f"{hdf5_path}: acting_agent has unexpected values {unknown} -- this script "
                    "only handles 'human'/'robot' (verified on LoadCoffee; a 'steered' label "
                    "would need handling like convert_hitl_hdf5_to_lerobot.py's --include_steered)"
                )

            if robot_only:
                keep = agent == "robot"
                is_intervention = np.zeros(len(agent), dtype=bool)
            else:
                is_intervention = agent == "human"
                drop = _preintv_drop_mask(is_intervention, preintv_window)
                keep = ~drop

            self.images = images[keep]
            self.wrist_images = wrist_images[keep]
            self.states = states[keep]
            self.actions = actions[keep]
            self.is_intervention = is_intervention[keep].astype(np.int64).reshape(-1, 1)


def main(args):
    if len(args.demo_name) == 1:
        demo_names = args.demo_name * len(args.raw_dataset_path)
    elif len(args.demo_name) == len(args.raw_dataset_path):
        demo_names = args.demo_name
    else:
        raise ValueError(
            f"--demo_name must be given once (applied to every path) or once per --raw_dataset_path "
            f"({len(args.raw_dataset_path)} paths, got {len(args.demo_name)} demo_name values)"
        )

    episodes = [
        _RealWorldHDF5Episode(
            path,
            demo_name=name,
            preintv_window=args.preintv_window,
            robot_only=args.robot_only,
        )
        for path, name in zip(args.raw_dataset_path, demo_names, strict=True)
    ]
    save_episodes_as_lerobot(
        episodes,
        args.repo_name,
        lerobot_home=args.lerobot_home,
        fps=FPS,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        image_size=224,  # matches openpi.models.model.IMAGE_RESOLUTION
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert one or more real-DROID-robot HITL demo_0.hdf5 files to a plain LeRobot dataset"
    )
    parser.add_argument("--raw_dataset_path", type=str, nargs="+", required=True, help="Path(s) to the raw hdf5 dataset(s)")
    parser.add_argument(
        "--demo_name",
        type=str,
        nargs="+",
        default=["demo_0"],
        help="Demo group name(s) inside the hdf5(s), e.g. 'demo_0'. Either one value applied to every "
        "--raw_dataset_path, or exactly one per path.",
    )
    parser.add_argument("--repo_name", type=str, required=True, help="LeRobot repo_id to write, e.g. hitl_loadcoffee_all4")
    parser.add_argument("--lerobot_home", type=str, default=None, help="Defaults to ~/.cache/huggingface/lerobot")
    parser.add_argument(
        "--preintv_window",
        type=int,
        default=10,
        help="Number of robot frames immediately before each human takeover to drop (SIRIUS-style "
        "preintv). Default of 10 matches convert_hitl_hdf5_to_lerobot.py's default.",
    )
    parser.add_argument(
        "--robot_only",
        action="store_true",
        help="Keep only 'robot' (autonomous-policy) frames, dropping every 'human' frame outright "
        "-- no preintv trimming, is_intervention all-zero. For a no-corrections ablation.",
    )
    args = parser.parse_args()
    main(args)
