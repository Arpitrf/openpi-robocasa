"""Convert one or more saved HITL demo_0.hdf5 files into a single plain LeRobot dataset, using
`save_episodes_as_lerobot` from `lerobot_export.py` (copied from Arpitrf/semantic_corrections)
unchanged -- that function already takes a *list* of EpisodeRecorder-like objects and writes one
episode per entry, so multiple --raw_dataset_path values become multiple episodes in one dataset.
`_HDF5Episode` below just reads a saved hdf5's arrays into an object with the same `.images` /
`.wrist_images` / `.states` / `.actions` / `.task_lang` attributes `save_episodes_as_lerobot`
iterates over.

No reordering is applied to `states`/`actions`: per semantic_corrections/hitl/obs.py's
`obs_to_policy_state` and episode_recorder.py's `_write_hdf5` (see their frs branch), the hdf5's
raw obs/robot0_* fields are written in the exact order pi0 expects (base_to_eef_pos, base_to_eef_quat,
base_pos, base_quat, gripper_qpos), and `actions` is recorded pre-`convert_action()` -- i.e. already
in the policy-output space it's used as a training target for.

CAVEAT: these hdf5 files record a "monitored" rollout with a `correction_semantic_annotations`
field describing *where the rollout went wrong* (collisions, missed placements, etc.) -- they are
not necessarily successful demonstrations (`rewards`/`dones` are 0/False throughout in the files
checked so far). This script converts the raw recorded actions/observations as-is; it does NOT
filter, reweight, or otherwise use `correction_semantic_annotations`. Confirm with whoever owns
the semantic_corrections pipeline how that field is meant to be incorporated before training a
policy on this data and expecting it to reproduce successful behavior.

Usage:
    python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug \
        --raw_dataset_path data/CoffeeSetupMug/2026-06-30-22-00/demo_0.hdf5 \
                           data/CoffeeSetupMug/2026-08-19-12-55/demo_0.hdf5 \
                           data/CoffeeSetupMug/2026-08-21-20-26/demo_0.hdf5
"""

import argparse
import json

import h5py
import numpy as np

from lerobot_export import save_episodes_as_lerobot


class _HDF5Episode:
    """Minimal EpisodeRecorder stand-in, populated by reading a saved hdf5 instead of a live
    HITL session."""

    def __init__(self, hdf5_path: str, demo_name: str = "demo_0"):
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_name]
            obs = demo["obs"]
            self.images = obs["robot0_agentview_left_image"][:]
            self.wrist_images = obs["robot0_eye_in_hand_image"][:]
            self.states = np.concatenate(
                [
                    obs["robot0_base_to_eef_pos"][:],
                    obs["robot0_base_to_eef_quat"][:],
                    obs["robot0_base_pos"][:],
                    obs["robot0_base_quat"][:],
                    obs["robot0_gripper_qpos"][:],
                ],
                axis=1,
            ).astype(np.float64)  # raw hdf5 obs fields are float32; schema declares float64
            self.actions = demo["actions"][:].astype(np.float64)
            self.task_lang = json.loads(demo.attrs["ep_meta"])["lang"]


def main(args):
    episodes = [_HDF5Episode(path, demo_name=args.demo_name) for path in args.raw_dataset_path]
    save_episodes_as_lerobot(episodes, args.repo_name, lerobot_home=args.lerobot_home)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert one or more HITL demo_0.hdf5 files to a plain LeRobot dataset")
    parser.add_argument("--raw_dataset_path", type=str, nargs="+", required=True, help="Path(s) to the raw hdf5 dataset(s)")
    parser.add_argument("--demo_name", type=str, default="demo_0")
    parser.add_argument("--repo_name", type=str, required=True, help="LeRobot repo_id to write, e.g. hitl_coffeesetupmug")
    parser.add_argument("--lerobot_home", type=str, default=None, help="Defaults to ~/.cache/huggingface/lerobot")
    args = parser.parse_args()
    main(args)
