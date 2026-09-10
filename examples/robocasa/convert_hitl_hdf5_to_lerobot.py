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
checked so far). This script does NOT use `correction_semantic_annotations`. Confirm with whoever
owns the semantic_corrections pipeline if that field should factor in too.

SIRIUS-style intervention handling (https://ut-austin-rpl.github.io/sirius/, arXiv:2211.08416):
each hdf5 has a per-timestep `acting_agent` field (sibling of `actions`, values among
"human"/"robot"/"steered") marking when a human took over from the autonomous policy. This script
drops the `preintv_window` frames immediately preceding each robot->takeover transition (human OR
steered -- both signify the robot was about to need help) -- these are the frames judged bad
enough to trigger a takeover, and SIRIUS's own scheme excludes them from training rather than
reproducing them. `preintv_window` defaults to 10, carried over from SIRIUS's own ballpark (their
hyperparameter is 15; no domain-specific value has been established for this fps=20 setup -- treat
as unverified either way).

`--include_steered` controls how "steered" frames (FRS-style assisted control, distinct from raw
human teleop) are handled, independent of `preintv_window`: by default (False) they're dropped
entirely and `is_intervention` is True only for "human" -- SIRIUS's own scheme has no analog of
"steered", so this is the closer adaptation. Pass `--include_steered` to instead keep them and
fold them into `is_intervention` alongside "human" (the original, pre-ablation behavior) for an
A/B comparison of whether dropping them mattered.

Usage:
    python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug \
        --raw_dataset_path data/CoffeeSetupMug/2026-06-30-22-00/demo_0.hdf5 \
                           data/CoffeeSetupMug/2026-08-19-12-55/demo_0.hdf5 \
                           data/CoffeeSetupMug/2026-08-21-20-26/demo_0.hdf5

--demo_name also accepts one value per --raw_dataset_path, for datasets where the demo group
isn't "demo_0" in every file.
"""

import argparse
import json

import h5py
import numpy as np

from lerobot_export import save_episodes_as_lerobot


def _preintv_drop_mask(is_intv: np.ndarray, preintv_window: int) -> np.ndarray:
    """SIRIUS-style preintv mask: for each robot->intervention transition, drop the
    `preintv_window` frames immediately before it, clipped to the true start of that contiguous
    robot run so it never reaches into a preceding intervention block or before frame 0."""
    drop = np.zeros(len(is_intv), dtype=bool)
    for t in range(1, len(is_intv)):
        if is_intv[t] and not is_intv[t - 1]:
            run_start = t - 1
            while run_start > 0 and not is_intv[run_start - 1]:
                run_start -= 1
            window_start = max(run_start, t - preintv_window)
            drop[window_start:t] = True
    return drop


class _HDF5Episode:
    """Minimal EpisodeRecorder stand-in, populated by reading a saved hdf5 instead of a live
    HITL session."""

    def __init__(
        self,
        hdf5_path: str,
        demo_name: str = "demo_0",
        preintv_window: int = 10,
        include_steered: bool = False,
    ):
        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_name]
            obs = demo["obs"]
            images = obs["robot0_agentview_left_image"][:]
            wrist_images = obs["robot0_eye_in_hand_image"][:]
            states = np.concatenate(
                [
                    obs["robot0_base_to_eef_pos"][:],
                    obs["robot0_base_to_eef_quat"][:],
                    obs["robot0_base_pos"][:],
                    obs["robot0_base_quat"][:],
                    obs["robot0_gripper_qpos"][:],
                ],
                axis=1,
            ).astype(np.float64)  # raw hdf5 obs fields are float32; schema declares float64
            actions = demo["actions"][:].astype(np.float64)
            self.task_lang = json.loads(demo.attrs["ep_meta"])["lang"]

            agent = np.array([a.decode() for a in demo["acting_agent"][:]])
            is_takeover = np.isin(agent, ("human", "steered"))  # any non-autonomous control
            is_steered = agent == "steered"
            preintv_drop = _preintv_drop_mask(is_takeover, preintv_window)
            if include_steered:
                # Original (pre-ablation) behavior: steered frames are kept and counted as
                # intervention alongside human, like SIRIUS's own "human"/"steered" pooling.
                drop = preintv_drop
                is_intervention = is_takeover
            else:
                # Steered frames dropped entirely like preintv (see module docstring);
                # is_intervention is human-only.
                drop = preintv_drop | is_steered
                is_intervention = agent == "human"
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
        _HDF5Episode(
            path,
            demo_name=name,
            preintv_window=args.preintv_window,
            include_steered=args.include_steered,
        )
        for path, name in zip(args.raw_dataset_path, demo_names, strict=True)
    ]
    save_episodes_as_lerobot(episodes, args.repo_name, lerobot_home=args.lerobot_home)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert one or more HITL demo_0.hdf5 files to a plain LeRobot dataset")
    parser.add_argument("--raw_dataset_path", type=str, nargs="+", required=True, help="Path(s) to the raw hdf5 dataset(s)")
    parser.add_argument(
        "--demo_name",
        type=str,
        nargs="+",
        default=["demo_0"],
        help="Demo group name(s) inside the hdf5(s), e.g. 'demo_0'. Either one value applied to every "
        "--raw_dataset_path, or exactly one per path.",
    )
    parser.add_argument("--repo_name", type=str, required=True, help="LeRobot repo_id to write, e.g. hitl_coffeesetupmug")
    parser.add_argument("--lerobot_home", type=str, default=None, help="Defaults to ~/.cache/huggingface/lerobot")
    parser.add_argument(
        "--preintv_window",
        type=int,
        default=10,
        help="Number of robot frames immediately before each human/steered takeover to drop "
        "(SIRIUS-style preintv). Default of 10 is in SIRIUS's own ballpark (their hyperparameter "
        "is 15) -- unverified for this domain either way.",
    )
    parser.add_argument(
        "--include_steered",
        action="store_true",
        help="Keep 'steered' frames and count them as is_intervention alongside 'human', instead "
        "of dropping them entirely (the default). For an A/B comparison against the no-steered "
        "ablation.",
    )
    args = parser.parse_args()
    main(args)
