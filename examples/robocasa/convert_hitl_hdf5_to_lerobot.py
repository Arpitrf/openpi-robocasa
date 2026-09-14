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

`--robot_only` is a third, mutually-exclusive mode for isolating the effect of corrections
entirely: it keeps only frames where `acting_agent == "robot"` and drops every "human" and
"steered" frame outright, with no `preintv_window` trimming (that trimming exists to keep
robot->takeover transitions clean when intervention frames are trained on right alongside them;
with zero intervention frames in the output there's no transition to protect). `is_intervention`
is all-zero, so pair this with `intervention_p_target=None` on the data config -- the
`WeightedRandomSampler` in data_loader.py requires both classes present and raises otherwise.

`--olaf_sidecar` is a fourth mode, orthogonal to the three above and to `--preintv_window`. It
takes a sidecar JSON written by `olaf_relabel.py` (OLAF, arXiv:2310.17555) and, instead of dropping
that demo's pre-intervention frames, keeps them with their `actions` replaced by the ones a VLM
chose from a pool of pi0 samples as best carrying out the human's recorded verbal correction, and
marks them `is_intervention`. So the OLAF dataset is exactly the default dataset plus those frames.
The source hdf5 is never modified -- the swap happens in memory here. Mutually exclusive with
`--robot_only` (which does no preintv trimming, so there is nothing to un-drop).

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
import pathlib

import h5py
from lerobot_export import save_episodes_as_lerobot
import numpy as np


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


def _load_olaf_relabels(
    sidecar_path: str, hdf5_path: str, demo_name: str, num_frames: int, drop: np.ndarray
) -> dict[int, np.ndarray]:
    """Read an olaf_relabel.py sidecar and return {frame_index: 12-D action}.

    Validates hard rather than trusting the file: a sidecar silently paired with the wrong demo,
    or written with a different --preintv_window than this run, would produce a dataset labeled
    "OLAF" that isn't, which quietly invalidates the A/B against the other configs.
    """
    sidecar = json.loads(pathlib.Path(sidecar_path).read_text())
    if sidecar.get("schema_version") != 2:
        raise ValueError(
            f"{sidecar_path}: unsupported schema_version {sidecar.get('schema_version')} "
            "(expected 2). Regenerate it with examples/robocasa/olaf_relabel.py."
        )
    src = sidecar["source"]
    if src["demo"] != demo_name:
        raise ValueError(f"{sidecar_path}: sidecar is for demo {src['demo']!r}, not {demo_name!r}")
    # Compare (parent dir, basename) rather than the full path: the extraction assets record paths
    # from a different machine, so absolute paths legitimately differ.
    want, got = pathlib.Path(hdf5_path), pathlib.Path(src["hdf5"])
    if (want.parent.name, want.name) != (got.parent.name, got.name):
        raise ValueError(f"{sidecar_path}: sidecar is for {src['hdf5']}, not {hdf5_path}")
    if src["num_frames"] != num_frames:
        raise ValueError(
            f"{sidecar_path}: sidecar has {src['num_frames']} frames, hdf5 has {num_frames}"
        )

    window = sidecar["window"]
    expected = list(range(window["start"], window["end"]))
    got_frames = [f["t"] for f in sidecar["frames"]]
    if got_frames != expected:
        raise ValueError(f"{sidecar_path}: frames {got_frames} != window {expected}")

    relabels = {}
    for f in sidecar["frames"]:
        if f["action"] is None:
            raise ValueError(
                f"{sidecar_path}: frame {f['t']} has no action (a --dry_run sidecar?); "
                "rerun olaf_relabel.py without --dry_run"
            )
        action = np.asarray(f["action"], dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError(f"{sidecar_path}: frame {f['t']} action is not a finite 12-vector")
        if not drop[f["t"]]:
            raise ValueError(
                f"{sidecar_path}: frame {f['t']} is not currently dropped, so relabeling it would "
                f"change an action this dataset already trains on. Most likely the sidecar's "
                f"--preintv_window ({window['preintv_window']}) differs from this run's."
            )
        relabels[f["t"]] = action
    return relabels


class _HDF5Episode:
    """Minimal EpisodeRecorder stand-in, populated by reading a saved hdf5 instead of a live
    HITL session."""

    def __init__(
        self,
        hdf5_path: str,
        demo_name: str = "demo_0",
        preintv_window: int = 10,
        include_steered: bool = False,
        robot_only: bool = False,
        olaf_sidecar: str | None = None,
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
            if robot_only:
                # No-corrections ablation: keep only autonomous-policy frames, drop every
                # human/steered frame outright. No preintv trimming (nothing to protect a
                # transition into, since no intervention frames survive) and no intervention
                # frames left to label.
                keep = agent == "robot"
                is_intervention = np.zeros(len(agent), dtype=bool)
            else:
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

                if olaf_sidecar is not None:
                    # OLAF: keep the pre-intervention frames instead of dropping them, with the
                    # VLM-chosen actions. Note this tests membership in `drop`, not `preintv_drop`:
                    # when the window sits on "steered" frames (which happens on the demos where a
                    # steered run precedes the human takeover), they were dropped by is_steered
                    # instead -- and should be un-dropped just the same. That is also why this
                    # composes with --include_steered without special-casing.
                    relabels = _load_olaf_relabels(
                        olaf_sidecar, hdf5_path, demo_name, len(agent), drop
                    )
                    idx = np.fromiter(sorted(relabels), dtype=int)
                    actions[idx] = np.stack([relabels[t] for t in idx])
                    is_intervention = is_intervention.copy()  # may alias is_takeover
                    is_intervention[idx] = True
                    drop = drop.copy()
                    drop[idx] = False
                keep = ~drop

            self.images = images[keep]
            self.wrist_images = wrist_images[keep]
            self.states = states[keep]
            self.actions = actions[keep]
            self.is_intervention = is_intervention[keep].astype(np.int64).reshape(-1, 1)


def main(args):
    if args.robot_only and args.include_steered:
        raise ValueError("--robot_only and --include_steered are mutually exclusive.")
    if args.robot_only and args.olaf_sidecar:
        raise ValueError(
            "--robot_only and --olaf_sidecar are mutually exclusive: --robot_only does no preintv "
            "trimming, so there are no dropped frames for OLAF to relabel."
        )
    if len(args.demo_name) == 1:
        demo_names = args.demo_name * len(args.raw_dataset_path)
    elif len(args.demo_name) == len(args.raw_dataset_path):
        demo_names = args.demo_name
    else:
        raise ValueError(
            f"--demo_name must be given once (applied to every path) or once per --raw_dataset_path "
            f"({len(args.raw_dataset_path)} paths, got {len(args.demo_name)} demo_name values)"
        )
    if args.olaf_sidecar:
        if len(args.olaf_sidecar) != len(args.raw_dataset_path):
            raise ValueError(
                f"--olaf_sidecar needs exactly one value per --raw_dataset_path "
                f"({len(args.raw_dataset_path)} paths, got {len(args.olaf_sidecar)}); a sidecar is "
                f"demo-specific, so it is never broadcast. Use 'none' to skip a path."
            )
        sidecars = [None if s.lower() == "none" else s for s in args.olaf_sidecar]
    else:
        sidecars = [None] * len(args.raw_dataset_path)

    episodes = [
        _HDF5Episode(
            path,
            demo_name=name,
            preintv_window=args.preintv_window,
            include_steered=args.include_steered,
            robot_only=args.robot_only,
            olaf_sidecar=sidecar,
        )
        for path, name, sidecar in zip(args.raw_dataset_path, demo_names, sidecars, strict=True)
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
    parser.add_argument(
        "--olaf_sidecar",
        type=str,
        nargs="+",
        default=None,
        help="OLAF relabeling: one sidecar JSON from olaf_relabel.py per --raw_dataset_path (use "
        "'none' to skip a path). Keeps that demo's pre-intervention frames with VLM-chosen actions "
        "instead of dropping them, and marks them is_intervention. Mutually exclusive with "
        "--robot_only.",
    )
    parser.add_argument(
        "--robot_only",
        action="store_true",
        help="Keep only 'robot' (autonomous-policy) frames, dropping every 'human' and 'steered' "
        "frame outright -- no preintv trimming, is_intervention all-zero. For a no-corrections "
        "ablation; mutually exclusive with --include_steered.",
    )
    args = parser.parse_args()
    main(args)
