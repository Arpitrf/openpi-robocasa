"""Convert saved HITL / HG-DAGGER demo hdf5 files into a single plain LeRobot dataset, using
`save_episodes_as_lerobot` from `lerobot_export.py` (copied from Arpitrf/semantic_corrections) --
that function takes a *list* of EpisodeRecorder-like objects and writes one episode per entry, so
each kept segment below (see "Episode splitting") becomes one LeRobot episode. `_Segment` is a
minimal stand-in with the same `.images` / `.wrist_images` / `.states` / `.actions` /
`.is_intervention` / `.task_lang` attributes that function iterates over.

No reordering is applied to `states`/`actions`: per semantic_corrections/hitl/obs.py's
`obs_to_policy_state` and episode_recorder.py's `_write_hdf5`, the hdf5's raw obs/robot0_* fields
are written in the exact order pi0 expects (base_to_eef_pos, base_to_eef_quat, base_pos,
base_quat, gripper_qpos), and `actions` is recorded pre-`convert_action()` -- i.e. already in the
policy-output space it's used as a training target for.

SIRIUS-style intervention handling (https://ut-austin-rpl.github.io/sirius/, arXiv:2211.08416):
each hdf5 has a per-timestep `acting_agent` field (sibling of `actions`, "human" / "robot")
marking when the human took over from the autonomous policy. Every surviving frame is labeled
`is_intervention` (human -> 1, robot -> 0) for training-time reweighting (see
LeRobotRobocasaHitlDataConfig.intervention_p_target in training/config.py and
data_loader.create_torch_data_loader), and the `preintv_window` frames immediately preceding each
robot->human handoff are dropped -- those are the frames judged bad enough to trigger a
correction, and SIRIUS excludes them from training rather than reproducing them. Robot frames
that survive ARE kept and trained on; only the preintv window is discarded.

Episode splitting: pi0 trains on `action_horizon`-long action chunks (50 frames = 2.5s at 20fps),
and openpi applies no `*_is_pad` loss mask anywhere, so a chunk that straddles a dropped preintv
window would be trained as if the robot teleported across the gap. Rather than writing one
LeRobot episode per demo with holes in it, each demo is split at every drop boundary and each
contiguous run of kept frames is written as its own episode, so no chunk ever crosses a gap. The
cost is more short episodes whose tail chunks are end-padded (already true of the last
action_horizon frames of any episode); `--min_segment_len` drops segments too short to be worth
that, and the emitted `dataset_manifest.json` records the segment-length distribution.

Usage (HG-DAGGER rounds -- aggregates rounds 1..r into one dataset):
    python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hgdagger_coffeesetupmug_r2 \
        --round_dirs <expdata>/hgdagger/CoffeeSetupMug/round_1 \
                     <expdata>/hgdagger/CoffeeSetupMug/round_2

Usage (explicit files, as in README_HITL_LORA.md):
    python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug \
        --raw_dataset_path data/CoffeeSetupMug/2026-06-30-22-00/demo_0.hdf5 \
                           data/CoffeeSetupMug/2026-08-19-12-55/demo_0.hdf5

--demo_name also accepts one value per --raw_dataset_path, for datasets where the demo group
isn't "demo_0" in every file.
"""

import argparse
import json
import pathlib
import re

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


def _contiguous_runs(keep: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index pairs for each contiguous True run in `keep`."""
    runs = []
    start = None
    for t, k in enumerate(keep):
        if k and start is None:
            start = t
        elif not k and start is not None:
            runs.append((start, t))
            start = None
    if start is not None:
        runs.append((start, len(keep)))
    return runs


class _Segment:
    """One contiguous run of kept frames, shaped like an EpisodeRecorder for the exporter."""

    def __init__(self, images, wrist_images, states, actions, is_intervention, task_lang):
        self.images = images
        self.wrist_images = wrist_images
        self.states = states
        self.actions = actions
        self.is_intervention = is_intervention
        self.task_lang = task_lang


def load_segments(hdf5_path, demo_name="demo_0", preintv_window=10, min_segment_len=1):
    """Read one demo and split it into training segments. Returns (segments, stats)."""
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
        task_lang = json.loads(demo.attrs["ep_meta"])["lang"]
        agent = np.array([a.decode() for a in demo["acting_agent"][:]])

    unknown = set(np.unique(agent)) - {"human", "robot"}
    if unknown:
        raise ValueError(
            f"{hdf5_path}: unexpected acting_agent value(s) {sorted(unknown)}; this converter "
            f"only understands 'human' and 'robot'."
        )

    is_human = agent == "human"
    drop = _preintv_drop_mask(is_human, preintv_window)
    keep = ~drop

    segments, seg_stats = [], []
    short_frames = 0
    for start, stop in _contiguous_runs(keep):
        length = stop - start
        n_intv = int(is_human[start:stop].sum())
        if length < min_segment_len:
            short_frames += length
            seg_stats.append({"start": start, "len": length, "intervention": n_intv, "kept": False})
            continue
        segments.append(
            _Segment(
                images=images[start:stop],
                wrist_images=wrist_images[start:stop],
                states=states[start:stop],
                actions=actions[start:stop],
                is_intervention=is_human[start:stop].astype(np.int64).reshape(-1, 1),
                task_lang=task_lang,
            )
        )
        seg_stats.append({"start": start, "len": length, "intervention": n_intv, "kept": True})

    kept_frames = int(sum(s["len"] for s in seg_stats if s["kept"]))
    stats = {
        "path": str(hdf5_path),
        "demo": demo_name,
        "task_lang": task_lang,
        "frames_raw": int(len(agent)),
        "frames_human_raw": int(is_human.sum()),
        "frames_dropped_preintv": int(drop.sum()),
        "frames_dropped_short_segments": int(short_frames),
        "frames_kept": kept_frames,
        "frames_kept_intervention": int(sum(s["intervention"] for s in seg_stats if s["kept"])),
        "num_segments_written": int(len(segments)),
        "segments": seg_stats,
    }
    return segments, stats


def _demo_files_in_round(round_dir):
    """demo_<i>.hdf5 in a round dir, numerically ordered (human_only_* excluded)."""
    round_dir = pathlib.Path(round_dir)
    files = [p for p in round_dir.glob("demo_*.hdf5") if re.fullmatch(r"demo_\d+\.hdf5", p.name)]
    if not files:
        raise FileNotFoundError(f"no demo_<i>.hdf5 files in {round_dir}")
    return sorted(files, key=lambda p: int(re.findall(r"\d+", p.name)[0]))


def main(args):
    if args.round_dirs:
        sources = [(p, "demo_0", pathlib.Path(d).name) for d in args.round_dirs for p in _demo_files_in_round(d)]
    else:
        if len(args.demo_name) == 1:
            demo_names = args.demo_name * len(args.raw_dataset_path)
        elif len(args.demo_name) == len(args.raw_dataset_path):
            demo_names = args.demo_name
        else:
            raise ValueError(
                f"--demo_name must be given once (applied to every path) or once per "
                f"--raw_dataset_path ({len(args.raw_dataset_path)} paths, got "
                f"{len(args.demo_name)} demo_name values)"
            )
        sources = [(p, n, None) for p, n in zip(args.raw_dataset_path, demo_names, strict=True)]

    episodes, source_stats = [], []
    for path, demo_name, round_name in sources:
        segments, stats = load_segments(
            path,
            demo_name=demo_name,
            preintv_window=args.preintv_window,
            min_segment_len=args.min_segment_len,
        )
        stats["round"] = round_name
        episodes.extend(segments)
        source_stats.append(stats)
        print(
            f"{path} [{demo_name}]: {stats['frames_raw']} frames -> "
            f"{stats['frames_kept']} kept in {stats['num_segments_written']} segment(s) "
            f"({stats['frames_kept_intervention']} intervention), "
            f"dropped {stats['frames_dropped_preintv']} preintv "
            f"+ {stats['frames_dropped_short_segments']} short-segment"
        )

    kept = sum(s["frames_kept"] for s in source_stats)
    intv = sum(s["frames_kept_intervention"] for s in source_stats)
    if not episodes:
        raise SystemExit("no segments survived; nothing to write")
    if intv == 0 or intv == kept:
        raise SystemExit(
            f"dataset has only one class ({intv} intervention / {kept - intv} robot frames). "
            f"LeRobotRobocasaHitlDataConfig.intervention_p_target needs both present -- either "
            f"collect demos that contain a human takeover, or set intervention_p_target=None."
        )

    save_episodes_as_lerobot(episodes, args.repo_name, lerobot_home=args.lerobot_home)

    lerobot_home = args.lerobot_home or pathlib.Path("~/.cache/huggingface/lerobot").expanduser()
    manifest = {
        "repo_name": args.repo_name,
        "preintv_window": args.preintv_window,
        "min_segment_len": args.min_segment_len,
        "totals": {
            "source_demos": len(source_stats),
            "episodes_written": len(episodes),
            "frames_raw": sum(s["frames_raw"] for s in source_stats),
            "frames_dropped_preintv": sum(s["frames_dropped_preintv"] for s in source_stats),
            "frames_dropped_short_segments": sum(s["frames_dropped_short_segments"] for s in source_stats),
            "frames_kept": kept,
            "frames_kept_intervention": intv,
            "frames_kept_robot": kept - intv,
            "intervention_fraction": round(intv / kept, 4),
        },
        "sources": source_stats,
    }
    manifest_path = pathlib.Path(lerobot_home) / args.repo_name / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"\n{len(episodes)} episodes / {kept} frames "
        f"({intv} intervention = {100 * intv / kept:.1f}%) -> {manifest_path.parent}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HITL / HG-DAGGER demo hdf5 files to a plain LeRobot dataset")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw_dataset_path", type=str, nargs="+", help="Path(s) to the raw hdf5 dataset(s)")
    src.add_argument(
        "--round_dirs",
        type=str,
        nargs="+",
        help="HG-DAGGER round director(ies); every demo_<i>.hdf5 inside each is used. Pass "
        "rounds 1..r to build round r's aggregated dataset.",
    )
    parser.add_argument(
        "--demo_name",
        type=str,
        nargs="+",
        default=["demo_0"],
        help="Demo group name(s) inside the hdf5(s), e.g. 'demo_0'. Either one value applied to every "
        "--raw_dataset_path, or exactly one per path. Ignored with --round_dirs (always demo_0).",
    )
    parser.add_argument("--repo_name", type=str, required=True, help="LeRobot repo_id to write, e.g. hgdagger_coffeesetupmug_r1")
    parser.add_argument("--lerobot_home", type=str, default=None, help="Defaults to ~/.cache/huggingface/lerobot")
    parser.add_argument(
        "--preintv_window",
        type=int,
        default=10,
        help="Number of robot frames immediately before each human takeover to drop "
        "(SIRIUS-style preintv). Default of 10 is in SIRIUS's own ballpark (their hyperparameter "
        "is 15) -- unverified for this domain either way.",
    )
    parser.add_argument(
        "--min_segment_len",
        type=int,
        default=1,
        help="Drop contiguous kept-frame segments shorter than this many frames. Short segments "
        "only yield end-padded action chunks (openpi applies no pad mask). Default 1 keeps "
        "everything; see dataset_manifest.json for the length distribution.",
    )
    args = parser.parse_args()
    main(args)
