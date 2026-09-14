"""Sanity checks over olaf_relabel.py sidecars. No GPU, no API calls.

For each relabeled window, reports where the chosen plan takes the gripper relative to every scene
keypoint, compared against the action the robot actually took, and how the chosen plan ranks among
all the candidates it was offered.

Reading the numbers:
  * `gain` is negative when the chosen plan ends closer to the goal keypoint than the recorded
    action did. Negative is usually good -- but not always: some corrections ask the robot to back
    off first ("go away from the handle a bit, then approach"), and some are about the gripper
    rather than about moving anywhere. Read `gain` alongside the correction text, which is printed.
  * `rank` is how many of the candidates would have ended closer to the goal. 0 means the VLM chose
    the single best available plan; a rank near the pool size means it chose one of the worst.
  * `oracle` is the best gain any candidate could have achieved. When `oracle` is ~0 or positive,
    no plan in the pool improves on the recorded action and the relabeling cannot help there --
    that is a property of the candidate pool, not of the VLM's choice.

Usage:
    python examples/robocasa/olaf_report.py olaf_sidecars/*.olaf.json
    python examples/robocasa/olaf_report.py olaf_sidecars/*.olaf.json --goal kettle_lever
"""

from __future__ import annotations

import argparse
import json
import pathlib

import h5py
import numpy as np

ACHIEVED_M_PER_UNIT = 0.0108  # see olaf_relabel.py


def gripper_start(sidecar: dict) -> np.ndarray:
    src = sidecar["source"]
    with h5py.File(src["hdf5"], "r") as f:
        return np.asarray(
            f["data"][src["demo"]]["obs"]["robot0_base_to_eef_pos"][sidecar["window"]["start"]],
            dtype=np.float64,
        )


def pick_goal(keypoints: list[dict], preferences: list[str]) -> dict:
    """The keypoint a correction is about -- the first of `preferences` present in this demo.

    There is no reliable way to infer this. A correction's goal is sometimes a static landmark
    (CoffeeSetupMug's dispenser_nozzle), sometimes part of a fixture the robot is operating
    (StartElectricKettle's kettle_lever), and the held object (the mug) is never the goal even
    though it is the first keypoint listed. So the caller names it; the per-keypoint table below
    shows every keypoint regardless, which is the honest output.
    """
    ids = [k["id"] for k in keypoints]
    for want in preferences:
        for k in keypoints:
            if k["id"] == want:
                return k
    raise SystemExit(f"none of --goal {preferences} found among {ids}")


def report(path: pathlib.Path, goal_preferences: list[str]) -> dict:
    sc = json.loads(path.read_text())
    kps = [{"id": k["id"], "pos": np.asarray(k["pos"]), "attachment": k["attachment"]} for k in sc["keypoints"]]
    goal = pick_goal(kps, goal_preferences)
    gp = gripper_start(sc)
    cands = np.asarray(sc["candidates"])
    recorded = np.asarray([f["original_action"] for f in sc["frames"]])
    chosen_id = sc["selection"]["chosen_candidate_id"]

    ends = gp + cands[:, :, 0:3].sum(axis=1) * ACHIEVED_M_PER_UNIT
    rec_end = gp + recorded[:, 0:3].sum(axis=0) * ACHIEVED_M_PER_UNIT
    d_all = np.linalg.norm(ends - goal["pos"], axis=1) * 100
    d_rec = float(np.linalg.norm(rec_end - goal["pos"]) * 100)

    lo, hi = sc["window"]["start"], sc["window"]["end"]
    print(f"\n{path.name}   window [{lo},{hi})   goal keypoint: {goal['id']}")
    print(f"  correction: {sc['correction_semantics'][:150]}")
    if chosen_id is None:
        print("  (--dry_run sidecar: no selection yet)")
        print(f"  oracle gain available: {d_all.min() - d_rec:+.2f} cm "
              f"({int((d_all < d_rec).sum())}/{len(d_all)} candidates beat the recorded action)")
        return {}

    chosen = int(chosen_id)
    grip_rec = float(recorded[0][6])
    grip_new = float(sc["frames"][0]["action"][6])
    print(f"  chose [{chosen}]: {sc['selection']['reason'][:120]}")
    print(f"  gripper -> {goal['id']}: recorded {d_rec:.2f} cm, chosen {d_all[chosen]:.2f} cm "
          f"({d_all[chosen] - d_rec:+.2f})   oracle {d_all.min() - d_rec:+.2f}   "
          f"rank {int((d_all < d_all[chosen]).sum())}/{len(d_all)}")
    if (grip_rec > 0) != (grip_new > 0):
        print(f"  gripper CHANGED: {'close' if grip_rec > 0 else 'open'} -> "
              f"{'close' if grip_new > 0 else 'OPEN'}")
    print("  distance to every keypoint (cm, recorded -> chosen):")
    for k in kps:
        a = np.linalg.norm(rec_end - k["pos"]) * 100
        b = np.linalg.norm(ends[chosen] - k["pos"]) * 100
        print(f"    {k['id']:22s} [{k['attachment']:7s}] {a:6.2f} -> {b:6.2f}  ({b - a:+.2f})")
    return {"gain": float(d_all[chosen] - d_rec), "oracle": float(d_all.min() - d_rec),
            "rank": int((d_all < d_all[chosen]).sum())}


def main() -> None:
    p = argparse.ArgumentParser(description="Sanity checks over OLAF sidecars")
    p.add_argument("sidecars", nargs="+")
    p.add_argument("--goal", required=True,
                   help="Comma-separated keypoint ids to measure against, in preference order; the "
                        "first present in each demo is used. e.g. "
                        "dispenser_nozzle,nozzle,kettle_lever,lever,kettle_lid")
    args = p.parse_args()

    prefs = [g.strip() for g in args.goal.split(",") if g.strip()]
    rows = [r for r in (report(pathlib.Path(s), prefs) for s in sorted(args.sidecars)) if r]
    if rows:
        g = np.array([r["gain"] for r in rows])
        print(f"\n{len(rows)} windows: {int((g < 0).sum())} moved toward the goal, "
              f"{int((g > 0).sum())} away, mean gain {g.mean():+.2f} cm")
        print(f"  optimal picks (rank 0): {sum(1 for r in rows if r['rank'] == 0)}/{len(rows)}   "
              f"top-2: {sum(1 for r in rows if r['rank'] <= 1)}/{len(rows)}")
        dead = [r for r in rows if r["oracle"] >= -0.25]
        if dead:
            print(f"  {len(dead)} window(s) where NO candidate meaningfully improves on the "
                  "recorded action -- a candidate-pool limit, not a VLM failure")


if __name__ == "__main__":
    main()
