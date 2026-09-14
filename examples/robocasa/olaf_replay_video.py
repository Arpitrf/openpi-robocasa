"""Render a cumulative open-loop replay of an OLAF window: recorded vs relabeled actions.

Both videos start from the SAME simulator state at the beginning of the correction window and run
open-loop -- no per-frame state restoration, which would mask the trajectory divergence this is
meant to show. The only difference between them is the first W actions (the pre-intervention
window): recorded in one, OLAF-relabeled in the other. Everything after is the same recorded
action sequence, so any difference you see downstream is caused by the relabeling.

NOTE: sim.data.body_xpos[i] is a live view into MuJoCo's buffer and np.asarray() will not copy a
float64 array -- every position read must .copy() or it silently tracks the simulation forward.

Usage:
    python examples/robocasa/olaf_replay_video.py --sidecar olaf_runs/grip/<demo>.olaf.json \
        --out_dir videos --after 50
"""

from __future__ import annotations

import argparse
import json
import pathlib

import gymnasium as gym
import h5py
from hitl_env import get_robosuite_env
from hitl_env import mark_gym_env_reset
import imageio
import numpy as np
from robocasa.utils.env_utils import convert_action
from sim_state import restore_sim_state_from_hdf5

CAMERA = "robot0_agentview_left"


def object_base(raw, object_name: str):
    """Manipulated object's position in the robot base frame (copied, not a live view).

    Returns None when the env has no such object -- some tasks operate a fixture rather than
    carrying something, and the video is still the point.
    """
    if object_name not in getattr(raw, "objects", {}):
        return None
    b = raw.objects[object_name].root_body
    obj = np.array(raw.sim.data.body_xpos[raw.sim.model.body_name2id(b)], dtype=float)
    site = raw.robots[0].robot_model.base.correct_naming("center")
    sim = raw.sim
    p = np.array(sim.data.site_xpos[sim.model.site_name2id(site)], dtype=float)
    rot = np.array(sim.data.get_site_xmat(site), dtype=float).reshape(3, 3)
    return rot.T @ (obj - p)


def render(raw, width, height):
    rgb = raw.sim.render(camera_name=CAMERA, width=width, height=height)
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    return np.ascontiguousarray(rgb[::-1])


def rollout(env, raw, hdf5, demo, start, actions, width, height, object_name):
    env.reset()
    restore_sim_state_from_hdf5(raw, hdf5, demo_name=demo, frame=start)
    mark_gym_env_reset(env)
    frames = [render(raw, width, height)]
    start_pos = object_base(raw, object_name)
    for a in actions:
        env.step(convert_action(np.asarray(a, dtype=np.float64)))
        frames.append(render(raw, width, height))
    return frames, start_pos, object_base(raw, object_name)


def main() -> None:
    p = argparse.ArgumentParser(description="Cumulative OLAF replay video")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--after", type=int, default=50,
                   help="Recorded frames to keep executing after the window, so the consequences "
                        "of the relabeled prefix are visible.")
    p.add_argument("--full", action="store_true",
                   help="Replay the WHOLE demo from frame 0 to the end instead of starting at the "
                        "correction window. Open-loop over hundreds of steps drifts far more than "
                        "over the 10-frame window, so treat it as qualitative.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--env_name", required=True,
                   help="RoboCasa env, e.g. CoffeeSetupMug or StartElectricKettle.")
    p.add_argument("--object_name", default="obj",
                   help="Manipulated object in the env, for the displacement readout. Ignored if "
                        "the env has no such object.")
    args = p.parse_args()

    sc = json.loads(pathlib.Path(args.sidecar).read_text())
    hdf5, demo = sc["source"]["hdf5"], sc["source"]["demo"]
    lo, hi = sc["window"]["start"], sc["window"]["end"]
    olaf = np.array([f["action"] for f in sc["frames"] if f.get("action") is not None])

    with h5py.File(hdf5, "r") as f:
        acts = f["data"][demo]["actions"][:].astype(np.float64)

    if args.full:
        # Whole demo from frame 0; OLAF actions swapped in over the window only.
        start = 0
        rec_seq = acts
        olaf_seq = acts.copy()
        olaf_seq[lo:hi] = olaf
    else:
        start = lo
        end = min(hi + args.after, len(acts))
        rec_seq = np.concatenate([acts[lo:hi], acts[hi:end]])
        olaf_seq = np.concatenate([olaf, acts[hi:end]])

    seqs = {"recorded": rec_seq, "olaf": olaf_seq}

    env = gym.make(f"robocasa/{args.env_name}", split=None, seed=7,
                   camera_widths=args.width, camera_heights=args.height)
    raw = get_robosuite_env(env)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(args.sidecar).stem.replace(".olaf", "") + ("_full" if args.full else "")
    results = {}
    for name, seq in seqs.items():
        frames, p0, p1 = rollout(env, raw, hdf5, demo, start, seq, args.width, args.height,
                                 args.object_name)
        path = out / f"{stem}_{name}.mp4"
        imageio.mimwrite(path, frames, fps=args.fps, quality=8)
        if p0 is None:
            print(f"{path}  ({len(frames)} frames)  [no '{args.object_name}' in env]")
        else:
            d = (p1 - p0) * 100
            results[name] = d
            print(f"{path}  ({len(frames)} frames)  {args.object_name} moved {np.round(d, 2)} cm "
                  f"|d|={np.linalg.norm(d):.2f}")
    env.close()

    span = f"frames [0,{len(seqs['recorded'])})" if args.full else f"window [{lo},{hi}) + after"
    print(f"\nreplayed {span} from a single restore at frame {start}; "
          f"only actions {lo}-{hi} differ between the two videos")


if __name__ == "__main__":
    main()
