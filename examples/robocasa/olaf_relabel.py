"""OLAF action relabeling: replace a demo's pre-intervention actions using the human's correction.

OLAF (https://ut-austin-rpl.github.io/olaf/, arXiv:2310.17555) does something better with the
pre-intervention frames than SIRIUS's "drop them": it uses the human's *verbal* correction to pick,
from a pool of candidate actions, the one that best carries it out, and overwrites the recorded
action. The policy then learns the right thing at the states where it used to go wrong.

Per the paper, this queries the VLM **once per correction**, not once per frame:

    "Notice that OLAF queries the LLM only once per verbal correction. We found that issuing one
     query and then applying the results to all time steps in the pre-intervention period achieves
     similar performance than issuing a separate query for each individual time step, but the
     former is significantly more cost effective (about 15x fewer LLM calls)."   -- SS II-C

The unit selected is a pi0 action *chunk*, matching semantic_corrections' steering loop
(`scripts/run_correction.py:647,1795`), which gathers candidate chunks and executes the chosen one
whole. A chunk is a coherent multi-step plan, so its displacement accumulates; independently
chosen per-frame actions partially cancel.

Because there is exactly one query, at the window start, the keypoints are consumed at the single
frame `keypoints.json` was extracted for -- so there is no tracking, no simulator, and no branching
on the `attachment` field. Keypoints are listed to the VLM with their ids, positions and attachment
labels verbatim, and every candidate is described by where it takes the *gripper*. That is
well-defined in every task, which is why this runs unmodified on both CoffeeSetupMug (gripper
carries the mug toward a nozzle) and StartElectricKettle (gripper reaches for a fixed lever).

Deviations from the paper, deliberate:
  * Candidates are pi0 samples that REPLACE the action, where the paper uses fixed one-dimensional
    deltas ADDED on top of the original action. Consequence of sampling from the policy itself.
  * Training is LoRA from the pretrained checkpoint rather than aggregating the relabeled data with
    the full pretraining set.

Writes a sidecar JSON; never modifies the source hdf5 (they live on a shared drive and are the
inputs the baselines were built from). Feed sidecars to
`convert_hitl_hdf5_to_lerobot.py --olaf_sidecar`.

Action space (verified against robosuite, NOT joint velocity for the arm):
    right arm: OSC_POSE, input_type=delta, input_ref_frame=base, output_max=[0.05]*3 + [0.5]*3
    base:      JOINT_VELOCITY  (dims 7:11; the base never moves in these tasks)

Usage:
    python examples/robocasa/olaf_relabel.py --out_dir olaf_sidecars \
        --raw_dataset_path <demo>.hdf5 ... --demo_name demo_0
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import random
import re
import time

import h5py
import numpy as np
from openpi_client import image_tools
from scipy.spatial.transform import Rotation

# --- Two distinct scales. They are not interchangeable. ---------------------------------------
# The controller's `output_max`: one unit of commanded action asks for a 0.05 m target offset.
COMMANDED_M_PER_UNIT = 0.05
# What the end effector ACTUALLY moves in one 20 Hz step -- OSC does not reach the commanded
# offset within a single control step. Regressed over the 5 CoffeeSetupMug demos: slopes
# 0.0111 / 0.0108 / 0.0104 m per unit for x/y/z, Pearson r = 0.94 / 0.81 / 0.88.
# This is the one to use for "where does the gripper end up over the next N steps".
ACHIEVED_M_PER_UNIT = 0.0108

MAIN_IMAGE_KEY = "robot0_agentview_left_image"
WRIST_IMAGE_KEY = "robot0_eye_in_hand_image"
RESIZE = 224
SCHEMA_VERSION = 2
MAX_RETRIES = 3

_JSON_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_INT_RE = re.compile(r"-?\d+")

# Verbatim from the paper, Fig. 4a.
SYSTEM_PROMPT = (
    "You are a helpful assistant who is good at employing math and computer science tools to "
    "arrive at the solution. You analyze numerical values carefully and think step by step. "
    "You also pay close attention to the human language correction, interpret the human "
    "intention, and use it to arrive at the solution."
)

CONTEXT_PROMPT = """\
You are supervising a Franka Panda arm on a mobile base in a simulated kitchen.
An operational-space (OSC) end-effector controller runs at 20 Hz. Each step the arm receives a
7-number command:

  dx, dy, dz     relative end-effector translation along the robot-base axes
                 (+x forward, +y left, +z up), normalized and clipped to [-1, 1].
                 1.0 unit moves the end effector about {cm_per_unit:.1f} cm in one step.
  drx, dry, drz  relative end-effector rotation, axis-angle, same normalization.
  grip           > 0 closes the gripper, < 0 opens it.

The arm also has 4 mobile-base numbers and a control-mode flag; the base does not move during this
task, so they are held fixed and are not under discussion.

All positions below are in the robot base frame, in metres. Scene keypoints carry an attachment
label: "static" means fixed in the scene, "dynamic" means part of the object being manipulated,
and "robot" means the point is on the robot's own gripper.

The robot was asked to: "{task_language}"
"""

RELABEL_PROMPT = """\
Below is the moment just before a human operator took control, because the robot was going wrong.
You are choosing what the robot should have done over the next {window} steps instead.

Robot state:
  gripper position     x={gx:.3f}  y={gy:.3f}  z={gz:.3f}
  gripper orientation  roll={roll:.1f}  pitch={pitch:.1f}  yaw={yaw:.1f}  (degrees)
  gripper opening      {grip:.3f}

Scene keypoints:
{keypoint_block}

Over the next {window} steps the robot was about to move its gripper by
({ox:+.1f}, {oy:+.1f}, {oz:+.1f}) cm, ending up at:
{original_block}

The human operator, watching this exact moment, said:
"{correction}"

Here are {n_cand} alternative {window}-step plans the robot could execute instead. Each line gives
the plan's first command, how far it moves the gripper in total, and where that leaves the gripper
relative to each keypoint.

{candidate_block}

Pick the single plan that best carries out the human's correction.
Respond with JSON only: {{"index": <integer>, "reason": "<one sentence>"}}
"""


# ---------------------------------------------------------------------------------------------
# Copied from Arpitrf/semantic_corrections, `origin/point-set` branch. point-set is the branch that
# produced these extraction assets, and unlike `constraints-idea` it handles attachment="robot".
#   human_takeover_starts   <- semantic_corrections/hitl/keypoints_helper.py
#   load_acting_agent_labels <- same
#   load_ep_meta / get_task_prompt / load_semantic_annotations / semantics_for_segment
#                            <- semantic_corrections/constraint_extraction/io.py
# Inlined rather than carried as a second module: these are the only pieces needed.


def load_acting_agent_labels(hdf5_path: str | pathlib.Path, demo: str) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        raw = f[f"data/{demo}/acting_agent"][:]
    return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]


def human_takeover_starts(labels: list[str]) -> list[int]:
    """Frame indices where control switches from non-human -> human.

    Covers classic HITL ``robot->human`` and correction-intervene ``steered->human`` (the case on
    2 of the 5 CoffeeSetupMug demos), and any other non-human label.
    """
    return [i for i in range(1, len(labels)) if labels[i] == "human" and labels[i - 1] != "human"]


def load_ep_meta(hdf5_path: str | pathlib.Path, demo: str) -> dict:
    with h5py.File(hdf5_path, "r") as f:
        return json.loads(f[f"data/{demo}"].attrs["ep_meta"])


def get_task_prompt(hdf5_path: str | pathlib.Path, demo: str) -> str:
    meta = load_ep_meta(hdf5_path, demo)
    return str(meta.get("lang") or meta.get("language") or "")


def load_semantic_annotations(hdf5_path: str | pathlib.Path, demo: str) -> list[dict]:
    with h5py.File(hdf5_path, "r") as f:
        g = f[f"data/{demo}"]
        if "correction_semantic_annotations" not in g:
            return []
        return json.loads(g["correction_semantic_annotations"][()]).get("segments", [])


def semantics_for_segment(hdf5_path: str | pathlib.Path, demo: str, segment_index: int) -> str:
    anns = load_semantic_annotations(hdf5_path, demo)
    if not 0 <= segment_index < len(anns):
        raise IndexError(f"segment_index {segment_index} out of range (0..{len(anns) - 1})")
    return str(anns[segment_index].get("semantics", ""))


# ---------------------------------------------------------------------------------------------


def load_keypoints(path: pathlib.Path) -> list[dict]:
    """Load the extraction assets, reducing `geometry: region` point clouds to centroids."""
    data = json.loads(path.read_text())
    out = []
    for kp in data["keypoints"]:
        if "pos" not in kp:
            raise ValueError(
                f"{path} keypoint {kp.get('id')!r} has no 'pos' -- this needs assets already "
                "projected to 3D (extract_keypoints_objectives.py --visualize-3d)"
            )
        pos = np.atleast_2d(np.asarray(kp["pos"], dtype=np.float64)).mean(axis=0)
        out.append({"id": str(kp["id"]), "pos": pos, "attachment": str(kp.get("attachment", "static"))})
    return out


def state_from_obs(obs: h5py.Group, t: int) -> np.ndarray:
    """The 16-D proprio vector pi0 expects.

    Concatenation order must match convert_hitl_hdf5_to_lerobot.py's `_HDF5Episode`.
    """
    return np.concatenate(
        [
            obs["robot0_base_to_eef_pos"][t],
            obs["robot0_base_to_eef_quat"][t],
            obs["robot0_base_pos"][t],
            obs["robot0_base_quat"][t],
            obs["robot0_gripper_qpos"][t],
        ]
    ).astype(np.float64)


def build_obs(demo: h5py.Group, t: int, task_lang: str) -> dict:
    obs = demo["obs"]
    main = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs[MAIN_IMAGE_KEY][t], RESIZE, RESIZE))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(obs[WRIST_IMAGE_KEY][t], RESIZE, RESIZE))
    return {
        "observation/image": main,
        "observation/wrist_image": wrist,
        "observation/state": state_from_obs(obs, t),
        "prompt": task_lang,
    }


def sample_chunks(policy, obs: dict, window: int, n: int, cache: pathlib.Path, key: str) -> np.ndarray:
    """(n, window, 12) pi0 chunks, cached so reruns show the VLM byte-identical candidates.

    Truncating each 50-step chunk to the window mirrors semantic_corrections'
    `gather_steering_candidates`: `r["actions"][:replan_steps]`.
    """
    path = cache / f"{key}.npz"
    if path.exists():
        return np.load(path)["chunks"]
    chunks = np.stack([r["actions"][:window] for r in policy.infer_batch(obs, n)]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, chunks=chunks)
    return chunks


def gripper_after(start_pos: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Where the gripper ends up after executing `chunk`, robot base frame."""
    return start_pos + chunk[:, 0:3].sum(axis=0) * ACHIEVED_M_PER_UNIT


def _distances(pos: np.ndarray, keypoints: list[dict]) -> str:
    return "  ".join(f"{k['id']} {np.linalg.norm(pos - k['pos']) * 100:.1f}cm" for k in keypoints)


def render_prompt(
    *, window: int, state: np.ndarray, keypoints: list[dict], correction: str,
    recorded: np.ndarray, chunks: np.ndarray,
) -> str:
    gp = state[0:3]
    rpy = Rotation.from_quat(state[3:7]).as_euler("xyz", degrees=True)

    kp_lines = [
        f"  {k['id']:24s} x={k['pos'][0]:.3f} y={k['pos'][1]:.3f} z={k['pos'][2]:.3f}"
        f"   [{k['attachment']}]  {np.linalg.norm(gp - k['pos']) * 100:.1f}cm from the gripper"
        for k in keypoints
    ]
    rec_end = gripper_after(gp, recorded)
    rec_d = (rec_end - gp) * 100

    cand_lines = []
    for j, ch in enumerate(chunks):
        end = gripper_after(gp, ch)
        d = (end - gp) * 100
        a = ch[0]
        cand_lines.append(
            f"  [{j}] first command dx={a[0]:+.2f} dy={a[1]:+.2f} dz={a[2]:+.2f} "
            f"drx={a[3]:+.2f} dry={a[4]:+.2f} drz={a[5]:+.2f} "
            f"grip={'close' if a[6] > 0 else 'OPEN'}\n"
            f"      moves the gripper ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) cm  ->  {_distances(end, keypoints)}"
        )

    return RELABEL_PROMPT.format(
        window=window,
        gx=gp[0], gy=gp[1], gz=gp[2],
        roll=rpy[0], pitch=rpy[1], yaw=rpy[2],
        grip=state[14],
        keypoint_block="\n".join(kp_lines),
        ox=rec_d[0], oy=rec_d[1], oz=rec_d[2],
        original_block=f"      {_distances(rec_end, keypoints)}",
        correction=correction,
        n_cand=len(chunks),
        candidate_block="\n".join(cand_lines),
    )


# ---------------------------------------------------------------------------------------------
# Gemini. Same SDK call shape as semantic_corrections' scripts/extract_keypoints_objectives.py.


def _is_transient(exc: Exception) -> bool:
    """Only back off on transient failures. A 404 (bad model) never succeeds on retry."""
    text = str(exc)
    if any(c in text for c in ("429", "500", "502", "503", "504")):
        return True
    return not any(c in text for c in ("400", "401", "403", "404"))


def call_gemini(prompt: str, *, model: str, image_png: bytes) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("set GEMINI_API_KEY")
    parts = [types.Part(text=prompt), types.Part(inline_data=types.Blob(mime_type="image/png", data=image_png))]
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {"index": {"type": "INTEGER"}, "reason": {"type": "STRING"}},
                "required": ["index"],
            },
        ),
    )
    if not response.text:
        raise RuntimeError("empty Gemini response")
    return response.text


def parse_choice(text: str, n: int) -> tuple[int, str]:
    """Parse {"index": i, "reason": ...}; tolerate a ```json fence or a bare integer."""
    raw = text.strip()
    m = _JSON_RE.search(text)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
        idx, reason = int(data["index"]), str(data.get("reason", ""))
    except Exception:
        m = _INT_RE.search(raw)
        if not m:
            raise ValueError(f"no index in response: {text[:200]!r}") from None
        idx, reason = int(m.group()), ""
    if not 0 <= idx < n:
        raise ValueError(f"index {idx} out of range [0, {n})")
    return idx, reason


def ask_gemini(prompt: str, *, model: str, image_png: bytes, n: int, cache: pathlib.Path) -> tuple[int, str, str]:
    """Return (index, reason, raw_text), caching by prompt hash.

    A retry appends a corrective suffix, changing the hash, so a failed attempt never poisons the
    cache entry for the successful one. Exhausting retries raises: there is no fallback that
    silently keeps the original action, because that would produce a dataset labeled "OLAF" that
    partly is not.
    """
    suffix, last = "", None
    for attempt in range(1, MAX_RETRIES + 1):
        full = prompt + suffix
        key = hashlib.sha256(f"{model}\x00{full}".encode()).hexdigest()
        path = cache / f"{key}.json"
        if path.exists():
            text = json.loads(path.read_text())["text"]
        else:
            try:
                text = call_gemini(full, model=model, image_png=image_png)
            except SystemExit:
                raise
            except Exception as exc:
                if not _is_transient(exc):
                    raise
                last = exc
                time.sleep(min(2**attempt, 30) + random.random())
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text, "created_utc": _now()}))
        try:
            idx, reason = parse_choice(text, n)
            return idx, reason, text
        except ValueError as exc:
            last = exc
            suffix = (
                f'\n\nYour previous response was rejected: {exc}. Respond again with JSON only, as '
                f'{{"index": <integer between 0 and {n - 1}>, "reason": "<one sentence>"}}.'
            )
    raise RuntimeError(f"Gemini failed after {MAX_RETRIES} attempts: {last}")


# ---------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _png(arr: np.ndarray) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def process_demo(policy, hdf5_path: str, demo_name: str, args, out_dir: pathlib.Path) -> dict:
    hdf5_path = str(pathlib.Path(hdf5_path).resolve())
    stem = pathlib.Path(hdf5_path).stem
    kp_path = pathlib.Path(hdf5_path).parent / f"{stem}_assets" / "keypoints.json"
    if not kp_path.exists():
        raise FileNotFoundError(f"no extraction assets at {kp_path}")

    labels = load_acting_agent_labels(hdf5_path, demo_name)
    starts = human_takeover_starts(labels)
    if len(starts) != 1:
        raise ValueError(
            f"{hdf5_path} [{demo_name}]: expected exactly one human takeover, found {len(starts)} "
            f"at {starts}. OLAF relabeling is defined against the single annotated correction."
        )
    hi = starts[0]
    lo = max(0, hi - args.window)
    keypoints = load_keypoints(kp_path)
    correction = semantics_for_segment(hdf5_path, demo_name, 0)
    task_lang = get_task_prompt(hdf5_path, demo_name)

    kp_meta = json.loads(kp_path.read_text())
    if int(kp_meta.get("frame", lo)) != lo:
        print(f"  WARNING: keypoints.json frame={kp_meta.get('frame')} != window start {lo}")

    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_name]
        recorded = demo["actions"][lo:hi].astype(np.float64)
        state = state_from_obs(demo["obs"], lo)
        obs = build_obs(demo, lo, task_lang)
        image_png = _png(demo["obs"][MAIN_IMAGE_KEY][lo])
        num_frames = len(labels)

    key = hashlib.sha256(
        f"{args.checkpoint_dir}\x00{args.seed}\x00{hdf5_path}\x00{demo_name}\x00{lo}"
        f"\x00{args.num_candidates}\x00{args.window}".encode()
    ).hexdigest()
    chunks = sample_chunks(policy, obs, args.window, args.num_candidates,
                           out_dir / "cache" / "candidates", key)

    context = CONTEXT_PROMPT.format(cm_per_unit=ACHIEVED_M_PER_UNIT * 100, task_language=task_lang)
    prompt = context + "\n" + render_prompt(
        window=args.window, state=state, keypoints=keypoints, correction=correction,
        recorded=recorded, chunks=chunks,
    )
    print(f"  window [{lo},{hi})  labels={sorted({str(x) for x in labels[lo:hi]})}  "
          f"keypoints={[k['id'] for k in keypoints]}")

    if args.dry_run:
        chosen, reason, raw = None, None, None
        actions = [None] * (hi - lo)
    else:
        chosen, reason, raw = ask_gemini(
            prompt, model=args.vlm_model, image_png=image_png,
            n=len(chunks), cache=out_dir / "cache" / "vlm",
        )
        chunk = chunks[chosen].astype(np.float64)
        actions = []
        for i in range(hi - lo):
            a = recorded[i].copy()
            a[0:6] = chunk[i, 0:6]
            # dim 6 is a discrete latch: write the decisive value, not a raw sample.
            a[6] = 1.0 if chunk[i, 6] > 0 else -1.0
            actions.append(a.tolist())
        end = gripper_after(state[0:3], chunk)
        print(f"  chose [{chosen}] -- {reason[:80]}")
        print(f"  gripper ends at {_distances(end, keypoints)} "
              f"(recorded: {_distances(gripper_after(state[0:3], recorded), keypoints)})")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _now(),
        "source": {"hdf5": hdf5_path, "demo": demo_name, "num_frames": int(num_frames),
                   "keypoints_json": str(kp_path)},
        "window": {"start": int(lo), "end": int(hi), "preintv_window": args.window,
                   "agent_labels": [str(x) for x in labels[lo:hi]]},
        "policy": {"config_name": args.config_name, "checkpoint_dir": args.checkpoint_dir,
                   "seed": args.seed, "num_candidates": int(args.num_candidates)},
        "vlm": {"model": args.vlm_model, "temperature": 0.0,
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()},
        "task_language": task_lang,
        "correction_semantics": correction,
        "keypoints": [{"id": k["id"], "pos": k["pos"].tolist(), "attachment": k["attachment"]}
                      for k in keypoints],
        "selection": {"chosen_candidate_id": chosen, "reason": reason, "vlm_raw": raw},
        "candidates": chunks.astype(float).round(6).tolist(),
        "frames": [{"t": int(lo + i), "action": actions[i],
                    "original_action": recorded[i].tolist()} for i in range(hi - lo)],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="OLAF action relabeling for HITL demos")
    p.add_argument("--raw_dataset_path", nargs="+", required=True)
    p.add_argument("--demo_name", nargs="+", default=["demo_0"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--config_name", default="pi0_robocasa_pretrain_human300")
    p.add_argument("--checkpoint_dir", default=str(
        pathlib.Path("~/.cache/openpi/robocasa/robocasa365_checkpoints/pi0/"
                     "pi0_robocasa_pretrain_human300/multitask_learning/75000").expanduser()),
        help="Policy that samples candidates. The bootstrap checkpoint that produced these "
             "rollouts -- a HITL-finetuned one would leak the corrections into their own relabeling.")
    p.add_argument("--window", type=int, default=10,
                   help="Pre-intervention frames to relabel. Must match the converter's "
                        "--preintv_window.")
    p.add_argument("--num_candidates", type=int, default=50)
    p.add_argument("--vlm_model", default="gemini-robotics-er-2-preview")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true",
                   help="Sample and cache candidates, make no API calls (the diversity gate).")
    args = p.parse_args()

    if len(args.demo_name) == 1:
        demo_names = args.demo_name * len(args.raw_dataset_path)
    elif len(args.demo_name) == len(args.raw_dataset_path):
        demo_names = args.demo_name
    else:
        raise ValueError(
            f"--demo_name must be given once or once per --raw_dataset_path "
            f"({len(args.raw_dataset_path)} paths, got {len(args.demo_name)})"
        )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint_dir)

    for path, name in zip(args.raw_dataset_path, demo_names, strict=True):
        print(f"{path} [{name}]")
        sidecar = process_demo(policy, path, name, args, out_dir)
        parent = pathlib.Path(path).parent.name
        dest = out_dir / f"{parent}__{name}.olaf.json"
        dest.write_text(json.dumps(sidecar, indent=1))
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
