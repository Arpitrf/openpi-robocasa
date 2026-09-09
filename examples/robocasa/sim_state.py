"""Capture, persist, and restore robosuite simulator state.

Copied verbatim from Arpitrf/semantic_corrections (semantic_corrections/utils/sim_state.py,
`point-set` branch @ c5c6cbe -- more current than `frs`, and already fixes upstream a bug `frs`
had), with one fix: `restore_sim_state_from_hdf5` (bottom of this file) still mis-orders its call
to `restore_sim_state` -- `load_sim_state_from_hdf5` returns
`(xml, ep_meta, state, ctrl, gripper_actions)`, but `restore_sim_state` takes
`(raw_env, saved_xml, saved_state, saved_ctrl, saved_gripper_actions, ep_meta=None)`, so
positional unpacking puts `ep_meta` where `saved_state` belongs. Confirmed unfixed on `point-set`
too. This function is never actually called within semantic_corrections itself (its own
`run_pi0_hitl.py` unpacks and calls `restore_sim_state` directly, correctly) -- found by running
this against real data, not by inspection.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np


def capture_gripper_actions(raw_env) -> dict[str, np.ndarray]:
    gripper_actions = {}
    for robot in raw_env.robots:
        for arm in robot.arms:
            if robot.has_gripper[arm]:
                gripper_actions[arm] = robot.gripper[arm].current_action.copy()
    return gripper_actions


def capture_sim_state(raw_env) -> dict:
    """Snapshot the simulator state aligned with the current observation."""
    return {
        "state": raw_env.sim.get_state().flatten().copy(),
        "ctrl": raw_env.sim.data.ctrl.copy(),
        "gripper_actions": capture_gripper_actions(raw_env),
    }


def capture_episode_metadata(raw_env) -> dict:
    return {
        "model_file": raw_env.sim.model.get_xml(),
        "ep_meta": raw_env.get_ep_meta(),
    }


def capture_retry_checkpoint(raw_env):
    """Return sim components needed to restore an episode for retry."""
    meta = capture_episode_metadata(raw_env)
    snap = capture_sim_state(raw_env)
    return meta["model_file"], snap["state"], snap["ctrl"], snap["gripper_actions"]


def restore_sim_state(
    raw_env,
    saved_xml,
    saved_state,
    saved_ctrl,
    saved_gripper_actions,
    ep_meta=None,
):
    """Restore simulator state including grasp from the saved components.

    When ``ep_meta`` is provided (cross-session loads), the environment's
    internal model is rebuilt to match the saved layout before loading the XML.
    """
    if ep_meta is not None:
        raw_env.set_ep_meta(ep_meta)
        raw_env.reset()
        saved_xml = raw_env.edit_model_xml(saved_xml)
    raw_env.reset_from_xml_string(saved_xml)
    raw_env.sim.set_state_from_flattened(saved_state)
    raw_env.sim.forward()
    if saved_ctrl is not None:
        raw_env.sim.data.ctrl[:] = saved_ctrl
    if saved_gripper_actions is not None:
        for robot in raw_env.robots:
            for arm in robot.arms:
                if robot.has_gripper[arm] and arm in saved_gripper_actions:
                    robot.gripper[arm].current_action = saved_gripper_actions[arm].copy()
    # for _ in range(10):
    #     raw_env.sim.step()


def save_sim_state_to_disk(path, raw_env):
    """Persist the full simulator state (including grasp) to an .npz file."""
    snap = capture_sim_state(raw_env)
    meta = capture_episode_metadata(raw_env)
    gripper_kv = {
        f"gripper_action_{arm}": action for arm, action in snap["gripper_actions"].items()
    }
    np.savez(
        path,
        xml=np.array(meta["model_file"]),
        ep_meta=np.array(json.dumps(meta["ep_meta"])),
        state=snap["state"],
        ctrl=snap["ctrl"],
        **gripper_kv,
    )


def load_sim_state_from_disk(path):
    """Load a state snapshot previously saved with ``save_sim_state_to_disk``.

    Returns (xml_string, ep_meta, state, ctrl, gripper_actions_dict).
    """
    data = np.load(path, allow_pickle=True)
    xml = str(data["xml"])
    ep_meta = json.loads(str(data["ep_meta"])) if "ep_meta" in data else None
    state = data["state"]
    ctrl = data["ctrl"]
    gripper_actions = {}
    for key in data.files:
        if key.startswith("gripper_action_"):
            arm = key[len("gripper_action_") :]
            gripper_actions[arm] = data[key]
    return xml, ep_meta, state, ctrl, gripper_actions


def save_initial_state_hdf5(
    path: str | os.PathLike,
    raw_env,
    *,
    demo_name: str = "demo_0",
    task_lang: str | None = None,
) -> None:
    """Persist a single-frame states-only HDF5 compatible with ``load_sim_state_from_hdf5``."""
    meta = capture_episode_metadata(raw_env)
    snap = capture_sim_state(raw_env)
    out = os.fspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    ep_meta = dict(meta["ep_meta"] or {})
    if task_lang and "lang" not in ep_meta:
        ep_meta.setdefault("lang", task_lang)

    with h5py.File(out, "w") as f:
        grp = f.create_group(f"data/{demo_name}")
        grp.create_dataset("states", data=np.asarray([snap["state"]]))
        grp.create_dataset("ctrl", data=np.asarray([snap["ctrl"]]))
        if snap["gripper_actions"]:
            gripper_grp = grp.create_group("gripper_actions")
            for arm, action in snap["gripper_actions"].items():
                gripper_grp.create_dataset(arm, data=np.asarray([action]))
        grp.attrs["model_file"] = meta["model_file"]
        grp.attrs["ep_meta"] = json.dumps(ep_meta)
        grp.attrs["num_samples"] = 1
        grp.attrs["recorded_cameras"] = json.dumps([])


def load_sim_state_from_hdf5(path, demo_name="demo_0", frame=0):
    """Load simulator state for one frame from a HITL HDF5 demo.

    Returns (xml_string, ep_meta, state, ctrl, gripper_actions_dict).
    """
    with h5py.File(path, "r") as f:
        grp = f[f"data/{demo_name}"]
        if "states" not in grp:
            raise KeyError(f"{path} demo {demo_name} has no simulator states")

        num_frames = grp["states"].shape[0]
        if frame < 0 or frame >= num_frames:
            raise IndexError(
                f"frame {frame} out of range for {path} ({demo_name} has {num_frames} frames)"
            )

        xml = grp.attrs["model_file"]
        ep_meta = json.loads(grp.attrs["ep_meta"])
        state = grp["states"][frame]
        ctrl = grp["ctrl"][frame]

        gripper_actions = {}
        if "gripper_actions" in grp:
            for arm in grp["gripper_actions"]:
                gripper_actions[arm] = grp["gripper_actions"][arm][frame]

    return xml, ep_meta, state, ctrl, gripper_actions


def object_is_grasped(raw_env, obj_name: str = "obj") -> bool:
    """Whether ``obj_name`` is currently held (RoboCasa contact + closed gripper)."""
    if raw_env is None or obj_name not in getattr(raw_env, "objects", {}):
        return False
    from robocasa.utils.object_utils import check_obj_grasped

    return bool(check_obj_grasped(raw_env, obj_name))


def first_grasp_frame_from_hdf5(
    path,
    demo_name="demo_0",
    *,
    sustained_steps: int = 5,
    gripper_closed_threshold: float = 0.0,
) -> int:
    """Approximate first grasp as the first sustained gripper-close command.

    HITL actions use index 6 for the gripper channel (``>0`` ≈ closed). This is a
    heuristic — not the online ``check_obj_grasped`` contact check.
    """
    with h5py.File(path, "r") as f:
        grp = f[f"data/{demo_name}"]
        if "actions" not in grp:
            raise KeyError(f"{path} demo {demo_name} has no actions")
        gripper = np.asarray(grp["actions"][:, 6], dtype=float)
    need = max(1, int(sustained_steps))
    if gripper.shape[0] < need:
        raise IndexError(
            f"demo {demo_name} in {path} has only {gripper.shape[0]} frames; "
            f"need >= {need} for first_grasp"
        )
    closed = gripper > float(gripper_closed_threshold)
    for i in range(gripper.shape[0] - need + 1):
        if bool(np.all(closed[i : i + need])):
            return int(i)
    raise ValueError(
        f"no first_grasp found in {path} ({demo_name}): "
        f"no {need} consecutive frames with actions[:,6] > {gripper_closed_threshold}"
    )


def resolve_bootstrap_frame(
    *,
    path,
    demo_name: str,
    frame_mode: str,
    frame: int,
) -> int:
    """Resolve bootstrap frame from ``fixed`` or ``first_grasp`` mode."""
    mode = str(frame_mode).strip().lower()
    if mode == "fixed":
        return int(frame)
    if mode == "first_grasp":
        return first_grasp_frame_from_hdf5(path, demo_name=demo_name)
    raise ValueError(
        f"bootstrap.load_state_frame_mode must be 'fixed' or 'first_grasp', got {frame_mode!r}"
    )


def restore_sim_state_from_hdf5(raw_env, path, demo_name="demo_0", frame=0):
    """Restore the simulator to the state stored at ``frame`` in an HDF5 demo."""
    xml, ep_meta, state, ctrl, gripper_actions = load_sim_state_from_hdf5(path, demo_name, frame)
    restore_sim_state(raw_env, xml, state, ctrl, gripper_actions, ep_meta=ep_meta)