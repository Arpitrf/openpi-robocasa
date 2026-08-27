"""Capture, persist, and restore robosuite simulator state.

Copied verbatim from Arpitrf/semantic_corrections (semantic_corrections/utils/sim_state.py,
frs branch) -- restore_sim_state here also restores ctrl/gripper_actions, which the reset_to()
this repo previously hand-rolled in eval_fixed_scene.py did not. That matters for replaying a
mid-episode (e.g. post-grasp) state, less so for frame 0.
"""

from __future__ import annotations

import json

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


def restore_sim_state_from_hdf5(raw_env, path, demo_name="demo_0", frame=0):
    """Restore the simulator to the state stored at ``frame`` in an HDF5 demo."""
    # NOTE: fixed from the original -- `restore_sim_state(raw_env, *load_sim_state_from_hdf5(...))`
    # silently mis-ordered arguments (load_sim_state_from_hdf5 returns
    # (xml, ep_meta, state, ctrl, gripper_actions); restore_sim_state expects
    # (xml, state, ctrl, gripper_actions, ep_meta=...)). This function is never actually called
    # anywhere in semantic_corrections -- run_pi0_hitl.py unpacks load_sim_state_from_hdf5's
    # return by name and calls restore_sim_state with explicit ep_meta=, which is correct; this
    # convenience wrapper just never got exercised the same way. Found by running this against
    # real data, not by inspection.
    xml, ep_meta, state, ctrl, gripper_actions = load_sim_state_from_hdf5(path, demo_name, frame)
    restore_sim_state(raw_env, xml, state, ctrl, gripper_actions, ep_meta=ep_meta)
