"""MuJoCo weld equality helpers: attach a task object rigidly to the robot EEF.

Copied from Arpitrf/semantic_corrections (semantic_corrections/hitl/mug_weld.py), like
hitl_env.py / hitl_obs.py / sim_state.py in this directory. One change: the `capture_sim_state`
import points at this directory's sim_state.py copy instead of the semantic_corrections package,
which is not importable from here.

Used by the eval paths so that rollouts run under the same contact dynamics the demos were
collected under -- run_pi0_hitl.py welds the mug to the EEF on grasp (task.weld_on_grasp), and an
eval without the weld is grading the policy on a physically different task.
"""

from __future__ import annotations

import dataclasses
import logging
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from sim_state import capture_sim_state


@dataclasses.dataclass(frozen=True)
class WeldBodies:
    """Resolved MJCF body names for a mug/object ↔ EEF weld."""

    object_name: str
    arm: str
    eef_body: str
    object_body: str


@dataclasses.dataclass(frozen=True)
class WeldResult:
    """Outcome of ``apply_object_eef_weld``."""

    bodies: WeldBodies
    weld_name: str
    relpose: str
    pos_error_m: float
    quat_error: float


def resolve_weld_bodies(
    raw_env,
    *,
    object_name: str = "obj",
    arm: str | None = None,
    eef_body: str | None = None,
) -> WeldBodies:
    """Return gripper and object root body names in the compiled sim model."""
    robot = raw_env.robots[0]
    arm = arm or robot.arms[0]
    if object_name not in raw_env.objects:
        known = ", ".join(sorted(raw_env.objects.keys()))
        raise KeyError(f"object {object_name!r} not in env.objects ({known})")

    resolved_eef = eef_body or robot.robot_model.eef_name[arm]
    object_body = raw_env.objects[object_name].root_body
    sim = raw_env.sim
    for label, body in (("eef", resolved_eef), ("object", object_body)):
        try:
            sim.model.body_name2id(body)
        except ValueError as exc:
            raise ValueError(f"unknown {label} body {body!r} in sim model") from exc
    return WeldBodies(
        object_name=object_name,
        arm=arm,
        eef_body=resolved_eef,
        object_body=object_body,
    )


def _body_pose_world(sim, body_name: str) -> tuple[np.ndarray, np.ndarray]:
    body_id = sim.model.body_name2id(body_name)
    pos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
    rot = np.asarray(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
    return pos, rot


def compute_weld_relpose(sim, body1: str, body2: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Compute MuJoCo ``relpose`` string (body2 relative to body1) from the live sim."""
    from robosuite.utils import transform_utils as T

    pos1, rot1 = _body_pose_world(sim, body1)
    pos2, rot2 = _body_pose_world(sim, body2)

    t_world_1 = np.eye(4, dtype=np.float64)
    t_world_1[:3, :3] = rot1
    t_world_1[:3, 3] = pos1
    t_world_2 = np.eye(4, dtype=np.float64)
    t_world_2[:3, :3] = rot2
    t_world_2[:3, 3] = pos2

    t_1_2 = np.linalg.inv(t_world_1) @ t_world_2
    rel_pos = t_1_2[:3, 3]
    rel_quat_xyzw = T.mat2quat(t_1_2[:3, :3])
    relpose = format_weld_relpose(rel_pos, rel_quat_xyzw)
    return relpose, rel_pos, rel_quat_xyzw


def format_weld_relpose(pos: np.ndarray, quat_xyzw: np.ndarray) -> str:
    """Format ``relpose`` for MJCF: ``x y z qw qx qy qz`` (MuJoCo w-first quaternion)."""
    p = np.asarray(pos, dtype=np.float64).reshape(3)
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return (
        f"{p[0]:.9g} {p[1]:.9g} {p[2]:.9g} "
        f"{q[3]:.9g} {q[0]:.9g} {q[1]:.9g} {q[2]:.9g}"
    )


def inject_weld_equality(
    xml_str: str,
    *,
    name: str,
    body1: str,
    body2: str,
    relpose: str,
    active: bool = True,
    solref: str = "0.02 1",
) -> str:
    """Insert a ``<weld>`` under ``<equality>``, replacing any that couples the same pair.

    Demos recorded with weld-on-grasp bake their weld into the saved model XML, so a
    restored scene can already hold the object. Leaving that constraint in place would
    give the object two rigid attachments with different ``relpose`` values, which fight
    each other and keep the object stuck when only the injected weld is released.
    """
    root = ET.fromstring(xml_str)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    pair = {body1, body2}
    for existing in list(equality):
        same_name = existing.get("name") == name
        couples_pair = existing.tag in ("weld", "connect") and {
            existing.get("body1"),
            existing.get("body2"),
        } == pair
        if not (same_name or couples_pair):
            continue
        equality.remove(existing)
        if not same_name:
            logging.info(
                "Replacing pre-existing <%s> %r coupling %s <-> %s",
                existing.tag,
                existing.get("name"),
                body1,
                body2,
            )

    weld = ET.SubElement(equality, "weld")
    weld.set("name", name)
    weld.set("body1", body1)
    weld.set("body2", body2)
    weld.set("relpose", relpose)
    weld.set("active", "true" if active else "false")
    weld.set("solref", solref)
    return ET.tostring(root, encoding="unicode")


def _restore_snap(raw_env, snap: dict[str, Any]) -> None:
    raw_env.sim.set_state_from_flattened(snap["state"])
    if snap.get("ctrl") is not None:
        raw_env.sim.data.ctrl[:] = snap["ctrl"]
    gripper_actions = snap.get("gripper_actions") or {}
    for robot in raw_env.robots:
        for arm in robot.arms:
            if robot.has_gripper[arm] and arm in gripper_actions:
                robot.gripper[arm].current_action = gripper_actions[arm].copy()
    raw_env.sim.forward()


def _weld_pose_error(sim, body1: str, body2: str, target_relpos: np.ndarray, target_quat_xyzw: np.ndarray) -> tuple[float, float]:
    from robosuite.utils import transform_utils as T

    _, rel_pos, rel_quat = compute_weld_relpose(sim, body1, body2)
    pos_err = float(np.linalg.norm(rel_pos - target_relpos))
    # Quaternion sign ambiguity: compare both signs.
    q_err_a = float(np.linalg.norm(rel_quat - target_quat_xyzw))
    q_err_b = float(np.linalg.norm(rel_quat + target_quat_xyzw))
    return pos_err, min(q_err_a, q_err_b)


def find_active_object_eef_weld(
    raw_env,
    *,
    weld_name: str | None = None,
    object_name: str = "obj",
    eef_body: str | None = None,
) -> str | None:
    """Return the name of an active object↔EEF weld/connect, or ``None``.

    Prefers ``weld_name`` when that constraint is active; otherwise any rigid
    equality on the same body pair (HITL demos bake ``task.weld_name`` into XML).
    """
    import mujoco

    if object_name not in raw_env.objects:
        # Task has no such object (e.g. a fixture-only task like a door) -- trivially
        # no weld can be active on it.
        return None
    bodies = resolve_weld_bodies(raw_env, object_name=object_name, eef_body=eef_body)
    sim = raw_env.sim
    model = sim.model._model
    data = sim.data._data
    pair = {
        int(sim.model.body_name2id(bodies.eef_body)),
        int(sim.model.body_name2id(bodies.object_body)),
    }
    rigid_types = (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT))
    objtype = getattr(model, "eq_objtype", None)

    preferred_id = -1
    if weld_name:
        preferred_id = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
        )

    found: list[tuple[int, str]] = []
    for i in range(int(model.neq)):
        if int(model.eq_type[i]) not in rigid_types:
            continue
        if objtype is not None and int(objtype[i]) != int(mujoco.mjtObj.mjOBJ_BODY):
            continue
        if {int(model.eq_obj1id[i]), int(model.eq_obj2id[i])} != pair:
            continue
        if not data.eq_active[i]:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or f"eq[{i}]"
        found.append((i, name))

    for eq_id, name in found:
        if eq_id == preferred_id:
            return name
    return found[0][1] if found else None


def set_weld_active(raw_env, weld_name: str, *, active: bool) -> None:
    """Toggle an existing weld equality constraint without reloading MJCF."""
    import mujoco

    sim = raw_env.sim
    model = sim.model._model
    data = sim.data._data
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if eq_id < 0:
        raise ValueError(f"equality constraint {weld_name!r} not found in model")
    # Runtime flag: eq_active0 only affects mj_resetData, not the current episode.
    data.eq_active[eq_id] = bool(active)
    model.eq_active0[eq_id] = bool(active)
    sim.forward()


def break_object_eef_weld(raw_env, weld_name: str) -> None:
    """Disable every rigid equality coupling the welded body pair.

    Breaking only ``weld_name`` is not enough: the restored model can carry additional
    welds between the same object and EEF (e.g. baked into a demo recorded with
    weld-on-grasp), and any one of them left active keeps the object attached.
    """
    import mujoco

    sim = raw_env.sim
    model = sim.model._model
    data = sim.data._data
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
    if eq_id < 0:
        raise ValueError(f"equality constraint {weld_name!r} not found in model")

    rigid_types = (int(mujoco.mjtEq.mjEQ_WELD), int(mujoco.mjtEq.mjEQ_CONNECT))
    objtype = getattr(model, "eq_objtype", None)
    body_pair = {int(model.eq_obj1id[eq_id]), int(model.eq_obj2id[eq_id])}

    released: list[str] = []
    for i in range(int(model.neq)):
        if int(model.eq_type[i]) not in rigid_types:
            continue
        if objtype is not None and int(objtype[i]) != int(mujoco.mjtObj.mjOBJ_BODY):
            continue
        if {int(model.eq_obj1id[i]), int(model.eq_obj2id[i])} != body_pair:
            continue
        if not data.eq_active[i]:
            continue
        # eq_active0 only affects mj_resetData; eq_active is the live flag.
        data.eq_active[i] = False
        model.eq_active0[i] = False
        released.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or f"eq[{i}]")
    sim.forward()
    logging.info("Broke object-EEF weld(s): %s", ", ".join(repr(n) for n in released) or "none active")


def apply_object_eef_weld(
    raw_env,
    *,
    object_name: str = "obj",
    arm: str | None = None,
    eef_body: str | None = None,
    weld_name: str = "debug_mug_eef_weld",
    active: bool = True,
    solref: str = "0.02 1",
) -> WeldResult:
    """Inject a weld between object root body and EEF, then reload MJCF and restore state.

    ``relpose`` is computed from the **current** sim poses so the object does not jump
    when the constraint is enabled.
    """
    bodies = resolve_weld_bodies(
        raw_env,
        object_name=object_name,
        arm=arm,
        eef_body=eef_body,
    )
    snap = capture_sim_state(raw_env)
    relpose, rel_pos, rel_quat = compute_weld_relpose(
        raw_env.sim,
        bodies.eef_body,
        bodies.object_body,
    )
    xml_weld = inject_weld_equality(
        raw_env.sim.model.get_xml(),
        name=weld_name,
        body1=bodies.eef_body,
        body2=bodies.object_body,
        relpose=relpose,
        active=active,
        solref=solref,
    )

    logging.info(
        "Applying weld %r: %s -> %s relpose=%s",
        weld_name,
        bodies.eef_body,
        bodies.object_body,
        relpose,
    )
    raw_env.reset_from_xml_string(xml_weld)
    _restore_snap(raw_env, snap)

    pos_err, quat_err = _weld_pose_error(
        raw_env.sim,
        bodies.eef_body,
        bodies.object_body,
        rel_pos,
        rel_quat,
    )
    logging.info("Post-weld relative pose error: pos=%.6f m quat=%.6f", pos_err, quat_err)
    return WeldResult(
        bodies=bodies,
        weld_name=weld_name,
        relpose=relpose,
        pos_error_m=pos_err,
        quat_error=quat_err,
    )
