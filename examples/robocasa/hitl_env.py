"""RoboCasa / robosuite environment unwrapping helpers.

Copied verbatim from Arpitrf/semantic_corrections (semantic_corrections/hitl/env.py, `point-set`
branch @ c5c6cbe). No local fixes needed -- this branch already fixes upstream the wrapper-depth
bug an earlier version of this file (copied from the older `frs` branch) had to patch locally:
`get_robosuite_env` must unwrap the full gymnasium wrapper chain rather than stopping at the first
`hasattr(wrapper, "robots")` hit, since intermediate wrappers like `PassiveEnvChecker` proxy
attribute access through to the real env underneath.
"""

import pathlib
import re


def infer_env_name_from_path(hdf5_path) -> str | None:
    """Guess the robocasa env name from a demo path, e.g. ``.../pi0_hitl/CoffeeSetupMug/...``."""
    parts = pathlib.Path(hdf5_path).parts
    for i, part in enumerate(parts):
        if part in ("pretrain", "target", "pi0_hitl") and i + 1 < len(parts):
            candidate = parts[i + 1]
            if re.match(r"^[A-Z][A-Za-z0-9]+$", candidate):
                return candidate
    for part in parts:
        if re.match(r"^[A-Z][A-Za-z0-9]+$", part) and part not in ("System",):
            return part
    return None


def bootstrap_replay_env(
    hdf5_path,
    demo: str,
    frame: int = 0,
    *,
    env_name: str | None = None,
    obj_instance_split: str | None = None,
    camera_size: int = 512,
):
    """Build an env and restore one recorded frame, for offline sim-based analysis.

    Layout and style come from the demo's own ``ep_meta``, so no scene config is needed.
    Returns ``(env, raw_env)``; the caller owns ``env.close()``.
    """
    import gymnasium as gym

    import robocasa  # noqa: F401 -- registers gym environments
    from semantic_corrections.utils.sim_state import load_sim_state_from_hdf5, restore_sim_state

    env_name = env_name or infer_env_name_from_path(hdf5_path)
    if env_name is None:
        raise ValueError(f"could not infer env name from {hdf5_path}; pass env_name")

    env = gym.make(
        f"robocasa/{env_name}",
        split=None,
        obj_instance_split=obj_instance_split,
        camera_widths=camera_size,
        camera_heights=camera_size,
    )
    raw_env = get_robosuite_env(env)
    while isinstance(raw_env, gym.Wrapper):
        raw_env = raw_env.env

    xml, ep_meta, state, ctrl, gripper = load_sim_state_from_hdf5(str(hdf5_path), demo, frame)
    restore_sim_state(raw_env, xml, state, ctrl, gripper, ep_meta=ep_meta)
    mark_gym_env_reset(env)
    return env, raw_env


def get_robosuite_env(gym_env):
    """Walk the gymnasium wrapper chain to the underlying robosuite/robocasa env.

    Must unwrap fully: intermediate wrappers (e.g. ``PassiveEnvChecker``) proxy
    attributes like ``robots``, so stopping at the first ``robots`` hit leaves
    setattr (e.g. ``layout_and_style_ids``) on the wrapper and does not affect
    the real Kitchen.
    """
    raw = gym_env
    while hasattr(raw, "env"):
        raw = raw.env
    return raw


def get_robocasa_gym_wrapper(gym_env):
    """Walk the gymnasium wrapper chain to find the robocasa gym wrapper
    that exposes ``get_observation`` for converting raw obs to gym format."""
    raw = gym_env
    while hasattr(raw, "env"):
        if hasattr(raw, "get_observation"):
            return raw
        raw = raw.env
    return None


def mark_gym_env_reset(gym_env):
    """Mark the gymnasium wrapper chain as reset after a manual sim state restore.

    Gymnasium's ``OrderEnforcing`` wrapper raises if ``step()`` is called without
    ``reset()``. After restoring state directly on the underlying robosuite env,
    call this so ``env.step()`` works.
    """
    wrapper = gym_env
    while hasattr(wrapper, "env"):
        if hasattr(wrapper, "_has_reset"):
            wrapper._has_reset = True
        wrapper = wrapper.env


def refresh_gym_observation(gym_env, raw_env, gym_wrapper):
    """Rebuild gym observation after MJCF reload or other raw sim mutation."""
    mark_gym_env_reset(gym_env)
    raw_obs = raw_env._get_observations(force_update=True)
    return gym_wrapper.get_observation(raw_obs)