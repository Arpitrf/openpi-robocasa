"""RoboCasa / robosuite environment unwrapping helpers.

Copied from Arpitrf/semantic_corrections (semantic_corrections/hitl/env.py, frs branch), with
get_robosuite_env / get_robocasa_gym_wrapper rewritten below. The original walked the chain
via `hasattr(raw, "env")` / `hasattr(raw, "robots")`, which returns 2 levels too early in this
repo's env (gymnasium==0.29.1): `Wrapper.__getattr__` delegates attribute *access* through the
whole chain, so `hasattr(some_outer_wrapper, "robots")` is True as soon as anything further in
has it -- confirmed by instrumenting the actual chain here (gym.make gives
OrderEnforcing -> PassiveEnvChecker -> RoboCasaGymEnv -> raw robosuite env; the original stopped
at PassiveEnvChecker). `.unwrapped` is gymnasium's own reliable way to skip every wrapper in one
step, so use that instead of a manual walk.
"""


def get_robocasa_gym_wrapper(gym_env):
    """Return the RoboCasaGymEnv wrapper, which exposes ``get_observation`` for converting raw
    obs to gym format."""
    return gym_env.unwrapped


def get_robosuite_env(gym_env):
    """Return the underlying raw robosuite env (RoboCasaGymEnv wraps it as ``.env``)."""
    return gym_env.unwrapped.env


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
