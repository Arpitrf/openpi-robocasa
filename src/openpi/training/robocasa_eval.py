"""Roll out the *live* training policy in RoboCasa from a fixed sim state, mid-run.

Used by scripts/train.py when `TrainConfig.eval_interval` is set: every N steps the current
parameters are rolled out in the simulator from the same recorded state the round was collected
from, and the videos are logged to the run's wandb page. Unlike
examples/robocasa/eval_hgdagger_checkpoints.py, which loads *saved checkpoints* afterwards, this
sees every N steps whether or not a checkpoint was written there, and its wandb points land on
the training run's own step axis.

Two implementation details worth knowing:

  * The sampler is jitted **once**, with the parameters passed in as an argument rather than
    closed over. `openpi.shared.nnx_utils.module_jit` (what `Policy` uses) freezes module state at
    construction, so a `Policy` built from step 100's params would keep evaluating those params
    forever; rebuilding one per eval would instead recompile -- and leak an executable -- every
    time. Passing `params` as an argument keeps one compilation for the whole run.
  * RoboCasa/robosuite and the hdf5 state-restore helpers live in examples/robocasa (copied there
    from Arpitrf/semantic_corrections). They are imported by path rather than copied a third time.
"""

import collections
import dataclasses
import logging
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import openpi.models.model as _model
import openpi.transforms as _transforms

logger = logging.getLogger(__name__)

_EXAMPLES_ROBOCASA = pathlib.Path(__file__).parents[3] / "examples" / "robocasa"


@dataclasses.dataclass(frozen=True)
class RolloutResult:
    success: bool
    steps: int
    video_path: pathlib.Path


class RobocasaEvaluator:
    """Builds the env once, then rolls out arbitrary parameter sets against it."""

    def __init__(
        self,
        *,
        init_state_hdf5: str,
        env_name: str,
        model_def,
        data_config,
        sample_kwargs: dict | None = None,
        demo_name: str = "demo_0",
        frame: int = 0,
        horizon: int | None = None,
        replan_steps: int = 5,
        resize_size: int = 224,
        camera_size: int = 512,
        video_stride: int = 2,
        video_fps: int = 20,
        seed: int = 7,
    ):
        if str(_EXAMPLES_ROBOCASA) not in sys.path:
            sys.path.insert(0, str(_EXAMPLES_ROBOCASA))

        self._init_state_hdf5 = str(init_state_hdf5)
        self._env_name = env_name
        self._demo_name = demo_name
        self._frame = frame
        self._replan_steps = replan_steps
        self._resize_size = resize_size
        self._camera_size = camera_size
        self._video_stride = video_stride
        self._video_fps = video_fps
        self._seed = seed
        self._env = None
        self._rng = jax.random.key(seed)

        if data_config.norm_stats is None:
            raise ValueError(
                "eval_interval is set but the data config has no norm stats -- run "
                "scripts/compute_norm_stats.py for this config first."
            )
        self._input_transform = _transforms.compose(
            [
                *data_config.data_transforms.inputs,
                _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._output_transform = _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )

        sample_kwargs = sample_kwargs or {}

        # Compiled once for the whole run: `params` is an argument, not a closure (see docstring).
        @jax.jit
        def _sample(params, rng, observation):
            model = nnx.merge(model_def, params)
            return model.sample_actions(rng, observation, **sample_kwargs)

        self._sample = _sample
        self._horizon = horizon

    def _ensure_env(self):
        if self._env is not None:
            return
        import gymnasium as gym
        from hitl_env import get_robocasa_gym_wrapper, get_robosuite_env
        from robocasa.utils.dataset_registry_utils import get_task_horizon

        logger.info("Building RoboCasa env %s for in-training eval...", self._env_name)
        self._env = gym.make(
            f"robocasa/{self._env_name}",
            split=None,
            seed=self._seed,
            camera_widths=self._camera_size,
            camera_heights=self._camera_size,
        )
        self._raw_env = get_robosuite_env(self._env)
        self._gym_wrapper = get_robocasa_gym_wrapper(self._env)
        if self._horizon is None:
            self._horizon = int(get_task_horizon(self._env_name) * 1.5)
        logger.info("Eval env ready (horizon %d)", self._horizon)

    def _infer(self, params, obs_dict: dict) -> np.ndarray:
        inputs = self._input_transform(jax.tree.map(lambda x: x, obs_dict))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample(params, sample_rng, _model.Observation.from_dict(inputs))
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), {"state": inputs["state"], "actions": actions})
        return self._output_transform(outputs)["actions"]

    def rollout(self, params, out_dir: pathlib.Path, num_rollouts: int) -> list[RolloutResult]:
        """Roll `params` out `num_rollouts` times from the fixed init state, writing one mp4 each."""
        self._ensure_env()
        import imageio
        from hitl_env import mark_gym_env_reset
        from hitl_obs import obs_to_camera_frames, obs_to_policy_state
        from openpi_client import image_tools
        from robocasa.utils.env_utils import convert_action
        from sim_state import restore_sim_state_from_hdf5

        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i in range(num_rollouts):
            self._env.reset()
            restore_sim_state_from_hdf5(
                self._raw_env, self._init_state_hdf5, demo_name=self._demo_name, frame=self._frame
            )
            mark_gym_env_reset(self._env)
            obs = self._gym_wrapper.get_observation(self._raw_env._get_observations(force_update=True))
            task_lang = obs["annotation.human.task_description"]

            action_plan = collections.deque()
            frames, done, t = [], False, 0
            while t < self._horizon and not done:
                main_img, wrist_img = obs_to_camera_frames(obs)
                if not action_plan:
                    chunk = self._infer(
                        params,
                        {
                            "observation/image": image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(main_img, self._resize_size, self._resize_size)
                            ),
                            "observation/wrist_image": image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_img, self._resize_size, self._resize_size)
                            ),
                            "observation/state": obs_to_policy_state(obs),
                            "prompt": task_lang,
                        },
                    )
                    action_plan.extend(chunk[: self._replan_steps])
                obs, _, _, _, info = self._env.step(convert_action(action_plan.popleft()))
                done = bool(info.get("success", False))
                if t % self._video_stride == 0 or done:
                    frames.append(image_tools.convert_to_uint8(np.ascontiguousarray(self._env.render())))
                t += 1

            path = out_dir / f"rollout_{i}_{'success' if done else 'failure'}.mp4"
            imageio.mimwrite(str(path), [np.asarray(f) for f in frames], fps=self._video_fps)
            results.append(RolloutResult(success=done, steps=t, video_path=path))
            logger.info("eval rollout %d: success=%s steps=%d -> %s", i, done, t, path.name)
        return results

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
