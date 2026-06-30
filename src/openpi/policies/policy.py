from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _unbatch_frs_diagnostics(diagnostics: dict) -> dict:
    """Convert batched JAX diagnostics to unbatched numpy for the client."""

    def convert(x):
        arr = np.asarray(x)
        if arr.ndim > 0 and arr.shape[0] == 1:
            return arr[0, ...]
        return float(arr)

    return {key: convert(value) for key, value in diagnostics.items()}


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_actions_frs = (
            nnx_utils.module_jit(model.sample_actions_frs)
            if hasattr(model, "sample_actions_frs")
            else None
        )
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        # A "reference" sub-dict (reference scene observation + reference action)
        # triggers Flow Reversal Steering. If absent, fall back to standard sampling.
        reference = inputs.pop("reference", None)

        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        frs_diagnostics = None

        if reference is not None:
            if self._sample_actions_frs is None:
                raise ValueError("Reference payload provided but the model does not support FRS sampling.")
            # FRS mode controls which observation conditions the reverse (noising)
            # pass. "paper" conditions both passes on the current observation
            # (matches the paper's single-observation FRS); "cross_scene" conditions
            # the reverse pass on the reference scene's observation. Pop before the
            # transforms so the flag does not reach them.
            frs_mode = reference.pop("mode", "paper")
            # Transform the reference payload through the same input pipeline so that
            # the reference observation and action live in the model's normalized space.
            ref_inputs = self._input_transform(jax.tree.map(lambda x: x, reference))
            ref_inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], ref_inputs)
            reference_action = ref_inputs["actions"]

            current_obs = _model.Observation.from_dict(inputs)
            if frs_mode == "paper":
                reference_obs = current_obs
            elif frs_mode == "cross_scene":
                reference_obs = _model.Observation.from_dict(ref_inputs)
            else:
                raise ValueError(f"Unknown FRS mode: {frs_mode!r} (expected 'paper' or 'cross_scene').")

            actions, frs_diagnostics = self._sample_actions_frs(
                current_obs,
                reference_obs,
                reference_action,
                **self._sample_kwargs,
            )
        else:
            actions = self._sample_actions(
                sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs
            )

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        # Unbatch and convert to np.ndarray.        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        if frs_diagnostics is not None:
            outputs["frs_diagnostics"] = _unbatch_frs_diagnostics(frs_diagnostics)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def infer_batch(self, obs: dict, num_samples: int) -> list[dict]:
        """Run batched inference producing ``num_samples`` diverse action samples
        from a single observation in one forward pass."""
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: jnp.tile(jnp.asarray(x)[np.newaxis, ...], [num_samples] + [1] * np.ndim(x)),
            inputs,
        )

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time

        results = []
        for i in range(num_samples):
            single = jax.tree.map(lambda x: np.asarray(x[i, ...]), outputs)
            single = self._output_transform(single)
            single["policy_timing"] = {"infer_ms": model_time * 1000}
            results.append(single)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
