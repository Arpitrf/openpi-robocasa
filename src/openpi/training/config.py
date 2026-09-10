"""See _CONFIGS for the list of available configs."""

import abc
import json
import os
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import numpy as np

import openpi.groot_utils.groot_openpi_dataset as _groot_openpi_dataset
from robocasa.macros import DATASET_BASE_PATH
from robocasa.utils.dataset_registry import DATASET_SOUP_REGISTRY

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    
    # Action dimension for padding (used by Groot datasets)
    action_dim: int | None = None
    
    # Multi-dataset support for Groot datasets
    data_dirs: list[str] | None = None  # List of data directories for multi-dataset
    dataset_weights: list[float] | None = None  # Weights for each dataset in multi-dataset

    # SIRIUS-style (https://ut-austin-rpl.github.io/sirius/) intervention/robot resampling
    # target for HITL datasets with an is_intervention column (see LeRobotRobocasaHitlDataConfig).
    # If None (default), no resampling is applied and training uses standard shuffled sampling.
    intervention_p_target: float | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        base = self.base_config or DataConfig()
        # Preserve pre-supplied norm_stats; only load if not provided
        existing_stats = base.norm_stats
        loaded_stats = None if existing_stats is not None else self._load_norm_stats(
            epath.Path(self.assets.assets_dir or assets_dirs), asset_id
        )
        return dataclasses.replace(
            base,
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=existing_stats if existing_stats is not None else loaded_stats,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            # NOTE: this used to fall back to _groot_openpi_dataset._convert_stats_from_repo_meta,
            # which doesn't exist (confirmed -- AttributeError, not just unimplemented) and was
            # already marked "# TODO: fix" here in the base commit before any HITL LoRA work
            # started. Since it could never have succeeded, removing the call is a no-op for any
            # config that actually needs a working fallback; it just stops this from crashing.
            # For non-Groot configs like LeRobotRobocasaHitlDataConfig there is no fallback at
            # all -- run `scripts/compute_norm_stats.py` first, same as before.
            logging.info(f"Norm stats not found in {data_assets_dir}.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    """Config for training on Groot datasets."""
    
    repo_id: str | None = None
    
    data_dirs: Any | None = None
    dataset_weights: list[float] | None = None
    
    action_dim: int | None = None
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group()

        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        base = self.create_base_config(assets_dirs)

        # Fallback: if norm_stats not found via assets/repo meta, combine from all data_dirs
        fallback_norm_stats = None
        if base.norm_stats is None and self.data_dirs and len(self.data_dirs) > 0:
            if len(self.data_dirs) == 1:
                d = self.data_dirs[0]
                norm_stats = _groot_openpi_dataset._load_norm_stats_from_groot_dataset(d)
                if norm_stats is not None:
                    fallback_norm_stats = norm_stats
                    logging.info(f"Loaded norm stats from local data dir: {d}")
            else:
                norm_stats = _groot_openpi_dataset._load_norm_stats_from_groot_mixture_dataset(self.data_dirs)
                if norm_stats is not None:
                    fallback_norm_stats = norm_stats
                    logging.info(f"Loaded combined norm stats from {len(self.data_dirs)} data dirs")
 
        return dataclasses.replace(
            base,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_dim=model_config.action_dim,
            data_dirs=self.data_dirs,
            dataset_weights=self.dataset_weights,
            norm_stats=base.norm_stats or fallback_norm_stats,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaHitlDataConfig(DataConfigFactory):
    """Plain-LeRobot (not Groot) config for datasets written by
    examples/robocasa/convert_hitl_hdf5_to_lerobot.py, which uses the image/wrist_image/state/
    actions schema from Arpitrf/semantic_corrections's lerobot_export.py rather than
    LeRobotRobocasaDataConfig's Groot format. Mirrors LeRobotLiberoDataConfig's repack pattern;
    no delta-action transform, since RoboCasa's 12-dim action space (EE pos/rot deltas + gripper +
    base motion + control mode) isn't joint angles and LeRobotRobocasaDataConfig doesn't apply one
    either.
    """

    # SIRIUS-style (https://ut-austin-rpl.github.io/sirius/) target fraction of intervention
    # frames per training batch, via a WeightedRandomSampler over the dataset's is_intervention
    # column (see data_loader.create_torch_data_loader). 0.5 is the natural 2-class analog of
    # SIRIUS's own default -- unablated for this dataset, see README_HITL_LORA.md.
    intervention_p_target: float | None = 0.5

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[robocasa_policy.RobocasaInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[robocasa_policy.RobocasaOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            intervention_p_target=self.intervention_p_target,
        )


@dataclasses.dataclass
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # --- In-training simulator eval (see openpi/training/robocasa_eval.py) ---
    # Roll the live parameters out in RoboCasa every `eval_interval` steps and log the videos to
    # this run's wandb page. None disables it. Rollouts are not free -- each is a full simulated
    # episode -- so a small interval can cost more wall time than the training itself.
    eval_interval: int | None = None
    # Rollouts per eval. They all start from the same recorded state, so they differ only by the
    # policy's sampling noise.
    eval_rollouts: int = 2
    # Recorded sim state every eval episode is restored from (an hdf5 written by
    # semantic_corrections' run_pi0_hitl.py, or an init_states/*_raw.hdf5).
    eval_init_state: str | None = None
    eval_env_name: str = "CoffeeSetupMug"
    # Weld the task object to the EEF on grasp during eval rollouts, matching what collection did
    # (semantic_corrections' task.weld_on_grasp). Must track the collection setting: an eval
    # without the weld grades the policy on different contact dynamics than it was trained on.
    eval_weld_on_grasp: bool = False
    # None -> robocasa's task horizon * 1.5, the same budget collection used.
    eval_horizon: int | None = None
    eval_replan_steps: int = 5

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Path to the pi0_robocasa_pretrain_human300 checkpoint downloaded by scripts/download_checkpoint.py
# (a pi0_base checkpoint already pretrained on the full RoboCasa pretrain_human300 soup). Used
# below as the LoRA finetuning start point for the CoffeeSetupMug HITL config, instead of generic
# pi0_base, so it only has to learn the task-specific correction from a handful of demos, not
# RoboCasa's action space/cameras/embodiment from scratch too.
_ROBOCASA_PRETRAIN_HUMAN300_PARAMS = os.path.expanduser(
    "~/.cache/openpi/robocasa/robocasa365_checkpoints/pi0/pi0_robocasa_pretrain_human300/"
    "multitask_learning/75000/params"
)

# Repo-local storage for the HG-DAGGER round configs. The older
# pi0_robocasa_coffeesetupmug_hitl_lora config hardcodes /mnt/hdd1/sa53925/... , which only
# existed on the previous (8-GPU) machine; deriving from the source tree keeps rounds working
# wherever the repo is checked out. Both dirs are gitignored.
_REPO_ROOT = pathlib.Path(__file__).parents[3]


def hgdagger_lora_configs(
    env_name: str,
    *,
    rounds: int = 8,
    num_train_steps: int = 5_000,
    save_interval: int = 500,
    batch_size: int = 8,
    freeze_vision_tower: bool = False,
    eval_interval: int | None = 100,
    eval_rollouts: int = 2,
    init_state: str = "init_states/CoffeeMugSetup/l0/demo_0_raw.hdf5",
    warm_start: bool = True,
) -> list["TrainConfig"]:
    """One LoRA TrainConfig per HG-DAGGER round for `env_name`.

    Round r trains on the *aggregated* demos from rounds 1..r (built by
    examples/robocasa/convert_hitl_hdf5_to_lerobot.py --round_dirs into repo_id
    hgdagger_<env>_r{r}), always restarting from the RoboCasa pretrain checkpoint rather than
    stacking LoRA on the previous round's LoRA -- DAgger re-fits on the grown dataset, and this
    keeps round r from inheriting round r-1's drift.

    Everything not named here is openpi's LoRA default: rank 16/alpha 16 on the VLM, rank
    32/alpha 32 on the action expert (attn + ffn), AdamW(b1=0.9, b2=0.95, wd=1e-10, clip=1.0),
    peak LR 2.5e-5, EMA off.

    `warm_start` (default) has round r>1 resume from round r-1's final checkpoint -- adapters and
    all -- instead of re-LoRA-ing the pretrain checkpoint from scratch. Since each round also
    trains on the *aggregated* data from rounds 1..r, round 1's demos are effectively revisited
    each round, and round r inherits whatever round r-1 converged to (textbook DAgger instead
    re-fits from the same start every round, which cannot inherit drift). Set warm_start=False
    for that behaviour. Round 1 always starts from the RoboCasa pretrain checkpoint.

    `freeze_vision_tower` emits the `_frozenvit` variant of each round instead. openpi's stock
    LoRA freeze filter covers `.*llm.*` only, so by default the PaliGemma vision tower is *fully*
    finetuned alongside the LoRA adapters -- 1.74 GiB of trainable fp32 params (plus AdamW
    moments) against a few thousand frames. The variant additionally freezes `.*img.*`, leaving
    0.20 GiB trainable: the LLM's LoRA adapters and the state/action projections. Both variants
    read the same per-round dataset, so a round can be trained each way and compared.
    """
    model = pi0.Pi0Config(
        max_token_len=96,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    freeze_filter = model.get_freeze_filter()
    if freeze_vision_tower:
        freeze_filter = nnx.Any(freeze_filter, nnx_utils.PathRegex(".*img.*"))
    suffix = "_frozenvit" if freeze_vision_tower else ""
    exp_suffix = "-frozenvit" if freeze_vision_tower else ""
    slug = env_name.lower()

    def _weights_for(r: int) -> weight_loaders.WeightLoader:
        if r == 1 or not warm_start:
            return weight_loaders.CheckpointWeightLoader(_ROBOCASA_PRETRAIN_HUMAN300_PARAMS)
        # The final step of round r-1's run: the loop is range(0, num_train_steps), so the last
        # index is num_train_steps - 1.
        prev = (
            _REPO_ROOT
            / "checkpoints"
            / f"pi0_robocasa_{slug}_hgdagger_r{r - 1}{suffix}"
            / f"hgdagger-rounds-{slug}-r{r - 1}{exp_suffix}"
            / str(num_train_steps - 1)
            / "params"
        )
        return weight_loaders.CheckpointWeightLoader(str(prev))

    return [
        TrainConfig(
            name=f"pi0_robocasa_{slug}_hgdagger_r{r}{suffix}",
            model=model,
            data=LeRobotRobocasaHitlDataConfig(
                repo_id=f"hgdagger_{slug}_r{r}",
                base_config=DataConfig(prompt_from_task=True),
            ),
            weight_loader=_weights_for(r),
            freeze_filter=freeze_filter,
            ema_decay=None,  # off for LoRA, as in the other *_low_mem_finetune configs
            # The default CosineDecaySchedule warms up for 1_000 steps and decays over 30_000 --
            # at 5_000 steps that would spend a fifth of the run warming up and never finish
            # decaying (LR would end near peak). Both are scaled to the actual run length here.
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=200,
                peak_lr=2.5e-5,
                decay_steps=num_train_steps,
                decay_lr=2.5e-6,
            ),
            num_train_steps=num_train_steps,
            # checkpoints.py hardcodes max_to_keep=1 globally, so every saved step except the most
            # recent is deleted unless step % keep_period == 0. keep_period == save_interval keeps
            # all of them (1000/2000/3000/4000 by divisibility, 4999 as the most recent -- the
            # loop is range(0, num_train_steps), so the last index is 4999, not 5000).
            save_interval=save_interval,
            keep_period=save_interval,
            # 32 measured to fit on a single 24GB RTX 4090 at XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
            # (so do 8 and 16); the only configuration that OOMs is disabling XLA preallocation.
            batch_size=batch_size,
            num_workers=2,
            # Simulator rollouts from the round's fixed init state, logged as wandb videos on the
            # training run's own step axis.
            eval_interval=eval_interval,
            eval_rollouts=eval_rollouts,
            eval_init_state=str(_REPO_ROOT / init_state),
            eval_env_name=env_name,
            # configs/hitl/hgdagger_coffee.yaml collects with weld_on_grasp: true.
            eval_weld_on_grasp=True,
            # wandb entity isn't a TrainConfig field (train.py never passes one) -- launch with
            # WANDB_ENTITY=robin-lab. Run name comes from --exp-name (hgdagger-rounds-*).
            project_name="semantic-corrections",
            assets_base_dir=str(_REPO_ROOT / "assets"),
            checkpoint_base_dir=str(_REPO_ROOT / "checkpoints"),
        )
        for r in range(1, rounds + 1)
    ]


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        name="pi0_fast_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    #
    # RoboCasa dataset configs.
    #
    TrainConfig(
        name="pi0_robocasa_target50",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target50"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=500000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_finetune_target_atomic_seen",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("INSERT_CKPTPOINT_HERE"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_finetune_target_composite_seen",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_seen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("INSERT_CKPTPOINT_HERE"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_finetune_target_composite_unseen",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_unseen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("INSERT_CKPTPOINT_HERE"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=5000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_atomic_seen",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_atomic_seen_random_weight_init",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
	    weight_loader=weight_loaders.NoOpWeightLoader(),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_atomic_seen_paligemma_init",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_atomic_seen"],
        ),
	    weight_loader=weight_loaders.PaliGemmaWeightLoader(),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_composite_seen",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_seen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_target_composite_unseen",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["target_composite_unseen"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_pretrain_human300_mg60",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human300_mg60"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        name="pi0_robocasa_pretrain_human300",
        model=pi0.Pi0Config(
            max_token_len=96,
        ),
        data=LeRobotRobocasaDataConfig(
            data_dirs=DATASET_SOUP_REGISTRY["pretrain_human300"],
        ),
	    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=100000,
            decay_lr=2.5e-6,
        ),
        num_train_steps=100000,
        save_interval=5000,
        keep_period=10000,
        batch_size=64,
        num_workers=4,
    ),
    TrainConfig(
        # LoRA finetune of pi0_robocasa_pretrain_human300 on 5 pooled HITL CoffeeSetupMug demos.
        # See README_HITL_LORA.md for the conversion command and norm-stats step.
        name="pi0_robocasa_coffeesetupmug_hitl_lora",
        model=pi0.Pi0Config(
            max_token_len=96,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotRobocasaHitlDataConfig(
            repo_id="hitl_coffeesetupmug_all5",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_ROBOCASA_PRETRAIN_HUMAN300_PARAMS),
        freeze_filter=pi0.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        num_train_steps=10_000,
        save_interval=2_500,
        # checkpoints.py hardcodes max_to_keep=1 (shared across every TrainConfig), which deletes
        # all but the most recent checkpoint unless a step's number is divisible by keep_period --
        # with save_interval=2_500/num_train_steps=10_000, saves land at steps 2500, 5000, 7500,
        # and 9999 (the loop is range(0, num_train_steps), so the last iteration is index 9999,
        # not 10000). keep_period=2_500 protects 2500/5000/7500 from that rotation (all divisible
        # by it); step 9999 survives anyway as the most recent one. Without this matching
        # save_interval, only the keep_period-divisible steps and the final one would survive --
        # the rest would get silently deleted as later checkpoints save.
        keep_period=2_500,
        batch_size=8,
        num_workers=2,
        # wandb.init()'s entity isn't a TrainConfig field (train.py doesn't pass one) -- set
        # WANDB_ENTITY=robin-lab in the environment when launching to log there.
        project_name="semantic-corrections",
        assets_base_dir="/mnt/hdd1/sa53925/openpi-robocasa/assets",
        checkpoint_base_dir="/mnt/hdd1/sa53925/openpi-robocasa/checkpoints",
    ),
    *hgdagger_lora_configs("CoffeeSetupMug"),
    *hgdagger_lora_configs("CoffeeSetupMug", freeze_vision_tower=True),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
