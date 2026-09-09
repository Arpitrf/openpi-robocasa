# LoRA finetuning on HITL demos, eval on a fixed scene

Converts recorded HITL `demo_0.hdf5` files (from the "semantic_corrections" project) into a
LeRobot dataset with SIRIUS-style intervention labeling, LoRA-finetunes pi0 on it, and evaluates
by replaying the exact recorded scene rather than a randomized layout.

## 0. Environment setup (once per machine)

`uv sync --group robocasa` installs everything, including `robocasa`/`robosuite` -- these are a
separate optional group (not the default `uv sync`) because they hard-pin `numpy==2.2.5`, which
conflicts with the unrelated `rlds` group's `tensorflow-cpu` (`numpy<2.0.0`); the two can't be
installed together (see `pyproject.toml`'s `[tool.uv] conflicts`). `config.py` imports RoboCasa
unconditionally at module level, so this group is required to use this repo's training entrypoint
at all, not just for RoboCasa-specific configs.

One thing `uv sync` can't fix on its own: `lerobot`'s `av` dependency needs ffmpeg 7 to build,
which isn't in Ubuntu 22.04's apt repos, and there's no sudo on this shared machine. Get it via a
self-contained conda-forge env (no root needed):

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C ~/.local/bin --strip-components=1 bin/micromamba
MAMBA_ROOT_PREFIX=~/.local/share/mamba ~/.local/bin/micromamba create -y -n ffmpeg7 -c conda-forge "ffmpeg=7" pkg-config
FFENV=~/.local/share/mamba/envs/ffmpeg7
PKG_CONFIG_PATH="$FFENV/lib/pkgconfig" LD_LIBRARY_PATH="$FFENV/lib" CPATH="$FFENV/include" uv sync --group robocasa
```

`av` also needs that same `ffmpeg7` `lib/` directory at **import** time, not just build time --
`.venv/bin/activate` exports it (deliberately *not* a global shell rc file: that env also carries
openvino/harfbuzz/pango/sdl2/libva/etc, some of which shadow system libraries sshd/PAM rely on).
`source .venv/bin/activate` before running anything below; for `uv run` (which doesn't source
`activate`), prefix commands with `LD_LIBRARY_PATH="$HOME/.local/share/mamba/envs/ffmpeg7/lib"` instead.

Finally, download the kitchen assets (~20GB, one-time):
```bash
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets --type all
```

This machine's 8 GPUs are shared with other users -- check `nvidia-smi` and pass
`CUDA_VISIBLE_DEVICES=<idx>` with a device that has free memory to every command below rather than
letting jax default to sharding across all 8 (which fails with NCCL errors under contention).

## 1. Convert HDF5s -> a LeRobot dataset

```bash
python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug_all5 \
    --raw_dataset_path \
      /mnt/hdd3/arpit/semantic_corrections/expdata/pi0_hitl/CoffeeSetupMug/monitored/2026-06-30-22-00/demo_0.hdf5 \
      /mnt/hdd3/arpit/semantic_corrections/expdata/pi0_hitl/CoffeeSetupMug/monitored/2026-08-19-12-55/demo_0.hdf5 \
      /mnt/hdd3/arpit/semantic_corrections/expdata/pi0_hitl/CoffeeSetupMug/monitored/2026-08-21-20-26/demo_0.hdf5 \
      /mnt/hdd3/arpit/semantic_corrections/expdata/pi0_hitl/CoffeeSetupMug/monitored/2026-08-29-23-13/demo_0.hdf5 \
      /mnt/hdd3/arpit/semantic_corrections/expdata/pi0_hitl/CoffeeSetupMug/monitored/2026-09-01-11-20-03-correction-2/demo_1.hdf5 \
    --demo_name demo_0 demo_0 demo_0 demo_0 demo_1
```

One episode per `--raw_dataset_path`, written to `~/.cache/huggingface/lerobot/<repo_name>`
(override with `--lerobot_home`). Verified end-to-end: 1761 total frames across the 5 demos.

### SIRIUS-style intervention reweighting

Each hdf5's per-timestep `acting_agent` field (`"human"`/`"robot"`/`"steered"`) marks when a human
took over from the autonomous policy. Adapted from SIRIUS
([ut-austin-rpl.github.io/sirius](https://ut-austin-rpl.github.io/sirius/), arXiv:2211.08416):
the conversion script drops the `--preintv_window` frames (default 15, carried over verbatim from
SIRIUS's own hyperparameter -- unverified for this domain) immediately preceding each
robot->intervention handoff (the frames judged bad enough to trigger a correction, excluded rather
than trained on), and labels every surviving frame `is_intervention` (`"human"`/`"steered"` -> 1,
`"robot"` -> 0). At train time, `LeRobotRobocasaHitlDataConfig.intervention_p_target` (default
`0.5`) builds a `WeightedRandomSampler` (`data_loader._intervention_sampler`) so each batch is, in
expectation, a 50/50 mix of intervention/robot frames regardless of the natural ratio (measured:
34% intervention / 66% robot, pooled across all 5 demos). This repo has no separate
human-demonstration class, so unlike SIRIUS's own 4-class scheme (`demo`, `robot`, `intv`,
`preintv`) this is a 2-class adaptation -- the exact `P(demo)=0` limit of SIRIUS's own formula, not
an approximation of it. `ℓ=15` and `p_target=0.5` are both unablated for this dataset. Set
`intervention_p_target=None` on the data config to disable and fall back to plain shuffling.

## 2. Download the RoboCasa-pretrained starting checkpoint

```bash
python scripts/download_checkpoint.py
```

Downloads `pi0_robocasa_pretrain_human300` (a pi0_base checkpoint already pretrained on the full
RoboCasa `pretrain_human300` soup) to
`~/.cache/openpi/robocasa/robocasa365_checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/`
-- `_ROBOCASA_PRETRAIN_HUMAN300_PARAMS` in `config.py` points at it. LoRA-finetuning from here
instead of generic `pi0_base` means the model only has to learn the task-specific correction from
a handful of demos, not RoboCasa's action space/cameras/embodiment from scratch too. One-time per
machine.

## 3. Compute norm stats

`pi0_robocasa_coffeesetupmug_hitl_lora`'s `repo_id`-based data config has no automatic norm-stats
fallback, unlike the `data_dirs`-based configs used elsewhere in this repo for the large RoboCasa
soups -- compute them once per dataset:

```bash
CUDA_VISIBLE_DEVICES=<idx> python scripts/compute_norm_stats.py --config-name=pi0_robocasa_coffeesetupmug_hitl_lora
```

## 4. Train

```bash
CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_ENTITY=robin-lab \
    python scripts/train.py pi0_robocasa_coffeesetupmug_hitl_lora --exp-name=<exp_name> --overwrite
```

`num_train_steps=10_000`, `save_interval=5_000` -- checkpoints land at step 5000 and step 9999
(the training loop is `range(0, num_train_steps)`, so the last iteration is index 9999, not
10000). `keep_period=5_000` matters here: `checkpoints.py` hardcodes `max_to_keep=1` globally
(every TrainConfig), which deletes all but the most recent checkpoint unless a step's number is
divisible by `keep_period` -- without this, the step-5000 checkpoint would get silently deleted
once step 9999 saves. Use `--resume` instead of `--overwrite` to continue an existing run.

## 5. Eval on a fixed scene

Serve the checkpoint, then in another terminal:

```bash
python scripts/serve_policy.py --port=8000 policy:checkpoint \
    --policy.config=pi0_robocasa_coffeesetupmug_hitl_lora \
    --policy.dir=/mnt/hdd1/sa53925/openpi-robocasa/checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora/<exp_name>/<step>

python examples/robocasa/eval_fixed_scene.py --args.port 8000 \
    --args.log_dir /mnt/hdd1/sa53925/openpi-robocasa/checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora/<exp_name> \
    --args.trials-per-scene 5 \
    --args.hdf5-paths <path-to-demo_0.hdf5-per-scene, from step 1's --raw_dataset_path>
```

Loads each hdf5's first recorded simulator state (`sim_state.restore_sim_state_from_hdf5`) instead
of randomizing the layout, so eval tests the scene(s) the policy actually trained on. Pass
`--args.frame` to start from a different point in a demo (e.g. post-grasp) instead of frame 0.
Results land in `.../evals_fixed_scene/CoffeeSetupMug/<scene>/` (rollout mp4s + `stats.json`).

**Caveat**: `info["success"]` measures the same task success-check used everywhere else in
robocasa. In the source HITL hdf5s, the per-timestep `rewards`/`dones` never fire even on demos
whose episode-level `success` attr is `True` -- a 0% success rate from this script does not
necessarily mean the policy failed to imitate its training data, only that it failed to complete
the task, which the training data may not have done either by this specific check.
