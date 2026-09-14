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
(override with `--lerobot_home`). 1685 total frames across the 5 demos, 29.1% labeled
`is_intervention`.

### SIRIUS-style intervention reweighting

Each hdf5's per-timestep `acting_agent` field (`"human"`/`"robot"`/`"steered"`) marks when a human
took over from the autonomous policy. Adapted from SIRIUS
([ut-austin-rpl.github.io/sirius](https://ut-austin-rpl.github.io/sirius/), arXiv:2211.08416):
the conversion script drops the `--preintv_window` frames (default 10, in SIRIUS's own ballpark --
their hyperparameter is 15, unverified for this domain either way) immediately preceding each
robot->takeover transition (the frames judged bad enough to trigger a takeover, excluded rather
than trained on). At train time, `LeRobotRobocasaHitlDataConfig.intervention_p_target` (default
`0.5`) builds a `WeightedRandomSampler` (`data_loader._intervention_sampler`) so each batch is, in
expectation, a 50/50 mix of intervention/robot frames regardless of the natural ratio. This repo
has no separate human-demonstration class, so unlike SIRIUS's own 4-class scheme (`demo`, `robot`,
`intv`, `preintv`) this is a 2-class adaptation -- the exact `P(demo)=0` limit of SIRIUS's own
formula, not an approximation of it. Set `intervention_p_target=None` on the data config to
disable and fall back to plain shuffling.

**"steered" frames -- two dataset variants for an A/B comparison.** Besides `"human"`/`"robot"`,
`acting_agent` can be `"steered"` (FRS-style assisted control, distinct from raw human teleop).
By default (`hitl_coffeesetupmug_all5` above) these are dropped entirely and `is_intervention` is
`True` only for `"human"` -- SIRIUS's own scheme has no analog of "steered". Passing
`--include_steered` instead keeps them and folds them into `is_intervention` alongside `"human"`
(the original, pre-ablation behavior), producing a second dataset:

```bash
python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug_all5_steered \
    --include_steered \
    --raw_dataset_path <same 5 paths as above> \
    --demo_name demo_0 demo_0 demo_0 demo_0 demo_1
```

1794 total frames, 33.4% labeled `is_intervention` -- 109 more frames than the no-steered variant
(1685 above), the raw count of "steered"-labeled frames across the 5 demos that are now kept
instead of dropped. `pi0_robocasa_coffeesetupmug_hitl_lora_steered` in `config.py` points at this
repo_id; it's otherwise identical to `pi0_robocasa_coffeesetupmug_hitl_lora` (same LR schedule,
step count, `preintv_window=10` on both) so a training run on each isolates the effect of
including "steered" frames, holding everything else fixed.

**"robot_only" -- a third variant with corrections removed entirely.** To measure how much the
human/steered correction frames are helping at all (as opposed to just having more autonomous
rollout to imitate), `--robot_only` keeps only `acting_agent == "robot"` frames and drops every
"human"/"steered" frame outright -- no `preintv_window` trimming (nothing to protect a transition
into, since no intervention frames survive) and `is_intervention` all-zero:

```bash
python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug_all5_robotonly \
    --robot_only \
    --raw_dataset_path <same 5 paths as above> \
    --demo_name demo_0 demo_0 demo_0 demo_0 demo_1
```

1265 total frames (vs. 1685/1794 above). `pi0_robocasa_coffeesetupmug_hitl_lora_robotonly` in
`config.py` points at this repo_id, with `intervention_p_target=None` (data_loader's
`WeightedRandomSampler` requires both classes present and raises otherwise) -- plain shuffling
instead of the 50/50 SIRIUS resampling the other two configs use. Same LR schedule/step count as
the other two, so comparing eval results across all three isolates the corrections' contribution.

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

Neither config's `repo_id`-based data config has an automatic norm-stats fallback, unlike the
`data_dirs`-based configs used elsewhere in this repo for the large RoboCasa soups -- compute them
once per dataset (each `config-name` below reads its own `repo_id`):

```bash
CUDA_VISIBLE_DEVICES=<idx> python scripts/compute_norm_stats.py --config-name=pi0_robocasa_coffeesetupmug_hitl_lora
CUDA_VISIBLE_DEVICES=<idx> python scripts/compute_norm_stats.py --config-name=pi0_robocasa_coffeesetupmug_hitl_lora_steered
CUDA_VISIBLE_DEVICES=<idx> python scripts/compute_norm_stats.py --config-name=pi0_robocasa_coffeesetupmug_hitl_lora_robotonly
```

## 4. Train

`pi0_robocasa_coffeesetupmug_hitl_lora` (steered frames dropped) and
`pi0_robocasa_coffeesetupmug_hitl_lora_steered` (steered frames kept, see step 1) are otherwise
identical -- same LR schedule and step count -- so running both isolates the effect of that one
difference:

```bash
CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_ENTITY=robin-lab \
    python scripts/train.py pi0_robocasa_coffeesetupmug_hitl_lora --exp-name=<exp_name> --overwrite

CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_ENTITY=robin-lab \
    python scripts/train.py pi0_robocasa_coffeesetupmug_hitl_lora_steered --exp-name=<exp_name> --overwrite

CUDA_VISIBLE_DEVICES=<idx> XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 WANDB_ENTITY=robin-lab \
    python scripts/train.py pi0_robocasa_coffeesetupmug_hitl_lora_robotonly --exp-name=<exp_name> --overwrite
```

`num_train_steps=20_000`, `save_interval=2_500` -- checkpoints land at 2500, 5000, ..., 17500, and
19999 (the training loop is `range(0, num_train_steps)`, so the last iteration is index 19999, not
20000, and it's always saved as the most recent checkpoint regardless of `save_interval`) -- 8
checkpoints, same cadence as the original `sirius_lora_v1` run. `keep_period=2_500` (matching
`save_interval`) matters here: `checkpoints.py` hardcodes `max_to_keep=1` globally (every
TrainConfig), which deletes all but the most recent checkpoint unless a step's number is divisible
by `keep_period` -- without this, every checkpoint but the last would get silently deleted as
training progresses. `lr_schedule` (`warmup_steps=800, decay_steps=20_000`) is rescaled from
`skand/coffeesetupmug-dagger-rounds`'s `hgdagger_lora_configs` (tuned for a 5_000-step run) to
match this run length, keeping the same warmup:decay ratio. Use `--resume` instead of `--overwrite`
to continue an existing run.

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

## 6. OLAF: relabeling the pre-intervention frames instead of dropping them

Steps 1-5 follow SIRIUS: the `--preintv_window` frames before each takeover are thrown away. OLAF
([ut-austin-rpl.github.io/olaf](https://ut-austin-rpl.github.io/olaf/), arXiv:2310.17555) does
something better with them -- it uses the human's *verbal* correction (already recorded in each
hdf5's `correction_semantic_annotations`) to pick, from a pool of candidate actions, the one that
best carries it out, and overwrites the recorded action. The policy then learns the right thing at
the states where it used to go wrong, rather than learning nothing there.

Only the pre-intervention region is replaced. Everything else -- including the human's intervention
actions -- is kept verbatim, per the paper (SS II-B):

> "the recorded trajectory would be a sequence of 1) the robot's initial trajectory, where the
> mistakes have not happened; 2) the pre-intervention region, which covers the mistakes; 3) the
> user correction 4) the robot's terminal trajectory after the user corrects the robot and releases
> back the control."

```bash
export GEMINI_API_KEY=...
python examples/robocasa/olaf_relabel.py --out_dir olaf_sidecars \
    --raw_dataset_path <the 5 CoffeeSetupMug paths from step 1> \
    --demo_name demo_0 demo_0 demo_0 demo_0 demo_1
```

This writes one **sidecar JSON per demo** and never touches the source hdf5s -- they live on a
shared drive and are the inputs steps 1-5 were built from. Each sidecar records the chosen plan,
all 50 candidates it was offered, the VLM's stated reason, and the raw response, so the checks
below need no re-inference.

### How it works, and where it differs from the paper

**One VLM query per correction, selecting a chunk.** Per the paper (SS II-C), *"OLAF queries the LLM
only once per verbal correction… we set the state as o_T and then use the output of the LLM to
relabel all the actions in the pre-intervention time interval."* So this samples 50 pi0 action
chunks once at the window start, shows them to the VLM, and relabels all 10 frames with the chosen
chunk. Selecting a *chunk* rather than a single action also matches `semantic_corrections`'
steering loop (`run_correction.py`), which executes a chosen chunk whole. A chunk is a coherent
plan whose displacement accumulates; independently chosen per-frame actions partially cancel.

Because there is exactly one query, at one frame, the keypoints are used exactly as
`keypoints.json` extracted them -- no tracking, no simulator, no interpretation of the `attachment`
field. Keypoints are listed to the VLM with their ids, positions and attachment labels verbatim,
and every candidate is described by **where it takes the gripper**. That is well-defined in every
task, which is why the same code runs unmodified on CoffeeSetupMug (the gripper carries a mug
toward a nozzle) and StartElectricKettle (the gripper reaches for a fixed lever).

Two deliberate deviations from the paper:

* **Candidates are pi0 samples that replace the action**, where the paper uses fixed
  one-dimensional deltas *added on top of* the original action. A consequence of sampling from the
  policy itself, which keeps every relabeled action in-distribution as a BC target.
* **Training is LoRA from the pretrained checkpoint**, where the paper aggregates the relabeled
  data with the full pretraining set.

The sampling policy defaults to the **bootstrap** `pi0_robocasa_pretrain_human300` checkpoint --
the policy that actually produced these rollouts. Sampling from a HITL-finetuned checkpoint would
leak the corrections back into their own relabeling.

`--dry_run` samples and caches candidates without spending a single API call. Use it as a gate:
compare the spread of the 50 chunks against the human's own corrective action, and if the
correction lies outside that spread, OLAF cannot express it no matter how good the prompt is.
Candidates cache under `<out_dir>/cache/candidates/` and Gemini responses under
`<out_dir>/cache/vlm/`, both keyed by content hash, so reruns are free and show the VLM
byte-identical candidates.

### Which frames get relabeled

One window per demo: the `--window` (default 10) frames before the human takeover, located with
the same rule `semantic_corrections` uses (`human_takeover_starts` -- "control switches non-human
-> human", which covers `steered->human` as well as `robot->human`). That is exactly where the
`<demo>_assets/keypoints.json` reference frame sits, verified across all 12 demos of both tasks.

Dims 0:7 of the chosen action are used. Dims 7:11 are excluded because they carry no signal
(measured: `|v| < 0.002`, dim 10 always 0, dim 11 a constant control-mode flag). Dim 6 (the
gripper) is relabeled but **snapped by sign to ±1** -- it is a discrete latch, so writing a raw
sample like `-0.3` would be both out of distribution and indecisive. This matters: on
`2026-08-29-23-13` the gripper *is* the entire correction ("not opening its gripper to release the
mug").

### Convert, train, eval

```bash
python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug_all5_olaf \
    --raw_dataset_path <the same 5 paths> --demo_name demo_0 demo_0 demo_0 demo_0 demo_1 \
    --olaf_sidecar olaf_sidecars/2026-06-30-22-00__demo_0.olaf.json \
                   olaf_sidecars/2026-08-19-12-55__demo_0.olaf.json \
                   olaf_sidecars/2026-08-21-20-26__demo_0.olaf.json \
                   olaf_sidecars/2026-08-29-23-13__demo_0.olaf.json \
                   olaf_sidecars/2026-09-01-11-20-03-correction-2__demo_1.olaf.json
```

Expect **1735 frames / 31.1% is_intervention** -- the 1685 / 29.1% of step 1 plus exactly the 50
relabeled frames. `--olaf_sidecar` takes one value per `--raw_dataset_path` (never broadcast, since
a sidecar is demo-specific); pass `none` to skip a path. It composes with `--include_steered` and
is rejected alongside `--robot_only`.

Then norm stats and training as in steps 3-4 with `pi0_robocasa_coffeesetupmug_olaf_lora`, which is
identical to `pi0_robocasa_coffeesetupmug_hitl_lora` in every field except `repo_id` (and
`checkpoint_base_dir`, on hdd4 because hdd1 has no room for another 72GiB run).

Evaluate with `examples/robocasa/eval_hgdagger_checkpoints.py` (`--init-state-hdf5` for a fixed
scene, `--init-state-dir` for a folder of init states).

### StartElectricKettle

Identical pipeline, no task-specific code. Same 7 demos as
`pi0_robocasa_startelectrickettle_hitl_lora` (verified by reproducing that dataset's exact
per-episode frame counts), same training setup, `pi0_robocasa_startelectrickettle_olaf_lora`.
Expect **3392 frames / 21.3% is_intervention** (3322 / 19.7% plus 70 relabeled frames). Eval init
states: L0 `bootstrap/2026-09-07-17-29-24/`, L1 `bootstrap/2026-09-10-16-48-55/`.

### Checking the relabeling

```bash
python examples/robocasa/olaf_report.py olaf_sidecars/*.olaf.json \
    --goal dispenser_nozzle,nozzle,kettle_lever,lever,kettle_lid
```

reports, per window, where the chosen plan leaves the gripper relative to every keypoint, how it
ranks among the 50 candidates, and the best any candidate could have done (`oracle`). `--goal` is
required and takes ids in preference order -- there is no reliable way to infer which keypoint a
correction is about, and the held object is never it.

```bash
python examples/robocasa/olaf_replay_video.py --sidecar olaf_sidecars/<demo>.olaf.json \
    --env_name CoffeeSetupMug --out_dir videos --full
```

renders two videos from the same initial simulator state, running open-loop through the whole demo
with only the relabeled window differing. Open-loop replay tracks the recording to within
**0.4-0.5 cm over 350 frames**, so differences well above that are real rather than drift.

### Reading the result honestly

Across both tasks, **9 of 12 windows** move the gripper toward the keypoint the correction names
(mean −1.20 cm), with 4 optimal candidate picks and 7 in the top 2 of 50. Of the 3 that move away,
two are the metric mismatching the correction rather than a bad choice -- one asks the robot to
*release* the mug, another explicitly asks it to *back off* from the kettle before approaching.

Two windows (`2026-09-01-11-20-03-correction-2` and `2026-09-08-15-05-57`) are dead: **no candidate
in the pool improves on the recorded action**. That is a limit of sampling candidates from pi0
itself -- at those states the policy is confidently doing the wrong thing and all 50 samples agree
-- not a VLM failure. Expect ~10 of the relabeled frames per task to contribute nothing.

OLAF changes 50 of 1735 frames (CoffeeSetupMug) and 70 of 3392 (kettle). That is 2.9% and 2.1% by
raw count, but `intervention_p_target=0.5` gives the intervention class half of every batch, so the
relabeled frames actually carry ~4.6% and ~4.9% of the training signal -- against 0% in the
baselines, where they are dropped. Even so, at n=50 rollouts the binomial SE near p=0.25 is ~6
points, and `no_steer` L0 moved 0.22 -> 0.28 across two runs of the identical checkpoint. Read the
checkpoint sweep and the per-window numbers above, not a single headline success rate.

Baselines. CoffeeSetupMug: steered 0.62 L0 / 0.10 L1, no_steer 0.22 / 0.02, robotonly ~0.00.
StartElectricKettle: both existing arms ~0.10 L0.
