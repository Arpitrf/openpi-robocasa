# LoRA finetuning on HITL demo_0.hdf5 files, eval on CoffeeSetupMug

This pipeline converts one or more `demo_0.hdf5` files from the "semantic_corrections" HITL
dataset into a LeRobot dataset, LoRA-finetunes pi0 on it, and evaluates by replaying the exact
recorded scene(s) rather than a randomized layout.

The conversion, state-restore, and observation-extraction code is copied from
[`Arpitrf/semantic_corrections`](https://github.com/Arpitrf/semantic_corrections) (`frs` branch:
`semantic_corrections/datasets/lerobot_export.py`, `semantic_corrections/utils/sim_state.py`,
`semantic_corrections/hitl/{obs,env}.py`) rather than reimplemented, so this repo's training data
matches how that project's own tooling (`run_pi0_hitl.py`'s `--load-state-hdf5`) already reads and
replays these files. Files copied here (`examples/robocasa/{lerobot_export,sim_state,hitl_obs,hitl_env}.py`)
carry a small number of targeted, documented fixes for version/environment differences and a couple
of genuine (pre-existing, never-exercised) bugs found by actually running them -- see each file's
docstring for specifics.

## 1. Convert HDF5(s) -> a LeRobot dataset

```bash
python examples/robocasa/convert_hitl_hdf5_to_lerobot.py --repo_name hitl_coffeesetupmug_all3 \
    --raw_dataset_path data/CoffeeSetupMug/2026-06-30-22-00/demo_0.hdf5 \
                       data/CoffeeSetupMug/2026-08-19-12-55/demo_0.hdf5 \
                       data/CoffeeSetupMug/2026-08-21-20-26/demo_0.hdf5
```

One or more `--raw_dataset_path` values become one episode each in a single LeRobot dataset,
written to `~/.cache/huggingface/lerobot/<repo_name>` (override with `--lerobot_home`). No
simulator/env replay is used -- see the script's docstring for why -- and no reordering is applied
to state/actions, since the hdf5's raw fields are already in the order pi0 expects (also explained
in the docstring).

## 2. Download the RoboCasa-pretrained starting checkpoint

These configs LoRA-finetune `pi0_robocasa_pretrain_human300` (a pi0_base checkpoint already
pretrained on the full RoboCasa `pretrain_human300` soup), not generic `pi0_base` -- otherwise the
model has to learn RoboCasa's action space/cameras/embodiment from scratch on top of the
task-specific correction, from only 1-3 demos.

```bash
python scripts/download_checkpoint.py
```

Downloads to `~/.cache/openpi/robocasa/robocasa365_checkpoints/pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/`,
which `_ROBOCASA_PRETRAIN_HUMAN300_PARAMS` in `src/openpi/training/config.py` points at. Only
needs to be done once per machine.

## 3. Point a TrainConfig at the converted data

`pi0_robocasa_coffeesetupmug_hitl_lora` in `src/openpi/training/config.py` points at
`repo_id="hitl_coffeesetupmug_all3"`. To train on a different dataset, copy that `TrainConfig` and
change `repo_id` to whatever you passed as `--repo_name` above.

Unlike `LeRobotRobocasaDataConfig`'s `data_dirs` path (used elsewhere in this repo for the large
pretrain/target RoboCasa soups), this `repo_id`-based config has no automatic norm-stats fallback,
so compute them once per dataset before training:

```bash
python scripts/compute_norm_stats.py --config-name=pi0_robocasa_coffeesetupmug_hitl_lora
```

## 4. Train

```bash
conda activate robocasa
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_robocasa_coffeesetupmug_hitl_lora \
    --exp-name=<exp_name> --overwrite
```

Use `--resume` instead of `--overwrite` to continue an existing run (e.g. after a crash).

## 5. Eval on CoffeeSetupMug

Serve the checkpoint:

```bash
python scripts/serve_policy.py --port=8000 policy:checkpoint \
    --policy.config=pi0_robocasa_coffeesetupmug_hitl_lora \
    --policy.dir=checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora/<exp_name>/<step>
```

Then, in another terminal:

**Recommended -- replay the exact training scene(s)** (`eval_fixed_scene.py`): loads each
training hdf5's first simulator state via `restore_sim_state_from_hdf5` (from `sim_state.py`)
instead of randomizing the layout, so eval actually tests the scene(s) the policy trained on. Pass
`--args.frame` to start from a different point in the demo (e.g. a post-grasp frame) instead of
frame 0.

```bash
python examples/robocasa/eval_fixed_scene.py --args.port 8000 \
    --args.log_dir checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora/<exp_name> \
    --args.trials-per-scene 5 \
    --args.hdf5-paths data/CoffeeSetupMug/<run1>/demo_0.hdf5 data/CoffeeSetupMug/<run2>/demo_0.hdf5
```

Results land in `checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora/<exp_name>/evals_fixed_scene/CoffeeSetupMug/<run>/`
(rollout mp4s + `stats.json` per scene).

**Randomized-layout eval** (`main.py`): tests generalization to unseen layouts. Needs a
`TASK_SET_REGISTRY["CoffeeSetupMug"]` entry added to `robocasa/utils/dataset_registry.py`
(lives outside this repo, in the `robocasa` package) since only multi-task groups exist by
default. Superseded by the fixed-scene eval above for this dataset -- included for completeness.

```bash
python examples/robocasa/main.py --args.port 8000 --args.task_set CoffeeSetupMug \
    --args.split target --args.log_dir checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora/<exp_name> --args.num_trials 25
python examples/robocasa/get_eval_stats.py --dir checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora/<exp_name>
```

## Result so far

**Controlled result (current, most informative): 0/15.** LoRA-finetuning from
`pi0_robocasa_pretrain_human300` (RoboCasa-competent starting point, see step 2) -- 3 separate
single-demo runs (`_run1`/`_run2`/`_run3`, `batch_size=4`), each evaluated with
`--args.trials-per-scene 5` against only its own training scene:

| run | scene | success |
|---|---|---|
| run1 | `2026-06-30-22-00` | 0/5 |
| run2 | `2026-08-19-12-55` | 0/5 |
| run3 | `2026-08-21-20-26` | 0/5 |

This isolates the question that matters: starting from a model that already knows RoboCasa, does
LoRA-finetuning on a single HITL correction demo teach it to reproduce that corrected behavior?
Answer so far: no, for all 3 demos independently, including the one (`2026-08-21-20-26`) whose
source recording has `acting_agent="steered"` frames and `success=True`.

**Earlier, confounded results (superseded, kept for history): 0/25 and 0/15.** From before this
pipeline started from the RoboCasa-pretrained checkpoint -- a 0/25 randomized-layout eval and a
0/15 fixed-scene eval (pooled 3-demos-in-one-run), both LoRA-finetuned from generic `pi0_base`
instead. Those results conflated "not enough data to learn RoboCasa's action space/cameras/
embodiment from scratch" with "the correction data itself doesn't teach the corrected behavior" --
the controlled result above resolves that ambiguity. (The 0/25 and the pooled 0/15 are from an
earlier version of this pipeline that converted to a Groot-LeRobot format instead of the plain
format above -- `LeRobotRobocasaDataConfig`/Groot modality.json, since replaced; checkpoints and
eval videos from that run are still under `checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora*/`
for reference. Verified numerically equivalent to what this pipeline produces, i.e. not an
artifact of the format change.)

## Caveats

- These "semantic_corrections" HITL hdf5s record a *monitored* rollout annotated with what went
  wrong (collisions, missed placements, etc.). The per-timestep `rewards`/`dones` arrays are 0/False
  throughout in all 3 files used so far -- but the episode-level `success` attr is `True` on 2 of the
  3 (`2026-08-19-12-55`, `2026-08-21-20-26`; the third, `2026-06-30-22-00`, has no `success` attr at
  all), and `acting_agent` includes a `"steered"` value (not just `"human"`/`"robot"`) in
  `2026-08-21-20-26` -- so at least 2 of 3 source demos *did* reach task success via a human
  correction or FRS-style steering, even though the reward/done signal never fires and plain
  behavior cloning on the raw trajectory doesn't reproduce that corrective behavior. How
  `correction_semantic_annotations` and `acting_agent` should factor into training (e.g. weighting
  toward the corrected/steered segments) is unresolved -- confirm with whoever owns that pipeline
  before drawing conclusions from eval results.
- Rendering needs a working NVIDIA EGL setup (a vendor file registering `libEGL_nvidia.so.0`
  under `/usr/share/glvnd/egl_vendor.d/`); the OSMesa software fallback segfaulted in this
  environment.
- Keep `keep_period` in `TrainConfig` well above `num_train_steps`. Setting it equal to
  `save_interval` makes the checkpoint manager retain every checkpoint instead of rotating,
  which can fill disk mid-run.
- Some state dimensions (mobile base position/rotation) have ~zero variance across these short
  demos, which can produce large normalized outliers (division by a near-zero std, epsilon-guarded
  in `transforms.Normalize` but not eliminated). Present in both the old and current pipeline
  equally -- not something introduced by either.
