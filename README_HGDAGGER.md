# HG-DAGGER rounds: collect with interventions, LoRA-finetune, repeat

Human-gated DAgger on RoboCasa: the policy drives, you take over with the SpaceMouse when it goes
wrong, and each round's mixed-agent rollouts get folded into the training set. Round `r` trains on
the aggregated demos from rounds `1..r`, always re-LoRA-ing from the RoboCasa pretrain checkpoint
(DAgger re-fits on the grown dataset; stacking LoRA on LoRA would let round `r` inherit round
`r-1`'s drift).

Every episode of every round starts from **one fixed recorded sim state**
(`init_states/CoffeeMugSetup/l0/demo_0_raw.hdf5`, layout 43 / style 19), so rounds differ only in
what the policy and the human did — not in scene or object placement.

This supersedes the single-shot pipeline in `README_HITL_LORA.md`, which trained one LoRA on a
pile of pre-existing demos.

## Environment

Everything (collection *and* training) runs from the `semantic_corrections` conda env:

```bash
P=/home/skand/anaconda3/envs/semantic_corrections/bin/python
```

It has robocasa/robosuite/hid **and** jax 0.5.3 + lerobot 0.3.3 + this repo installed editable, so
no cross-env dance is needed. Note lerobot 0.3.x removed `lerobot.common.*`; `lerobot_export.py`
and `training/data_loader.py` probe both module paths, and `add_frame`'s task argument moved out
of the frame dict — handled in `lerobot_export._add_frame`.

## One command per round

```bash
$P scripts/hgdagger_round.py --round 1
```

Stages, in order (`--stages` to run a subset, `--dry-run` to print the commands):

| Stage | What it does |
|---|---|
| `serve` | Starts the policy this round is collected under — round 1 the pretrain checkpoint, round `r>1` round `r-1`'s latest checkpoint |
| `collect` | Runs `semantic_corrections/scripts/run_pi0_hitl.py` until 10 **intervened** demos land in `<expdata>/hgdagger/CoffeeSetupMug/round_<r>/` — the only stage needing a human |
| `convert` | Aggregates rounds `1..r` into LeRobot dataset `hgdagger_coffeesetupmug_r<r>` |
| `norm` | `compute_norm_stats.py` for this round's config |
| `train` | `train.py` (kills the policy server first — one 24 GB GPU can't hold both) |
| `eval` | Rolls out each saved checkpoint from the fixed init state, logs videos to the round's wandb run |

State lives in `<expdata>/hgdagger/CoffeeSetupMug/manifest.json`. Interrupted collection resumes
by counting existing `demo_*.hdf5` — re-running `--stages collect` tops the round back up to 10
rather than restarting it.

## Data layout

Rounds live in **this repo's** gitignored `expdata/` (not `semantic_corrections/expdata` —
the demos belong next to the training code that consumes them); override with `--expdata`.

```
openpi-robocasa/expdata/hgdagger/CoffeeSetupMug/round_1/demo_0.hdf5 … demo_9.hdf5
                                         /human_only_demo_*.hdf5, rollout_*.mp4, stats.json
                                 /round_2/…
                                 /manifest.json
```

~280 MB per 500-step demo: the round config sets `cameras.record_depth: false` and disables the
custom camera, since only agentview + wrist RGB feed training (with depth and 3 cameras the same
demo is ~1.9 GB).

## What counts as training data

Both the human's frames **and** the policy's own frames are trained on. The only frames dropped
are the `--preintv_window` (10) immediately before each robot→human handoff — the frames bad
enough to have triggered the takeover, which SIRIUS
([arXiv:2211.08416](https://arxiv.org/abs/2211.08416)) excludes rather than reproduces.

Surviving frames are labeled `is_intervention` (human 1 / robot 0), and
`LeRobotRobocasaHitlDataConfig.intervention_p_target = 0.5` builds a `WeightedRandomSampler` so
each batch is ~50/50 intervention/robot regardless of the natural ratio.

**Episode splitting.** pi0 trains on 50-frame (2.5 s) action chunks and openpi applies no
`*_is_pad` loss mask anywhere, so a chunk straddling a dropped preintv window would be trained as
if the robot teleported across the gap. Each demo is therefore split at every drop boundary and
each contiguous run becomes its own LeRobot episode — no chunk ever crosses a gap. In practice a
demo has a handful of interventions, so segments stay long (measured on a real demo: 500 frames →
2 segments of 125 and 365). `dataset_manifest.json` in the dataset dir records the full
segment-length distribution, plus per-round frame and intervention counts.

## LoRA hyperparameters

`hgdagger_lora_configs` in `src/openpi/training/config.py` generates
`pi0_robocasa_coffeesetupmug_hgdagger_r{1..8}`. Everything is openpi's LoRA default except the
schedule and run length:

| | |
|---|---|
| VLM (Gemma-2B) LoRA | rank 16, alpha 16, attn + ffn |
| Action expert (Gemma-300M) LoRA | rank 32, alpha 32, attn + ffn |
| Optimizer | AdamW b1 0.9, b2 0.95, eps 1e-8, wd 1e-10, grad-clip 1.0 |
| LR | cosine, warmup **200**, peak 2.5e-5, decay over **5 000**, end 2.5e-6 |
| Steps / batch | **5 000** / **32** |
| Checkpoints | every **1 000** steps, `keep_period=1000` so none are rotated away (saves at 1000/2000/3000/4000/4999) |
| EMA | off |

The warmup and decay lengths are deliberately *not* the openpi defaults (1 000 warmup, 30 000
decay): at a 5 000-step run those would spend a fifth of training warming up and end near peak LR
without ever decaying.

At 32 × 5 000 = 160 k samples, round 1 (~5 k frames) is seen roughly 30 times over. Watch the
loss curve — if it overfits, cut `num_train_steps` rather than the batch size, so the two vision
variants below stay comparable.

### Two vision variants per round

`Pi0Config.get_freeze_filter()` freezes `.*llm.*` except `.*lora.*`, so the LLM is LoRA-only — but
the PaliGemma **vision tower is fully trainable** under openpi's stock LoRA filter, and carries
AdamW moments accordingly. That is upstream behavior, not something added here, but on a few
thousand frames it means a ~400M-param vision encoder is being fully finetuned. Each round
therefore has two configs reading the same dataset:

| Variant | Config | Trainable |
|---|---|---|
| `default` | `pi0_robocasa_coffeesetupmug_hgdagger_r<r>` | 1.74 GiB — LoRA adapters + **vision tower** + state/action projections |
| `frozenvit` | `pi0_robocasa_coffeesetupmug_hgdagger_r<r>_frozenvit` | 0.20 GiB — LoRA adapters + state/action projections |

```bash
$P scripts/hgdagger_round.py --round 1 --variant both --stages convert norm train eval
```

runs both in turn on round 1's data (`--variant default` / `frozenvit` for one at a time). For
round 2+, `--serve-variant` picks which one's checkpoint drives the next collection round.

## GPU (single RTX 4090, 24 GB) — measured

| batch | `XLA_PYTHON_CLIENT_MEM_FRACTION` | result |
|---|---|---|
| 8 | 0.9 | fits |
| 16 | 0.95 | fits |
| 32 | 0.95 | fits |
| 4 or 8 | `XLA_PYTHON_CLIENT_PREALLOCATE=false` | **OOM** |

The only configuration that fails is disabling XLA's preallocating arena — the train step then
fragments and dies asking for another ~6–7 GB. Always leave preallocation on; the driver passes
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (`--mem-fraction`). The train state
itself is 7.0 GiB (5.25 bf16 frozen + 1.74 fp32 trainable): `train.py` casts frozen params to
bf16, and `_merge_params` does that cast on the host, so the 12.1 GiB fp32 checkpoint never lands
on the GPU whole.

Serving and training cannot coexist on this card (the server preallocates ~18 GB), which is why
the `train` and `eval` stages stop the server first.

## wandb

Project `semantic-corrections`, entity `robin-lab` (passed as `WANDB_ENTITY` — it isn't a
`TrainConfig` field), run name `hgdagger-rounds-coffeesetupmug-r<r>` (plus `-frozenvit` for that
variant, so the two show up as separate runs on the same data).

Data **collection** logs nothing to wandb — it writes `stats.json` (per-episode success, step
count, intervention flag, discard count) and `rollout_*.mp4` into the round directory.

The `eval` stage resumes that same run via `<checkpoint_dir>/wandb_id.txt` and logs 3 rollout
videos per checkpoint under `eval_videos/rollout_<i>`, plus `eval/success_rate`. They are plotted
against a custom `eval/step` axis, since evaluation runs after training has already advanced
wandb's global step and wandb refuses to go backwards on it.

## Collecting by hand

```bash
cd ../semantic_corrections
$P scripts/run_pi0_hitl.py --config configs/hitl/hgdagger_coffee.yaml \
    --run.output-dir ../openpi-robocasa/expdata/hgdagger/CoffeeSetupMug/round_1 \
    --run.demo-start-idx 0 --run.num-episodes 10
```

Keys during an episode (the matplotlib window must have focus — `update()` flushes events every
step):

| Key | Effect |
|---|---|
| SpaceMouse | take over; release to hand control back to the policy |
| `R` | reset to the start of the current intervention (retry a botched correction) |
| `S` | **discard** this episode — not saved, does not count, immediately re-run |
| `K` | keep this episode even though the success check never fired |
| `Tab` | end the episode now (discarded unless it succeeded) |

**An episode counts toward the 10 only if it completed the task *and* contained a human
takeover.** Timeouts, failures and `S`-discards are re-run instead, so a round is the first 10
successful intervened demos, not the first 10 attempts. Two independent abort guards stop the loop
from spinning forever:

- 20 consecutive episodes with no takeover (`run.max_discarded_episodes`) — usually means the
  SpaceMouse isn't being read (`hitl.vendor_id`/`product_id` in `configs/local/default.yaml`,
  currently `9583`/`50734`)
- 30 consecutive unsuccessful episodes (`run.max_failed_episodes`) — either the task isn't being
  finished, or RoboCasa's success check never fires for this scene, in which case `K` is the
  escape hatch and `task.success` thresholds are worth a look

Each run writes its own log — `stats.json` for a fresh round, `stats_from_demo_<i>.json` when
resuming — so a top-up run can't clobber the earlier run's record. It records every attempt with
its `discarded_reason` (`skipped` / `failure` /
`no_intervention` / `null`), so the accept rate per round is auditable. Set
`--run.no-require-success` or `--run.no-require-intervention` to relax either gate.

## Known caveats

- **Train/test image resolution.** Training images are stored at 128×128 and upsampled to 224 by
  the model transform; at rollout time the 512×512 render is downsampled to 224. Pre-existing in
  this pipeline, unchanged here.
- **`info["success"]`.** RoboCasa's own success check frequently never fires in these demos, so a
  0% success rate does not by itself mean imitation failed — watch the eval videos.
- **Mug/EEF weld is ON during collection, OFF during eval.** `task.weld_on_grasp: true` welds the
  mug to the end effector once a grasp is detected (a gripper-open command breaks it), so
  SpaceMouse corrections don't drop it mid-intervention. `eval_hgdagger_checkpoints.py` builds a
  plain env with no weld, so a marginal grasp the weld held together in the training data can
  still fail at rollout. Keep the setting identical across *all* rounds — flipping it partway
  makes the aggregated dataset internally inconsistent. Note also that `run_pi0_hitl.py` records
  episode metadata *after* welding, so a collected demo's `model_file` MJCF carries the weld
  constraint into any sim state later restored from it (the round's init state file does not).
- **Fixed init state means no scene diversity.** With one start state, `object_pose.seed` changes
  nothing; the only variation across a round's 10 demos is pi0's sampling noise and your
  corrections. That is the intended design here, but it makes the resulting policy a
  single-scene specialist.
