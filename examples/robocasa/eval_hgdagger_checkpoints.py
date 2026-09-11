"""Roll out each saved checkpoint of an HG-DAGGER round from the round's fixed initial state and
log the rollout videos back into that round's wandb run.

Every episode is restored from the same recorded sim state the round was collected from
(`init_states/<Task>/l0/demo_0_raw.hdf5`), so the only variation across the rollouts of one
checkpoint is pi0's sampling noise -- the videos answer "what does this checkpoint do on the exact
scene it was trained on", not "does it generalize".

Policies are loaded in-process (`policy_config.create_trained_policy`), so no policy server is
needed; checkpoints are loaded one at a time in the order given.

wandb: the run is resumed from `<checkpoint_dir>/wandb_id.txt` (written by scripts/train.py), so
videos land in the same run as the training curves rather than a new one. They are logged against
a custom `eval/step` axis instead of wandb's global step, because these evals run *after* training
has already logged up to the final step and wandb rejects going backwards on the global step.

Usage:
    python examples/robocasa/eval_hgdagger_checkpoints.py \
        --config-name pi0_robocasa_coffeesetupmug_hgdagger_r1 \
        --exp-name hgdagger-rounds-coffeesetupmug-r1 \
        --init-state-hdf5 init_states/CoffeeMugSetup/l0/demo_0_raw.hdf5

CAVEAT: `info["success"]` is robocasa's own task success check. The HITL demos this trains on
often never trigger it themselves (see convert_hitl_hdf5_to_lerobot.py), so a 0% success rate
means the task wasn't completed, not necessarily that imitation failed -- watch the videos.
"""

import collections
import dataclasses
import json
import logging
import pathlib
import re

import gymnasium as gym
import imageio
import numpy as np
import tqdm
import tyro
from hitl_env import get_robocasa_gym_wrapper, get_robosuite_env, mark_gym_env_reset, refresh_gym_observation
from mug_weld import apply_object_eef_weld, break_object_eef_weld
from hitl_obs import obs_to_camera_frames, obs_to_policy_state
from openpi_client import image_tools
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from sim_state import restore_sim_state_from_hdf5

import openpi.policies.policy_config as _policy_config
import openpi.training.config as _config


@dataclasses.dataclass
class Args:
    config_name: str
    exp_name: str
    init_state_hdf5: str
    env_name: str = "CoffeeSetupMug"
    demo_name: str = "demo_0"
    frame: int = 0
    # Rollouts per checkpoint. All start from the same state; they differ only by sampling noise.
    n_rollouts: int = 3
    # Checkpoint steps to evaluate. Empty -> every step directory found, ascending.
    steps: list[int] = dataclasses.field(default_factory=list)
    replan_steps: int = 5
    resize_size: int = 224
    horizon: int | None = None
    seed: int = 7
    log_to_wandb: bool = True
    # Match collection's task.weld_on_grasp (configs/hitl/hgdagger_coffee.yaml sets it true):
    # without the weld this grades the policy under different contact dynamics than it trained on.
    weld_on_grasp: bool = True
    weld_obj_name: str = "obj"
    weld_eef_body: str | None = None
    weld_name: str = "hitl_mug_eef_weld"
    weld_solref: str = "0.02 1"
    # Overrides where rollout mp4s/stats.json are written (default: <exp_dir>/evals/step_<N>).
    out_dir: str | None = None


def _checkpoint_steps(exp_dir: pathlib.Path) -> list[int]:
    steps = [int(p.name) for p in exp_dir.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoint step directories under {exp_dir}")
    return sorted(steps)


def _rollout(env, raw_env, gym_wrapper, policy, args: Args, horizon: int):
    """One episode from the fixed init state. Returns (success, frames)."""
    env.reset()
    restore_sim_state_from_hdf5(raw_env, args.init_state_hdf5, demo_name=args.demo_name, frame=args.frame)
    mark_gym_env_reset(env)
    obs = gym_wrapper.get_observation(raw_env._get_observations(force_update=True))
    task_lang = obs["annotation.human.task_description"]

    action_plan = collections.deque()
    frames, done, t = [], False, 0
    # The init state's MJCF carries no weld, so each rollout starts unwelded.
    weld_active = False
    while t < horizon and not done:
        main_img, wrist_img = obs_to_camera_frames(obs)
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(main_img, args.resize_size, args.resize_size))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

        if not action_plan:
            action_chunk = policy.infer(
                {
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": obs_to_policy_state(obs),
                    "prompt": task_lang,
                }
            )["actions"]
            assert len(action_chunk) >= args.replan_steps
            action_plan.extend(action_chunk[: args.replan_steps])

        action = action_plan.popleft()
        obs, _, _, _, info = env.step(convert_action(action))
        done = bool(info.get("success", False))

        # Same order as run_pi0_hitl.py: release on a gripper-open command, then re-weld on grasp.
        gripper_cmd = float(np.asarray(action).reshape(-1)[6])
        if weld_active and gripper_cmd <= 0:
            break_object_eef_weld(raw_env, args.weld_name)
            weld_active = False
        elif args.weld_on_grasp and not weld_active and gripper_cmd > 0:
            from robocasa.utils.object_utils import check_obj_grasped

            if args.weld_obj_name in raw_env.objects and check_obj_grasped(raw_env, args.weld_obj_name):
                apply_object_eef_weld(
                    raw_env,
                    object_name=args.weld_obj_name,
                    eef_body=args.weld_eef_body,
                    weld_name=args.weld_name,
                    solref=args.weld_solref,
                )
                weld_active = True
                obs = refresh_gym_observation(env, raw_env, gym_wrapper)
        if t % 2 == 0 or done:
            frames.append(image_tools.convert_to_uint8(np.ascontiguousarray(env.render())))
        t += 1
    return done, frames


def main(args: Args) -> None:
    np.random.seed(args.seed)
    train_config = _config.get_config(args.config_name)
    exp_dir = pathlib.Path(train_config.checkpoint_base_dir) / args.config_name / args.exp_name
    steps = args.steps or _checkpoint_steps(exp_dir)
    horizon = args.horizon or int(get_task_horizon(args.env_name) * 1.5)
    out_base = pathlib.Path(args.out_dir) if args.out_dir else exp_dir / "evals"

    run = None
    if args.log_to_wandb:
        import wandb

        id_file = exp_dir / "wandb_id.txt"
        if id_file.exists():
            run = wandb.init(id=id_file.read_text().strip(), resume="must", project=train_config.project_name)
        else:
            logging.warning("%s not found -- starting a standalone wandb run", id_file)
            run = wandb.init(name=f"{args.exp_name}-eval", project=train_config.project_name)
        # Videos are logged after training already advanced wandb's global step, so plot them
        # against an explicit eval/step axis instead.
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        wandb.define_metric("eval_videos/*", step_metric="eval/step")

    env = gym.make(f"robocasa/{args.env_name}", split=None, seed=args.seed, camera_widths=512, camera_heights=512)
    raw_env = get_robosuite_env(env)
    gym_wrapper = get_robocasa_gym_wrapper(env)

    all_stats = {}
    try:
        for step in steps:
            policy = _policy_config.create_trained_policy(train_config, exp_dir / str(step))
            out_dir = out_base / f"step_{step}"
            out_dir.mkdir(parents=True, exist_ok=True)

            successes, videos = 0, []
            for i in tqdm.tqdm(range(args.n_rollouts), desc=f"step {step}"):
                success, frames = _rollout(env, raw_env, gym_wrapper, policy, args, horizon)
                successes += int(success)
                mp4 = out_dir / f"rollout_{i}_{'success' if success else 'failure'}.mp4"
                imageio.mimwrite(str(mp4), [np.asarray(f) for f in frames], fps=20)
                videos.append((i, mp4, success))
                logging.info("step %d rollout %d: success=%s (%d frames)", step, i, success, len(frames))

            stats = {"step": step, "n_rollouts": args.n_rollouts, "successes": successes,
                     "success_rate": successes / args.n_rollouts}
            (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
            all_stats[step] = stats

            if run is not None:
                import wandb

                payload = {
                    "eval/step": step,
                    "eval/success_rate": stats["success_rate"],
                    "eval/successes": successes,
                }
                for i, mp4, success in videos:
                    payload[f"eval_videos/rollout_{i}"] = wandb.Video(
                        str(mp4), caption=f"step {step} · rollout {i} · {'success' if success else 'failure'}", format="mp4"
                    )
                wandb.log(payload)

            del policy  # release the checkpoint's device memory before loading the next one
    finally:
        env.close()
        if run is not None:
            run.finish()

    print(json.dumps(all_stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
