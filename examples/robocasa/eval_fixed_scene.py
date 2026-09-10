"""Evaluate a policy by replaying the EXACT scene(s) it was trained on, instead of a randomized
layout. `examples/robocasa/main.py` calls `env.reset()` with `split=target`, which randomizes
object/counter placement every episode -- for a policy trained on only 1-3 fixed HITL demos, that
means eval is testing generalization to scenes the policy never saw, not whether it learned the
demonstrated behavior at all.

This script instead loads the first simulator state recorded in each training demo_0.hdf5 (via
`restore_sim_state_from_hdf5` from `sim_state.py` -- copied from Arpitrf/semantic_corrections,
where it's the state-restore mechanism behind `run_pi0_hitl.py`'s `--load-state-hdf5` /
`--load-state-frame`, the same "load a specific frame from the same hdf5" idea) and rolls the
policy out from there. Unlike the reset_to() this script used to hand-roll, this also restores
`ctrl` and `gripper_actions`, which matters if `--args.frame` points mid-episode (e.g. a
post-grasp state) rather than frame 0.

Weld-on-grasp (`--args.weld_on_grasp`, default True): `semantic_corrections`'s own
`configs/tasks/coffee.yaml` sets `weld_on_grasp: true` for CoffeeSetupMug, and `run_pi0_hitl.py`
applies it -- once the gripper grasps the target object, a rigid MuJoCo weld couples it to the
EEF so it can't slip/rotate independently of the gripper. The training demos were recorded with
this on; this script previously had no equivalent at all, so evaluated rollouts let the object
move freely after grasp in a way the policy never saw during training. Ported from
`mug_weld.py`, replicating `run_pi0_hitl.py`'s `_maybe_apply_grasp_weld`/
`_maybe_break_weld_on_gripper_open` trigger logic exactly (weld on grasp detection, unless this
step's action is a gripper-open command; break the weld when an open command is issued).

Verified: state restore reproduces the recorded scene's first frame pixel-for-pixel (compared
against `obs/robot0_agentview_left_image[0]` straight out of the hdf5, not just against this
script's own rendering), and `reset_from_xml_string` correctly re-derives task fixture references
(`_setup_kitchen_references`) from the loaded scene, so `info["success"]` reflects the actual
loaded scene rather than whatever `gym.make` randomized on construction.

CAVEAT: `info["success"]` measures the same task success-check used everywhere else in robocasa.
For the "semantic_corrections" HITL demos this repo currently trains on, that check never fires in
the source data either (see the caveat in convert_hitl_hdf5_to_lerobot.py) -- a 0% success
rate from this script does not necessarily mean the policy failed to imitate its training data,
only that it failed to complete the task, which the training data may not have done either.

Usage:
    python examples/robocasa/eval_fixed_scene.py --args.port 8000 \
        --args.hdf5-paths data/CoffeeSetupMug/2026-06-30-22-00/demo_0.hdf5 \
                          data/CoffeeSetupMug/2026-08-19-12-55/demo_0.hdf5 \
                          data/CoffeeSetupMug/2026-08-21-20-26/demo_0.hdf5 \
        --args.log_dir checkpoints/pi0_robocasa_coffeesetupmug_hitl_lora_all3/coffeesetupmug_lora_all3_run1 \
        --args.trials_per_scene 5
"""

import collections
import dataclasses
import json
import logging
import os
import pathlib

import gymnasium as gym
import imageio
import numpy as np
import tqdm
import tyro
from hitl_env import get_robosuite_env, get_robocasa_gym_wrapper, mark_gym_env_reset, refresh_gym_observation
from hitl_obs import obs_to_camera_frames, obs_to_policy_state
from mug_weld import apply_object_eef_weld, break_object_eef_weld
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from robocasa.utils.dataset_registry_utils import get_task_horizon
from robocasa.utils.env_utils import convert_action
from sim_state import object_is_grasped, restore_sim_state_from_hdf5


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    env_name: str = "CoffeeSetupMug"
    hdf5_paths: list[str] = dataclasses.field(default_factory=list)
    demo_name: str = "demo_0"
    frame: int = 0  # which frame of each hdf5 to restore -- 0 for the start, or e.g. a
    # post-grasp frame index to test just the placement phase
    trials_per_scene: int = 5
    log_dir: str = None
    seed: int = 7

    # Weld the grasped object rigidly to the EEF, matching how the training demos were recorded
    # (semantic_corrections's configs/tasks/coffee.yaml sets weld_on_grasp=true for this task).
    weld_on_grasp: bool = True
    weld_obj_name: str = "obj"
    weld_name: str = "eval_obj_eef_weld"
    weld_solref: str = "0.02 1"


def eval_main(args: Args) -> None:
    np.random.seed(args.seed)
    assert args.hdf5_paths, "Must pass at least one --args.hdf5-paths"

    task_horizon = get_task_horizon(args.env_name)
    horizon = int(task_horizon * 1.5)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    env = gym.make(f"robocasa/{args.env_name}", split="target", seed=args.seed)
    raw_env = get_robosuite_env(env)
    gym_wrapper = get_robocasa_gym_wrapper(env)

    for hdf5_path in args.hdf5_paths:
        scene_name = pathlib.Path(hdf5_path).parent.name

        log_path = f"{args.log_dir}/evals_fixed_scene/{args.env_name}/{scene_name}"
        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)

        total_successes = 0
        for episode_idx in tqdm.tqdm(range(args.trials_per_scene), desc=scene_name):
            env.reset()
            restore_sim_state_from_hdf5(raw_env, hdf5_path, demo_name=args.demo_name, frame=args.frame)
            mark_gym_env_reset(env)
            raw_obs = raw_env._get_observations(force_update=True)
            obs = gym_wrapper.get_observation(raw_obs)
            task_lang = obs["annotation.human.task_description"]
            action_plan = collections.deque()

            t = 0
            replay_images = []
            done = False
            weld_active = False
            logging.info(f"[{scene_name}] Starting episode {episode_idx + 1}...")
            while t < horizon:
                main_img, wrist_img = obs_to_camera_frames(obs)
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(main_img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                if not action_plan:
                    state = obs_to_policy_state(obs)
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state,
                        "prompt": task_lang,
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps
                    action_plan.extend(action_chunk[: args.replan_steps])

                raw_action = action_plan.popleft()
                action = convert_action(raw_action)
                obs, reward, done, truncated, info = env.step(action)
                done = info["success"]
                replay_img = np.ascontiguousarray(env.render())
                replay_img = image_tools.convert_to_uint8(replay_img)
                if t % 2 == 0 or t == horizon - 1 or done:
                    replay_images.append(replay_img)

                if args.weld_on_grasp:
                    # Action layout: index 6 is gripper; >0 = closed, <=0 = open. Mirrors
                    # run_pi0_hitl.py's _maybe_break_weld_on_gripper_open/_maybe_apply_grasp_weld.
                    gripper_cmd = float(np.asarray(raw_action).reshape(-1)[6])
                    if weld_active and gripper_cmd <= 0:
                        break_object_eef_weld(raw_env, args.weld_name)
                        weld_active = False
                    elif (
                        not weld_active
                        and gripper_cmd > 0
                        and args.weld_obj_name in raw_env.objects
                        and object_is_grasped(raw_env, args.weld_obj_name)
                    ):
                        apply_object_eef_weld(
                            raw_env,
                            object_name=args.weld_obj_name,
                            weld_name=args.weld_name,
                            solref=args.weld_solref,
                        )
                        weld_active = True
                        obs = refresh_gym_observation(env, raw_env, gym_wrapper)

                if done:
                    total_successes += 1
                    break
                t += 1

            suffix = "success" if done else "failure"
            imageio.mimwrite(
                pathlib.Path(log_path) / f"rollout_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=20,
            )
            logging.info(f"[{scene_name}] Episode {episode_idx + 1} success={done}")

        success_rate = total_successes / args.trials_per_scene
        logging.info(f"[{scene_name}] success rate: {total_successes}/{args.trials_per_scene}")
        with open(os.path.join(log_path, "stats.json"), "w") as f:
            json.dump({"num_episodes": args.trials_per_scene, "successes": total_successes, "success_rate": success_rate}, f, indent=4)

    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_main)
