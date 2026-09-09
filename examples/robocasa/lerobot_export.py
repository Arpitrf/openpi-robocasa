"""LeRobot dataset export for recorded HITL episodes.

Copied verbatim from Arpitrf/semantic_corrections (semantic_corrections/datasets/lerobot_export.py,
`frs` branch @ f790a5e -- the only branch with this file; `point-set` replaced it with
`episode_recorder.py`, which writes robomimic-format HDF5 for live recording, not a LeRobot
training dataset). Two changes:
  1. `output_path` (lerobot_home/repo_name) is computed and used to check/clear an existing
     dataset, but was never actually passed to `LeRobotDataset.create()` -- so `lerobot_home`
     silently did nothing and every dataset landed in the global default cache dir regardless.
     Added `root=output_path` below to fix that.
  2. Added an `is_intervention` feature (SIRIUS-style human/robot label -- see
     convert_hitl_hdf5_to_lerobot.py) so training can reweight toward intervention frames; see
     LeRobotRobocasaHitlDataConfig.intervention_p_target in training/config.py.
"""

import logging
import os
import shutil

from openpi_client import image_tools


def save_episodes_as_lerobot(episodes, repo_name, lerobot_home=None):
    """Convert a list of EpisodeRecorder objects into a LeRobot dataset."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if lerobot_home is None:
        lerobot_home = os.path.expanduser("~/.cache/huggingface/lerobot")

    output_path = os.path.join(lerobot_home, repo_name)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="panda",
        fps=20,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float64",
                "shape": (16,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float64",
                "shape": (12,),
                "names": ["actions"],
            },
            "is_intervention": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["is_intervention"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for recorder in episodes:
        for i in range(len(recorder.actions)):
            img = image_tools.resize_with_pad(recorder.images[i], 128, 128)
            wrist = image_tools.resize_with_pad(recorder.wrist_images[i], 128, 128)
            img = image_tools.convert_to_uint8(img)
            wrist = image_tools.convert_to_uint8(wrist)

            dataset.add_frame(
                {
                    "image": img,
                    "wrist_image": wrist,
                    "state": recorder.states[i],
                    "actions": recorder.actions[i],
                    "is_intervention": recorder.is_intervention[i],
                    "task": recorder.task_lang,
                }
            )
        dataset.save_episode()

    logging.info("LeRobot dataset saved to %s", output_path)
