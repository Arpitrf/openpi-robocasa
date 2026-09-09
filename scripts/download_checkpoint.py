import os

from huggingface_hub import snapshot_download

# snapshot_download's local_dir does not expand `~` -- passing it unexpanded silently creates a
# literal directory named "~" under the current working directory instead of the home directory.
snapshot_download(
    repo_id="robocasa/robocasa365_checkpoints",
    allow_patterns="pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/*",
    local_dir=os.path.expanduser("~/.cache/openpi/robocasa/robocasa365_checkpoints"),
)