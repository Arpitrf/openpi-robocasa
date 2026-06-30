from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="robocasa/robocasa365_checkpoints",
    allow_patterns="pi0/pi0_robocasa_pretrain_human300/multitask_learning/75000/*",
    local_dir="~/.cache/openpi/robocasa/robocasa365_checkpoints",
)