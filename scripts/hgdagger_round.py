"""Drive one HG-DAGGER round end to end: serve -> collect -> convert -> norm -> train -> eval.

HG-DAGGER (human-gated DAgger): the policy drives, the human takes over with the SpaceMouse when
it goes wrong, and the round's data is the resulting mixed-agent rollouts. Round r trains on the
*aggregated* demos from rounds 1..r, always re-LoRA-ing from the RoboCasa pretrain checkpoint (see
`hgdagger_lora_configs` in src/openpi/training/config.py). Every episode of every round starts
from one fixed recorded sim state, so rounds differ only in what the policy and the human did.

Stages (default: all of them, in order):
  serve    launch the policy server this round is collected under -- round 1 serves the RoboCasa
           pretrain checkpoint, round r>1 serves round r-1's trained checkpoint
  collect  run semantic_corrections/scripts/run_pi0_hitl.py until N intervened demos land in
           <expdata>/hgdagger/<env>/round_<r>/ (this is the stage that needs a human)
  convert  aggregate rounds 1..r into the LeRobot dataset hgdagger_<env>_r<r>
  norm     compute_norm_stats.py for this round's train config
  train    train.py (stops the policy server first -- one 24GB GPU can't hold both)
  eval     roll out each saved checkpoint from the fixed init state, log videos to the wandb run

Usage:
    python scripts/hgdagger_round.py --round 1
    python scripts/hgdagger_round.py --round 1 --stages convert norm train eval
    python scripts/hgdagger_round.py --round 2 --dry-run
"""

import argparse
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
# semantic_corrections lives beside this repo and owns the SpaceMouse / recorder loop.
DEFAULT_SC_ROOT = REPO_ROOT.parent / "semantic_corrections"
PRETRAIN_POLICY_DIR = (
    "hf://robocasa/robocasa365_checkpoints/pi0/pi0_robocasa_pretrain_human300/"
    "multitask_learning/75000"
)
PRETRAIN_CONFIG = "pi0_robocasa_pretrain_human300"
ALL_STAGES = ("serve", "collect", "convert", "norm", "train", "eval")


def round_dir(expdata: pathlib.Path, env_name: str, r: int) -> pathlib.Path:
    return expdata / "hgdagger" / env_name / f"round_{r}"


def config_name(env_name: str, r: int, variant: str = "default") -> str:
    suffix = "_frozenvit" if variant == "frozenvit" else ""
    return f"pi0_robocasa_{env_name.lower()}_hgdagger_r{r}{suffix}"


def repo_id(env_name: str, r: int) -> str:
    return f"hgdagger_{env_name.lower()}_r{r}"


def exp_name(env_name: str, r: int, variant: str = "default") -> str:
    suffix = "-frozenvit" if variant == "frozenvit" else ""
    return f"hgdagger-rounds-{env_name.lower()}-r{r}{suffix}"


def count_demos(d: pathlib.Path) -> int:
    return len([p for p in d.glob("demo_*.hdf5")]) if d.exists() else 0


def load_manifest(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"rounds": {}}


def save_manifest(path: pathlib.Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def run(cmd, *, cwd=None, env=None, dry_run=False, check=True):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ (cd {cwd or os.getcwd()} && {printable})\n", flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=cwd, env=env, check=check).returncode


def port_open(host: str, port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def policy_for_round(args, r: int) -> tuple[str, str]:
    """(config, dir) of the policy that *drives* round r's rollouts."""
    if r == 1:
        return PRETRAIN_CONFIG, PRETRAIN_POLICY_DIR
    prev = config_name(args.env_name, r - 1, args.serve_variant)
    exp_dir = REPO_ROOT / "checkpoints" / prev / exp_name(args.env_name, r - 1, args.serve_variant)
    steps = sorted(int(p.name) for p in exp_dir.iterdir() if p.is_dir() and p.name.isdigit()) if exp_dir.exists() else []
    if not steps:
        raise SystemExit(f"round {r} needs round {r - 1}'s checkpoint, none found under {exp_dir}")
    return prev, str(exp_dir / str(args.serve_step or steps[-1]))


def stage_serve(args, r: int, pidfile: pathlib.Path) -> None:
    if port_open(args.host, args.port):
        print(f"policy server already up on {args.host}:{args.port} -- leaving it alone")
        return
    cfg, policy_dir = policy_for_round(args, r)
    cmd = [
        args.python, str(REPO_ROOT / "scripts" / "serve_policy.py"), f"--port={args.port}",
        "policy:checkpoint", f"--policy.config={cfg}", f"--policy.dir={policy_dir}",
    ]
    print(f"\n$ (cd {REPO_ROOT} && {' '.join(cmd)}) &\n", flush=True)
    if args.dry_run:
        return
    log = pidfile.with_suffix(".log")
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT)
    pidfile.write_text(str(proc.pid))
    for _ in range(args.serve_timeout):
        if port_open(args.host, args.port):
            print(f"policy server up (pid {proc.pid}, log {log})")
            return
        if proc.poll() is not None:
            raise SystemExit(f"policy server exited with {proc.returncode}; see {log}")
        time.sleep(1)
    raise SystemExit(f"policy server did not open port {args.port} in {args.serve_timeout}s; see {log}")


def stop_serve(pidfile: pathlib.Path, dry_run: bool = False) -> None:
    if not pidfile.exists():
        return
    pid = int(pidfile.read_text().strip())
    print(f"stopping policy server (pid {pid}) -- training needs the whole GPU")
    if not dry_run:
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(30):
                time.sleep(1)
                os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    pidfile.unlink(missing_ok=True)


def stage_collect(args, r: int, rdir: pathlib.Path) -> None:
    have = count_demos(rdir)
    if have >= args.demos_per_round:
        print(f"round {r} already has {have}/{args.demos_per_round} demos in {rdir} -- skipping collect")
        return
    if not port_open(args.host, args.port):
        raise SystemExit(f"no policy server on {args.host}:{args.port} -- run the 'serve' stage first")
    if have:
        print(f"resuming round {r}: {have} demos already recorded, collecting {args.demos_per_round - have} more")
    cmd = [
        args.python, "scripts/run_pi0_hitl.py",
        "--config", args.collect_config,
        "--run.output-dir", str(rdir),
        "--run.demo-start-idx", str(have),
        "--run.num-episodes", str(args.demos_per_round - have),
        "--server.port", str(args.port),
    ]
    run(cmd, cwd=args.sc_root, dry_run=args.dry_run)


def stage_convert(args, r: int) -> None:
    dirs = [str(round_dir(args.expdata, args.env_name, i)) for i in range(1, r + 1)]
    missing = [d for d in dirs if not count_demos(pathlib.Path(d))]
    if missing and not args.dry_run:
        raise SystemExit(f"no demos found in {missing} -- collect those rounds first")
    cmd = [
        args.python, "convert_hitl_hdf5_to_lerobot.py",
        "--repo_name", repo_id(args.env_name, r),
        "--round_dirs", *dirs,
        "--preintv_window", str(args.preintv_window),
    ]
    run(cmd, cwd=REPO_ROOT / "examples" / "robocasa", dry_run=args.dry_run)


def stage_norm(args, r: int, variant: str) -> None:
    cfg = config_name(args.env_name, r, variant)
    if r > 1 and not args.recompute_norm_stats:
        # Round r>1 warm-starts from round r-1's weights, so it MUST keep round r-1's
        # normalization: the loaded weights were fit under it, and re-normalizing the same
        # observations differently is exactly as damaging as feeding the model rescaled inputs.
        #
        # Recomputing is also actively unsafe here. RunningStats derives std from
        # sqrt(E[x^2] - E[x]^2), which loses all precision on near-constant, large-magnitude
        # dims (e.g. actions[11]: mean -1.0005, std ~1e-3) -- on the round-2 sample several such
        # dims collapsed to std 0, and Normalize's `std + 1e-6` divisor then turned a 1e-3
        # deviation into ~1e3, taking step-0 loss from 0.55 to 17430.
        prev = REPO_ROOT / "assets" / config_name(args.env_name, r - 1, variant) / repo_id(args.env_name, r - 1)
        dst = REPO_ROOT / "assets" / cfg / repo_id(args.env_name, r)
        src_file = prev / "norm_stats.json"
        if not src_file.exists():
            raise SystemExit(f"round {r} inherits norm stats from {src_file}, which does not exist")
        print(f"\ncopying norm stats {src_file} -> {dst}/norm_stats.json (warm start keeps round "
              f"{r - 1}'s normalization; pass --recompute-norm-stats to override)")
        if not args.dry_run:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst / "norm_stats.json")
        return
    run(
        [args.python, "scripts/compute_norm_stats.py", f"--config-name={cfg}"],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
        dry_run=args.dry_run,
    )


def stage_train(args, r: int, variant: str, pidfile: pathlib.Path) -> None:
    stop_serve(pidfile, args.dry_run)
    cfg = config_name(args.env_name, r, variant)
    ckpt_dir = REPO_ROOT / "checkpoints" / cfg / exp_name(args.env_name, r, variant)
    cmd = [args.python, "scripts/train.py", cfg, f"--exp-name={exp_name(args.env_name, r, variant)}"]
    cmd.append("--resume" if ckpt_dir.exists() and not args.overwrite else "--overwrite")
    run(
        cmd,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "XLA_PYTHON_CLIENT_MEM_FRACTION": args.mem_fraction,
            "WANDB_ENTITY": args.wandb_entity,
        },
        dry_run=args.dry_run,
    )


def stage_eval(args, r: int, variant: str, pidfile: pathlib.Path) -> None:
    stop_serve(pidfile, args.dry_run)
    cmd = [
        args.python, "eval_hgdagger_checkpoints.py",
        "--config-name", config_name(args.env_name, r, variant),
        "--exp-name", exp_name(args.env_name, r, variant),
        "--init-state-hdf5", str(args.init_state),
        "--env-name", args.env_name,
        "--n-rollouts", str(args.eval_rollouts),
    ]
    run(
        cmd,
        cwd=REPO_ROOT / "examples" / "robocasa",
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "WANDB_ENTITY": args.wandb_entity},
        dry_run=args.dry_run,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, required=True, help="Round index, 1-based")
    p.add_argument("--stages", nargs="+", default=list(ALL_STAGES), choices=ALL_STAGES)
    p.add_argument("--env-name", default="CoffeeSetupMug")
    p.add_argument("--variant", default="default", choices=("default", "frozenvit", "both"),
                   help="Which LoRA variant to norm/train/eval. 'default' finetunes the PaliGemma "
                        "vision tower alongside the LoRA adapters (openpi's stock LoRA filter); "
                        "'frozenvit' freezes it; 'both' runs each in turn on the same data.")
    p.add_argument("--serve-variant", default="default", choices=("default", "frozenvit"),
                   help="For round r>1, which variant's round r-1 checkpoint drives collection.")
    p.add_argument("--demos-per-round", type=int, default=10)
    p.add_argument("--preintv-window", type=int, default=10)
    p.add_argument("--eval-rollouts", type=int, default=3)
    p.add_argument("--sc-root", type=pathlib.Path, default=DEFAULT_SC_ROOT)
    p.add_argument("--expdata", type=pathlib.Path, default=None,
                   help="Where rounds are stored. Default: <this repo>/expdata (gitignored), NOT "
                        "semantic_corrections/expdata -- the demos belong with the training code "
                        "that consumes them.")
    p.add_argument("--collect-config", default="configs/hitl/hgdagger_coffee.yaml",
                   help="HITL YAML, relative to --sc-root")
    p.add_argument("--init-state", type=pathlib.Path,
                   default=REPO_ROOT / "init_states" / "CoffeeMugSetup" / "l0" / "demo_0_raw.hdf5")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--serve-step", type=int, default=None,
                   help="Checkpoint step to serve for round r>1 (default: the latest)")
    p.add_argument("--serve-timeout", type=int, default=600)
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    # 0.9, not 0.95: in-training sim eval needs GPU headroom for MuJoCo's GL buffers alongside
    # XLA's preallocated arena. Batch 8 fits comfortably at 0.9 (measured).
    p.add_argument("--mem-fraction", default="0.9")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "robin-lab"))
    p.add_argument("--recompute-norm-stats", action="store_true",
                   help="Recompute norm stats for round r>1 instead of inheriting round r-1's. "
                        "Only correct if that round is NOT warm-starting from r-1's weights.")
    p.add_argument("--overwrite", action="store_true", help="Restart training from scratch instead of --resume")
    p.add_argument("--dry-run", action="store_true", help="Print every command without running it")
    args = p.parse_args()

    if args.expdata is None:
        args.expdata = REPO_ROOT / "expdata"
    if not args.init_state.exists():
        raise SystemExit(f"init state not found: {args.init_state}")
    if shutil.which("nvidia-smi") is None:
        print("warning: nvidia-smi not found; GPU checks skipped")

    r = args.round
    rdir = round_dir(args.expdata, args.env_name, r)
    rdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.expdata / "hgdagger" / args.env_name / "manifest.json"
    manifest = load_manifest(manifest_path)
    pidfile = args.expdata / "hgdagger" / args.env_name / "serve.pid"

    variants = ("default", "frozenvit") if args.variant == "both" else (args.variant,)
    served_cfg, served_dir = policy_for_round(args, r) if r == 1 or "serve" in args.stages else ("", "")
    entry = manifest["rounds"].setdefault(str(r), {})
    entry.update({
        "round_dir": str(rdir),
        "policy_config": served_cfg,
        "policy_dir": served_dir,
        "train_configs": [config_name(args.env_name, r, v) for v in variants],
        "exp_names": [exp_name(args.env_name, r, v) for v in variants],
        "repo_id": repo_id(args.env_name, r),
        "init_state": str(args.init_state),
        "demos_target": args.demos_per_round,
    })

    for stage in ALL_STAGES:
        if stage not in args.stages:
            continue
        print(f"\n=== round {r} · stage {stage} ===")
        if stage == "serve":
            stage_serve(args, r, pidfile)
        elif stage == "collect":
            stage_collect(args, r, rdir)
        elif stage == "convert":
            stage_convert(args, r)
        elif stage == "norm":
            for v in variants:
                stage_norm(args, r, v)
        elif stage == "train":
            for v in variants:
                stage_train(args, r, v, pidfile)
        elif stage == "eval":
            for v in variants:
                stage_eval(args, r, v, pidfile)
        entry.setdefault("stages_done", [])
        if stage not in entry["stages_done"] and not args.dry_run:
            entry["stages_done"].append(stage)
        entry["demos_collected"] = count_demos(rdir)
        if not args.dry_run:
            save_manifest(manifest_path, manifest)

    print(f"\nround {r}: {count_demos(rdir)} demos in {rdir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
