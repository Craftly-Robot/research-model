"""Safe scheduler-to-PyTorch distributed process launcher.

The launcher accepts only the Craftly pretraining module and validated numeric
topology. It never accepts a shell command and replaces itself with torchrun so
signals and scheduler exit codes propagate without an intermediary process.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess  # nosec B404 - fixed scontrol argv, shell is never used
import sys
from pathlib import Path

ALLOWED_MODULE = "src.craftly.model_ops.pretrain_loop"
SAFE_HOST = re.compile(r"^[A-Za-z0-9._-]{1,253}$")


def _positive_int(value: str, *, name: str, maximum: int = 65536) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 0 <= parsed <= maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return parsed


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"scheduler did not provide required environment variable {name}")
    return value


def _slurm_master_address() -> str:
    existing = os.environ.get("MASTER_ADDR", "").strip()
    if existing:
        if not SAFE_HOST.fullmatch(existing):
            raise ValueError("MASTER_ADDR contains unsafe characters")
        return existing
    node_list = _required_env("SLURM_JOB_NODELIST")
    scontrol = shutil.which("scontrol")
    if not scontrol:
        raise RuntimeError("scontrol is required to resolve the Slurm rendezvous node")
    completed = subprocess.run(  # nosec B603 - canonical executable and fixed argv
        [str(Path(scontrol).resolve()), "show", "hostnames", node_list],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"scontrol hostname resolution failed: {completed.stderr[-1000:]}")
    host = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not SAFE_HOST.fullmatch(host):
        raise RuntimeError("scontrol returned an invalid rendezvous hostname")
    return host


def normalize_slurm_environment() -> None:
    rank = _positive_int(_required_env("SLURM_PROCID"), name="SLURM_PROCID")
    local_rank = _positive_int(_required_env("SLURM_LOCALID"), name="SLURM_LOCALID")
    world_size = _positive_int(_required_env("SLURM_NTASKS"), name="SLURM_NTASKS")
    if world_size < 1 or rank >= world_size:
        raise ValueError("invalid Slurm rank topology")
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": _slurm_master_address(),
            "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
        }
    )


def kubernetes_torchrun_argv(
    *, nodes: int, processes_per_node: int, training_args: list[str]
) -> list[str]:
    node_rank = _positive_int(_required_env("RANK"), name="RANK")
    if node_rank >= nodes:
        raise ValueError("Kubernetes replica rank exceeds configured node count")
    master_addr = _required_env("MASTER_ADDR")
    if not SAFE_HOST.fullmatch(master_addr):
        raise ValueError("MASTER_ADDR contains unsafe characters")
    master_port = _positive_int(
        os.environ.get("MASTER_PORT", "29500"), name="MASTER_PORT", maximum=65535
    )
    if master_port < 1024:
        raise ValueError("MASTER_PORT must be an unprivileged port")
    torchrun = shutil.which("torchrun")
    if not torchrun:
        raise RuntimeError("torchrun executable is required in the immutable training image")
    return [
        str(Path(torchrun).resolve()),
        "--nnodes",
        str(nodes),
        "--nproc-per-node",
        str(processes_per_node),
        "--node-rank",
        str(node_rank),
        "--master-addr",
        master_addr,
        "--master-port",
        str(master_port),
        "-m",
        ALLOWED_MODULE,
        *training_args,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch validated Craftly distributed workers.")
    parser.add_argument("--scheduler", choices=["kubernetes", "slurm"], required=True)
    parser.add_argument("--target", choices=["craftly", "megatron"], default="craftly")
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--processes-per-node", type=int, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.nodes <= 4096 or not 1 <= args.processes_per_node <= 64:
        parser.error("invalid distributed topology")
    if not args.training_args or args.training_args[0] != "--":
        parser.error("training arguments must follow --")
    args.training_args = args.training_args[1:]
    total_argument_bytes = 0
    for value in args.training_args:
        encoded_size = len(value.encode("utf-8", "strict"))
        total_argument_bytes += encoded_size
        if encoded_size > 8_192:
            parser.error("an individual training argument exceeds 8192 UTF-8 bytes")
        if "\x00" in value or any(character in value for character in ("\r", "\n")):
            parser.error("training arguments cannot contain null bytes or line delimiters")
    if total_argument_bytes > 131_072:
        parser.error("training arguments exceed the 131072-byte aggregate limit")
    forbidden = {"--cluster-plan-only", "--scheduler", "--megatron-root"}
    if args.target == "craftly" and forbidden.intersection(args.training_args):
        parser.error("scheduler worker received forbidden training arguments")
    if args.target == "megatron":
        root_value = os.environ.get("CRAFTLY_MEGATRON_ROOT", "")
        root = Path(root_value).expanduser().resolve() if root_value else None
        if len(args.training_args) < 3 or args.training_args[:2] != [
            "-u",
            str((root / "pretrain_gpt.py").resolve()) if root else "",
        ]:
            parser.error("Megatron target must execute CRAFTLY_MEGATRON_ROOT/pretrain_gpt.py")
    return args


def main() -> None:
    args = parse_args()
    if args.scheduler == "slurm":
        normalize_slurm_environment()
        argv = (
            [sys.executable, *args.training_args]
            if args.target == "megatron"
            else [sys.executable, "-u", "-m", ALLOWED_MODULE, *args.training_args]
        )
    else:
        argv = kubernetes_torchrun_argv(
            nodes=args.nodes,
            processes_per_node=args.processes_per_node,
            training_args=args.training_args,
        )
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if args.target == "craftly":
        # The executable and module are fixed constants; only bounded,
        # non-control-character trainer options remain variable.
        os.execve(  # nosemgrep: python.lang.security.audit.dangerous-os-exec-tainted-env-args.dangerous-os-exec-tainted-env-args
            argv[0],
            argv,
            environment,
        )
    from src.craftly.shared.progress import progress_from_options

    progress = progress_from_options(path=None, to_stdout=True)
    # argv[0] is sys.executable, argv[1] is the canonical trusted Megatron
    # pretrain_gpt.py path, and all remaining arguments passed the bounds above.
    child = subprocess.Popen(  # nosec B603  # nosemgrep: python.lang.security.audit.dangerous-os-exec-tainted-env-args.dangerous-os-exec-tainted-env-args
        argv,
        env=environment,
        stdin=subprocess.DEVNULL,
    )

    def forward(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    progress.emit("megatron_training", "running", message="Megatron rank process started")
    returncode = child.wait()
    progress.emit(
        "megatron_training",
        "complete" if returncode == 0 else "failed",
        message=f"Megatron process exited with code {returncode}",
        failure_class="runtime" if returncode else None,
    )
    progress.close()
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
