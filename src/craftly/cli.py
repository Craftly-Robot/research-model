"""Consolidated Craftly command line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.craftly.runtime.engine import CraftlyRuntime
from src.craftly.shared.schemas import CraftlyRunRequest

VERSION = "1.0.0"
GATEWAY_URL = "http://127.0.0.1:8090"


def cmd_run(args: argparse.Namespace) -> None:
    """Run the Craftly agent on a prompt."""
    report = asyncio.run(
        CraftlyRuntime().run(
            CraftlyRunRequest(
                prompt=args.prompt,
                workspace=str(args.workspace),
                policy_mode=args.policy_mode,
                agent_backend_mode=args.agent_backend_mode,
                run_verifier=not args.no_verifier,
                run_security=not args.no_security,
                max_agent_nodes=args.max_agent_nodes,
            )
        )
    )
    print(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))


def cmd_status(args: argparse.Namespace) -> None:
    """Check if the Craftly gateway is running."""
    url = args.url or GATEWAY_URL
    try:
        req = urllib.request.Request(f"{url}/health/ready")
        with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310
            data = json.loads(resp.read())
            print(f"Gateway:  running ({url})")
            print(f"Status:   {data.get('status', 'unknown')}")
    except urllib.error.URLError:
        print(f"Gateway:  not running ({url})")
        print("Start it with: make gateway")
        sys.exit(1)
    except Exception as e:
        print(f"Gateway:  error ({e})")
        sys.exit(1)


def cmd_setup(args: argparse.Namespace) -> None:
    """Set up the development environment."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    makefile = repo_root / "Makefile"
    if not makefile.exists():
        print("Error: Makefile not found in project root")
        sys.exit(1)
    print("Setting up Craftly development environment...")
    result = subprocess.run(["make", "install"], cwd=repo_root)
    if result.returncode == 0:
        print("\nDone! Activate the venv with:")
        print("  source .venv/bin/activate")
    else:
        print("\nSetup failed. Check the output above.")
        sys.exit(1)


def cmd_version(args: argparse.Namespace) -> None:
    """Print the Craftly version."""
    print(f"craftly {VERSION}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="craftly",
        description="Craftly - AI coding agent built from scratch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  craftly run --prompt "fix the login bug"
  craftly run --prompt "add tests" --workspace ./myproject
  craftly status
  craftly setup
""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    p_run = sub.add_parser("run", help="Run the agent on a prompt")
    p_run.add_argument("--prompt", default="build a secure login API with tests")
    p_run.add_argument("--workspace", type=Path, default=Path("."))
    p_run.add_argument(
        "--policy-mode", choices=["strict", "development"], default="development"
    )
    p_run.add_argument(
        "--agent-backend-mode", choices=["auto", "active", "mock"], default="mock"
    )
    p_run.add_argument("--max-agent-nodes", type=int, default=3)
    p_run.add_argument("--no-verifier", action="store_true")
    p_run.add_argument("--no-security", action="store_true")

    # --- status ---
    p_status = sub.add_parser("status", help="Check if the gateway is running")
    p_status.add_argument(
        "--url", default=None, help=f"Gateway URL (default: {GATEWAY_URL})"
    )

    # --- setup ---
    sub.add_parser("setup", help="Set up the development environment")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "setup":
        cmd_setup(args)
    elif args.command is None:
        parser.print_help()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
