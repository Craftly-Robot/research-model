"""Consolidated Craftly command line entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.craftly.runtime.engine import CraftlyRuntime
from src.craftly.shared.schemas import CraftlyRunRequest


async def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    report = await CraftlyRuntime().run(
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
    return report.model_dump()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the consolidated Craftly runtime.")
    parser.add_argument("--prompt", default="build a secure login API with tests")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--policy-mode", choices=["strict", "development"], default="development")
    parser.add_argument("--agent-backend-mode", choices=["auto", "active", "mock"], default="mock")
    parser.add_argument("--max-agent-nodes", type=int, default=3)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--no-security", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(asyncio.run(run_smoke(args)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


