"""Native release gate entrypoint for the final Craftly architecture."""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404
import sys
from types import SimpleNamespace

from src.craftly.architecture_integrity import run_architecture_integrity
from src.craftly.production_readiness import run_production_readiness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Craftly local release gate.")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Reuse a full unittest run already executed by the same parent qualification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    readiness = run_production_readiness(mode="dev").model_dump()
    architecture = run_architecture_integrity().model_dump(mode="json")
    completed = (
        SimpleNamespace(returncode=0, stdout="tests reused from parent qualification", stderr="")
        if args.skip_tests
        else subprocess.run(  # nosec B603
            [sys.executable, "-m", "unittest"],
            capture_output=True,
            text=True,
            check=False,
        )
    )
    passed = completed.returncode == 0 and architecture["status"] == "passed"
    print(
        json.dumps(
            {
                "decision": "release" if passed else "block",
                "passed": passed,
                "architecture_integrity": architecture,
                "production_readiness": readiness,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()

