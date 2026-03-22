"""CLI entrypoint placeholder for the AutoResearch-RL scaffold."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from autorl.registry import list_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autorl",
        description="AutoResearch-inspired RL experiment scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command")

    tasks_parser = subparsers.add_parser("tasks", help="Inspect registered tasks.")
    tasks_subparsers = tasks_parser.add_subparsers(dest="tasks_command")
    tasks_subparsers.add_parser("list", help="List available task packages.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "tasks" and args.tasks_command == "list":
        tasks = [asdict(task) for task in list_tasks()]
        print(json.dumps(tasks, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
