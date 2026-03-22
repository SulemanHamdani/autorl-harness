from __future__ import annotations

from bootstrap import bootstrap_autorl_paths


def main():
    bootstrap_autorl_paths()
    from autorl.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
