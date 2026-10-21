"""Checkout entry point for Factor Agent's shared command-line interface.

Run python main.py --help. Business logic and argument validation live in the
selected package modules; this file only forwards to the common dispatcher.
"""
from factor_research.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
