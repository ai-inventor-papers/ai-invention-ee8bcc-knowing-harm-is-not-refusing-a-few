#!/usr/bin/env python3
"""Entry-point shim for the iteration-2 safety-screen experiment.

The full driver lives in method2.py (the artifact plan's canonical module
name, mirroring the iteration-1 method.py layout); this shim exists so that
a downstream consumer invoking ./method.py gets the identical pipeline.

Usage (identical to method2.py):
    uv run python method.py --test
    uv run python method.py --stage mini|partb|parta|partd|full [--models ...] [--light] [--force]
    uv run python method.py --assemble-only
"""

from __future__ import annotations

import sys


def main() -> None:
    import method2  # noqa: F401 - full driver (module-level argparse in main)
    method2.main()


if __name__ == "__main__":
    sys.exit(main())