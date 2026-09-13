#!/usr/bin/env python
"""Thin launcher so ``python linswap.py ...`` works from a checkout without installing;
``pip install -e .`` provides the ``linswap`` console command (see src/linswap/cli.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from linswap.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
