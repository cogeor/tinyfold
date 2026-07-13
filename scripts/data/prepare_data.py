#!/usr/bin/env python
"""Thin shim -> tinyfold.cli.prepare_data.

Kept so ``python scripts/data/prepare_data.py ...`` keeps working; the
canonical entry point is the ``tinyfold-prepare-data`` console script.
"""
from tinyfold.cli.prepare_data import main

if __name__ == "__main__":
    main()
