#!/usr/bin/env python
"""Thin shim -> tinyfold.cli.prepare_esm2.

Kept so ``python scripts/prepare_esm2_embeddings.py ...`` keeps working; the
canonical entry point is the ``tinyfold-embed-esm`` console script.
"""
import sys

from tinyfold.cli.prepare_esm2 import main

if __name__ == "__main__":
    sys.exit(main())
