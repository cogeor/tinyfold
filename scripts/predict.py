#!/usr/bin/env python
"""Thin shim -> tinyfold.cli.predict.

Kept so ``python scripts/predict.py ...`` keeps working; the canonical entry
point is the ``tinyfold-predict`` console script ([project.scripts]).
"""
from tinyfold.cli.predict import main

if __name__ == "__main__":
    main()
