#!/usr/bin/env python
"""Quick test of imports."""

import sys
sys.path.insert(0, "src")

try:
    from tinyfold.model.resfold import ResidueDenoiser, ResFoldPipeline
    print("resfold import OK")
except ImportError as e:
    print(f"resfold import FAILED: {e}")

try:
    from tinyfold.model.resfold.onestep import ResFoldOneStep
    print("resfold_onestep import OK")
except ImportError as e:
    print(f"resfold_onestep import FAILED: {e}")

print("Done")
