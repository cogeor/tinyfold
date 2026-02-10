#!/usr/bin/env python
"""Test models package import without warnings."""

import sys
import warnings

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

# Capture warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    import models

    if w:
        print("WARNINGS:")
        for warning in w:
            print(f"  {warning.category.__name__}: {warning.message}")
    else:
        print("No warnings!")

print(f"Available models: {models.list_models()}")
