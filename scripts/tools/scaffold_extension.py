#!/usr/bin/env python
"""Scaffold files for a new model/loss training extension.

Pure-Python utility to speed up onboarding without introducing dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path


MODEL_TEMPLATE = """from torch import nn


class {class_name}(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # TODO: define layers

    def forward(self, *args, **kwargs):
        raise NotImplementedError
"""


LOSS_TEMPLATE = """import torch


def {fn_name}(pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    # TODO: implement loss
    return ((pred - target) ** 2).mean()
"""


TEST_TEMPLATE = """def test_{slug}_scaffold_import():
    from {model_import} import {class_name}
    assert {class_name} is not None
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold TinyFold extension files")
    parser.add_argument("--name", required=True, help="Extension slug, e.g. my_model")
    parser.add_argument("--with-loss", action="store_true", help="Also scaffold a loss function")
    args = parser.parse_args()

    slug = args.name.strip().lower().replace("-", "_")
    class_name = "".join(part.capitalize() for part in slug.split("_"))
    model_path = Path("src/tinyfold/model/extensions") / f"{slug}.py"
    test_path = Path("tests/unit") / f"test_{slug}_scaffold.py"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        model_path.write_text(MODEL_TEMPLATE.format(class_name=class_name), encoding="utf-8")
        print(f"Created {model_path}")
    else:
        print(f"Skipped existing {model_path}")

    if args.with_loss:
        loss_path = Path("src/tinyfold/model/losses") / f"{slug}.py"
        fn_name = f"{slug}_loss"
        if not loss_path.exists():
            loss_path.write_text(LOSS_TEMPLATE.format(fn_name=fn_name), encoding="utf-8")
            print(f"Created {loss_path}")
        else:
            print(f"Skipped existing {loss_path}")

    model_import = f"tinyfold.model.extensions.{slug}"
    if not test_path.exists():
        test_path.write_text(
            TEST_TEMPLATE.format(slug=slug, model_import=model_import, class_name=class_name),
            encoding="utf-8",
        )
        print(f"Created {test_path}")
    else:
        print(f"Skipped existing {test_path}")


if __name__ == "__main__":
    main()

