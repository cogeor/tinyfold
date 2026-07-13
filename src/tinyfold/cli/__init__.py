"""Console-script entry points for TinyFold.

Each module here exposes a ``main()`` wired into ``[project.scripts]`` in
pyproject.toml (``tinyfold-predict``, ``tinyfold-embed-esm``,
``tinyfold-prepare-data``). The thin ``scripts/*.py`` shims re-export these
so the documented ``python scripts/...`` invocations keep working.
"""
