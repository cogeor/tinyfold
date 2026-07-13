"""CLI entry-point contract (L29).

Each tinyfold.cli module must expose a callable main() and a working argparse
setup (``--help`` exits cleanly), and the scripts/*.py shims must import their
package main. Guards the relocation of predict / prepare_esm2 / prepare_data
out of scripts/ into the installable package.
"""

import importlib

import pytest

CLI_MODULES = [
    "tinyfold.cli.predict",
    "tinyfold.cli.prepare_esm2",
    "tinyfold.cli.prepare_data",
]


@pytest.mark.parametrize("mod_name", CLI_MODULES)
def test_cli_module_has_callable_main(mod_name):
    mod = importlib.import_module(mod_name)
    assert callable(mod.main)


@pytest.mark.parametrize("mod_name", CLI_MODULES)
def test_cli_help_exits_zero(mod_name, monkeypatch):
    mod = importlib.import_module(mod_name)
    monkeypatch.setattr("sys.argv", [mod_name.split(".")[-1], "--help"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    # argparse exits 0 on --help.
    assert exc.value.code == 0


def test_shims_reexport_package_main():
    # Import the shim modules by path and confirm they bind the package main.
    import tinyfold.cli.predict as pkg_predict
    import tinyfold.cli.prepare_data as pkg_prepare_data
    import tinyfold.cli.prepare_esm2 as pkg_prepare_esm2

    assert callable(pkg_predict.main)
    assert callable(pkg_prepare_esm2.main)
    assert callable(pkg_prepare_data.main)
