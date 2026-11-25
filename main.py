# Entry point for the CoSMIC API.
if __package__:
    from .apis.cores.apis import app
else:
    # Native `fastapi dev` imports this file as a top-level module. Register
    # the checkout as a package so the relative imports keep working.
    import sys
    import types
    from importlib import import_module
    from pathlib import Path

    _package = types.ModuleType("cosmic_core")
    _package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["cosmic_core"] = _package

    app = import_module("cosmic_core.apis.cores.apis").app