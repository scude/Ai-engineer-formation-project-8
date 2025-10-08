"""Gunicorn entrypoint exposing the Flask application instance."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
_PACKAGE_NAME = "_segmentation_app"
_PACKAGE_DIR = Path(__file__).resolve().parent / "app"

_spec = importlib.util.spec_from_file_location(
    _PACKAGE_NAME,
    _PACKAGE_DIR / "__init__.py",
    submodule_search_locations=[str(_PACKAGE_DIR)],
)
if _spec is None or _spec.loader is None:
    raise ImportError("Unable to load the application package for Gunicorn.")

_package = importlib.util.module_from_spec(_spec)
sys.modules[_PACKAGE_NAME] = _package
_spec.loader.exec_module(_package)

create_app = getattr(_package, "create_app")

current_module = sys.modules[__name__]

for name, module in list(sys.modules.items()):
    if not name.startswith(f"{_PACKAGE_NAME}."):
        continue
    alias = name.replace(_PACKAGE_NAME, __name__, 1)
    sys.modules.setdefault(alias, module)
    parts = alias.split(".", 1)
    if len(parts) == 2 and "." not in parts[1]:
        setattr(current_module, parts[1], module)

config_module_name = f"{__name__}.config"
config_module = sys.modules.get(config_module_name)
if config_module is None:
    raise ImportError("Application configuration module could not be loaded.")

setattr(sys.modules[__name__], "config", config_module)

MAX_CONTENT_LENGTH = getattr(config_module, "MAX_CONTENT_LENGTH")

app = create_app({"MAX_CONTENT_LENGTH": MAX_CONTENT_LENGTH})

__all__ = ["app", "create_app", "MAX_CONTENT_LENGTH"]