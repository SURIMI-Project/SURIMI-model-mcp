"""Holds `APP_ROOT`, the directory above the package, where the server finds files that sit beside it."""
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
