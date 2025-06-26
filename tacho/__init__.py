"""CLI tool for measuring and comparing LLM inference speeds."""
from importlib.metadata import version

try:
    __version__ = version("tacho")
except Exception:
    __version__ = "dev"

__all__ = ["__version__"]