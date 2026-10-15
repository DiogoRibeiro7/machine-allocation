"""Machine allocation on unrelated parallel machines: instance, model, baselines, simulation."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("machine-allocation")
except PackageNotFoundError:          # imported from a source tree that was never installed
    __version__ = "0+unknown"
