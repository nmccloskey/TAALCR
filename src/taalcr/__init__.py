from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("taalcr")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"