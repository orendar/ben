"""Version reporting and seeding for the inference backend."""

from importlib import metadata

from nn.torch_config import set_seed  # noqa: F401

PACKAGES = ("torch", "numpy")


def banner():
    found = []
    for name in PACKAGES:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            continue
    return "Backends: " + ", ".join(found)
