"""Version reporting and seeding for the inference backend."""

from importlib import metadata

import torch

PACKAGES = ("torch", "numpy")


def set_seed(seed):
    torch.manual_seed(seed)


def banner():
    found = []
    for name in PACKAGES:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            continue
    return "Backends: " + ", ".join(found)
