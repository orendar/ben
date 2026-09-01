"""Backend-agnostic seeding and version reporting.

The entry points used to `import tensorflow` only to print a banner and seed an
RNG, which made TensorFlow a hard dependency of every deployment including ones
that load no Keras model at all.
"""

import sys
from importlib import metadata

PACKAGES = ("torch", "tensorflow", "tensorflow-intel", "onnxruntime", "keras", "numpy")


def set_seed(seed):
    """Seed the backends that are already imported.

    Deliberately does not import anything: a torch deployment must not pull in
    TensorFlow just to honour a seed, and inference is deterministic regardless.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None:
        tensorflow.random.set_seed(seed)


def banner():
    found = []
    for name in PACKAGES:
        try:
            found.append(f"{name} {metadata.version(name)}")
        except metadata.PackageNotFoundError:
            continue
    return "Backends: " + (", ".join(found) if found else "none found")
