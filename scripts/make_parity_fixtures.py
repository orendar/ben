"""Record Keras reference inputs/outputs for the torch parity tests.

Run once, in an environment that still has TensorFlow:

    python scripts/make_parity_fixtures.py

The fixtures are what `tests/test_torch_parity.py` compares against, so that
test needs neither TensorFlow nor the 241 MB of `.keras` models.
"""

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf  # noqa: E402
from tensorflow.keras.models import load_model  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The models the benchmark configs actually load. Others convert the same way.
MODELS = [
    "BEN-21GF-8730_2025-04-18-E30",
    "BEN-21GF-Info-8730_2025-04-18-E30",
    "Contract_2024-12-09-E50",
    "Tricks_2024-12-09-E50",
    "Lead-NT_2024-11-04-E200",
    "Lead-Suit_2024-11-04-E200",
    "SD_2024-07-08-E20",
    "RPDD_2024-07-08-E02",
    "lefty_nt_2024-07-08-E20",
    "dummy_nt_2024-07-08-E20",
    "righty_nt_2024-07-16-E20",
    "decl_nt_2024-07-08-E20",
    "lefty_suit_2024-07-08-E20",
    "dummy_suit_2024-07-08-E20",
    "righty_suit_2024-07-16-E20",
    "decl_suit_2024-07-08-E20",
]

BATCH, STEPS = 16, 8


def sample(shape, rng):
    """Half Gaussian rows, half Bernoulli — BEN's real features are mostly binary."""
    dims = [BATCH if d is None else d for d in shape]
    if len(dims) == 3 and shape[1] is None:
        dims[1] = STEPS
    x = rng.standard_normal(dims).astype(np.float32)
    x[: BATCH // 2] = (rng.random(dims[1:]) < 0.3).astype(np.float32)[None].repeat(BATCH // 2, 0)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.path.join(BASE, "models", "TF2models"))
    parser.add_argument("--dst", default=os.path.join(BASE, "tests", "fixtures"))
    parsed = parser.parse_args()
    os.makedirs(parsed.dst, exist_ok=True)

    rng = np.random.default_rng(20260901)
    for name in MODELS:
        path = os.path.join(parsed.src, name + ".keras")
        if not os.path.exists(path):
            print(f"  skip {name}: not found")
            continue
        model = load_model(path, compile=False)
        inputs = [sample(t.shape, rng) for t in model.inputs]
        dtype = model.inputs[0].dtype
        tensors = [tf.cast(tf.constant(x), dtype) for x in inputs]
        result = model(tensors[0] if len(tensors) == 1 else tensors, training=False)
        outputs = [np.asarray(r) for r in (result if isinstance(result, (list, tuple)) else [result])]

        payload = {f"in{i}": x for i, x in enumerate(inputs)}
        payload.update({f"out{i}": y for i, y in enumerate(outputs)})
        np.savez_compressed(os.path.join(parsed.dst, name + ".npz"), **payload)
        shapes = " ".join(str(y.shape) for y in outputs)
        print(f"  ok   {name}: {len(inputs)} in -> {len(outputs)} out {shapes} dtype={dtype}")

    print(f"\nfixtures written to {parsed.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
