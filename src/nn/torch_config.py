"""Shared torch settings for all models.

A gameapi pool runs one server per core, so each process must stay
single-threaded; letting torch size its own pool reproduces the TensorFlow
oversubscription that cost 11x (docs: "Pin TF's thread pools").
"""

import os

import numpy as np
import torch

from nn.torch_graph import load_graph

THREADS = int(os.environ.get("BEN_TORCH_THREADS", "1"))
_configured = False


def _configure():
    global _configured
    if _configured:
        return
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(THREADS)
    except RuntimeError:
        pass  # already fixed by an earlier parallel region; the intra-op cap is what matters
    torch.set_grad_enabled(False)
    _configured = True


def create_model(model_path):
    _configure()
    return load_graph(model_path)


def run(graph, *arrays):
    """numpy in, numpy out - the wrappers' callers know nothing about torch."""
    tensors = [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in arrays]
    with torch.no_grad():
        return [t.numpy() for t in graph(*tensors)]
