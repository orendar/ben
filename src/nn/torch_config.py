"""Shared torch settings for all models.

A gameapi pool runs one server per core, so on CPU each process must stay
single-threaded; letting torch size its own pool oversubscribes every core.

Inference runs on the GPU whenever there is one, which is worth ~12x on the real
call mix because the bidder is called on batches of hundreds to thousands of
sampled auctions, not one at a time. Set `BEN_TORCH_DEVICE=cpu` to force it off:
a 16-server pool holds ~11.8 GB of VRAM, so it will contend with anything else
training on the same card.
"""

import os

import numpy as np
import torch

from nn.torch_graph import load_graph

THREADS = int(os.environ.get("BEN_TORCH_THREADS", "1"))
DEVICE = os.environ.get("BEN_TORCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
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
    if DEVICE.startswith("cuda"):
        # cudnn.allow_tf32 defaults to True and would silently drop the LSTM to
        # ~1e-3 accuracy. Measured worth: 5% on the largest call. Not a trade.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _configured = True


def create_model(model_path):
    _configure()
    return load_graph(model_path).to(DEVICE)


def run(graph, *arrays):
    """numpy in, numpy out - the wrappers' callers know nothing about torch."""
    tensors = [
        torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(DEVICE)
        for a in arrays
    ]
    with torch.no_grad():
        return [t.cpu().numpy() for t in graph(*tensors)]
