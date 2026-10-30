"""Shared torch settings for all models.

A gameapi pool runs one server per core, so on CPU each process must stay
single-threaded; letting torch size its own pool oversubscribes every core.

Inference runs on the GPU whenever there is one, which is worth ~12x on the real
call mix because the bidder is called on batches of hundreds to thousands of
sampled auctions, not one at a time. Set `BEN_TORCH_DEVICE=cpu` to force it off:
a 16-server pool holds ~11.8 GB of VRAM, so it will contend with anything else
training on the same card.

Set `BEN_GPU_BROKER=SOCKET` to send inference to `nn/gpu_broker.py` instead, and
this process imports neither torch nor CUDA.
"""

import json
import os
import threading
import time

import numpy as np

BROKER = os.environ.get("BEN_GPU_BROKER")
THREADS = int(os.environ.get("BEN_TORCH_THREADS", "1"))
_configured = False


def _configure():
    global _configured
    import torch

    if _configured:
        return
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(THREADS)
    except RuntimeError:
        pass  # already fixed by an earlier parallel region; the intra-op cap is what matters
    torch.set_grad_enabled(False)
    if _device().startswith("cuda"):
        # cudnn.allow_tf32 defaults to True and would silently drop the LSTM to
        # ~1e-3 accuracy. Measured worth: 5% on the largest call. Not a trade.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    _configured = True


def _device():
    import torch

    return os.environ.get("BEN_TORCH_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def _local_create_model(model_path):
    from nn.torch_graph import load_graph

    _configure()
    return load_graph(model_path).to(_device())


def _local_run(graph, *arrays):
    import torch

    tensors = [
        torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(_device())
        for a in arrays
    ]
    with torch.no_grad():
        return [t.cpu().numpy() for t in graph(*tensors)]


class _Broker:
    """One blocking connection: a server answers one request at a time, as it did with local CUDA."""

    def __init__(self, path):
        import _socket  # gevent patches socket.socket, and a yield here could interleave frames

        self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        deadline = time.monotonic() + 120  # the broker imports torch, and may start after us
        while True:
            try:
                self.sock.connect(path)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.5)
        self.lock = threading.Lock()

    def _status(self):
        from nn.gpu_broker import recv_blob, recv_struct

        if recv_struct(self.sock, "<I")[0] != 0:
            raise RuntimeError("gpu broker: " + recv_blob(self.sock).decode())

    def info(self):
        import struct

        from nn.gpu_broker import OP_INFO, recv_blob

        with self.lock:
            self.sock.sendall(struct.pack("<I", OP_INFO))
            self._status()
            return json.loads(recv_blob(self.sock))

    def load(self, model_path):
        import struct

        from nn.gpu_broker import OP_LOAD, recv_struct, send_blob

        with self.lock:
            self.sock.sendall(struct.pack("<I", OP_LOAD))
            send_blob(self.sock, os.path.abspath(model_path).encode())
            self._status()
            return recv_struct(self.sock, "<I")[0]

    def run(self, model, arrays):
        import struct

        from nn.gpu_broker import OP_RUN, recv_array, recv_struct, send_array

        with self.lock:
            self.sock.sendall(struct.pack("<III", OP_RUN, model, len(arrays)))
            for a in arrays:
                send_array(self.sock, a)
            self._status()
            n = recv_struct(self.sock, "<I")[0]
            return [recv_array(self.sock) for _ in range(n)]


_broker = _Broker(BROKER) if BROKER else None


def create_model(model_path):
    return _broker.load(model_path) if _broker else _local_create_model(model_path)


def run(graph, *arrays):
    """numpy in, numpy out - the wrappers' callers know nothing about torch."""
    if _broker:
        return _broker.run(graph, [np.ascontiguousarray(a, dtype=np.float32) for a in arrays])
    return _local_run(graph, *arrays)


def runtime():
    """The torch version and device that answer inference, for the deployment manifest."""
    if _broker:
        return _broker.info()
    import torch

    return {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def set_seed(seed):
    # Nothing in a graph draws randomness, so a broker client has nothing to seed.
    if not _broker:
        import torch

        torch.manual_seed(seed)
