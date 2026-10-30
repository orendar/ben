"""One GPU process per card serving every gameapi on it, so a server carries no CUDA or torch.

usage: CUDA_VISIBLE_DEVICES=N python nn/gpu_broker.py SOCKET_PATH

Frames, little-endian. Request: op u32, then for OP_RUN model u32, n u32, then n arrays.
Array: ndim u32, shape u64[ndim], dtype 8s, raw C-order bytes. Reply: status u32, then
a length-prefixed message on failure, or the op's result.
"""

import os
import socket
import struct
import sys
import threading
import traceback

OP_INFO, OP_LOAD, OP_RUN = 0, 1, 2


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:])
        if k == 0:
            raise ConnectionError("broker peer closed")
        got += k
    return buf


def recv_struct(sock, fmt):
    return struct.unpack(fmt, recv_exact(sock, struct.calcsize(fmt)))


def send_array(sock, a):
    sock.sendall(struct.pack(f"<I{a.ndim}Q8s", a.ndim, *a.shape, a.dtype.str.encode()))
    sock.sendall(memoryview(a).cast("B"))


def recv_array(sock):
    import numpy as np

    ndim, = recv_struct(sock, "<I")
    *shape, dtype = recv_struct(sock, f"<{ndim}Q8s")
    dt = np.dtype(dtype.rstrip(b"\0").decode())
    return np.frombuffer(recv_exact(sock, int(np.prod(shape, dtype=np.int64)) * dt.itemsize), dtype=dt).reshape(shape)


def send_blob(sock, blob):
    sock.sendall(struct.pack("<I", len(blob)) + blob)


def recv_blob(sock):
    return bytes(recv_exact(sock, recv_struct(sock, "<I")[0]))


def serve(path):
    import json

    import torch

    from nn.torch_config import _local_create_model, _local_run

    graphs, ids = [], {}
    load_lock = threading.Lock()

    def load(model_path):
        with load_lock:
            if model_path not in ids:
                graphs.append(_local_create_model(model_path))
                ids[model_path] = len(graphs) - 1
            return ids[model_path]

    def handle(conn):
        with conn:
            while True:
                try:
                    op, = recv_struct(conn, "<I")
                except ConnectionError:
                    return
                try:
                    if op == OP_RUN:
                        model, n = recv_struct(conn, "<II")
                        arrays = [recv_array(conn) for _ in range(n)]
                        out = _local_run(graphs[model], *arrays)
                        conn.sendall(struct.pack("<II", 0, len(out)))
                        for a in out:
                            send_array(conn, a)
                    elif op == OP_LOAD:
                        model = load(recv_blob(conn).decode())
                        conn.sendall(struct.pack("<II", 0, model))
                    elif op == OP_INFO:
                        info = {
                            "torch": torch.__version__,
                            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                        }
                        conn.sendall(struct.pack("<I", 0))
                        send_blob(conn, json.dumps(info).encode())
                    else:
                        raise ValueError(f"unknown broker op {op}")
                except ConnectionError:
                    return
                except Exception:
                    conn.sendall(struct.pack("<I", 1))
                    send_blob(conn, traceback.format_exc().encode())

    if os.path.exists(path):
        os.unlink(path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(64)
    print(f"gpu broker listening on {path}", flush=True)
    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    serve(sys.argv[1])
