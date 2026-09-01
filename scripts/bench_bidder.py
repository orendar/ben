"""Time the bidder network at the batch shapes a real auction produces.

Shapes come from instrumenting a live server: sequence length 1-4, batch from 1
to ~5,500, and 87% of calls (99% of the time) are batched. Timing counts the
numpy->numpy round trip, transfers included, because that is what the wrapper
costs its caller.

    python scripts/bench_bidder.py --model models/torch/BEN-21GF-8730_2025-04-18-E30.pt
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from nn.torch_graph import load_graph  # noqa: E402

SHAPES = [(1, 3), (256, 3), (1215, 3), (2000, 4), (5483, 4)]
FEATURES = 193


def timed(fn, x, repeats, cuda):
    for _ in range(3):
        fn(x)
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(x)
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000, out


def make_runner(graph, device, dtype, compiled):
    model = graph.to(device)
    if compiled:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    def run(x):
        with torch.no_grad():
            t = torch.from_numpy(x).to(device, non_blocking=True)
            if dtype is not None:
                with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
                    out = model(t)
            else:
                out = model(t)
            return [o.float().cpu().numpy() for o in out]

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    inputs = {s: (rng.random((s[0], s[1], FEATURES)) < 0.15).astype(np.float32) for s in SHAPES}

    configs = [("cpu t1", "cpu", None, False, 1), ("cpu t4", "cpu", None, False, 4),
               ("cpu t16", "cpu", None, False, 16)]
    if torch.cuda.is_available():
        configs += [("cuda fp32", "cuda", None, False, 0),
                    ("cuda tf32", "cuda", None, False, 0),
                    ("cuda bf16", "cuda", torch.bfloat16, False, 0),
                    ("cuda fp32 compiled", "cuda", None, True, 0)]

    reference = {}
    print(f"{'config':<20}" + "".join(f"{f'{b}x{s}':>12}" for b, s in SHAPES) + f"{'max_diff':>11}")
    for name, device, dtype, compiled, threads in configs:
        if device == "cpu":
            torch.set_num_threads(threads)
            torch.backends.cuda.matmul.allow_tf32 = False
        else:
            tf32 = "tf32" in name
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
        graph = load_graph(args.model)
        run = make_runner(graph, device, dtype, compiled)
        cells, worst = [], 0.0
        for shape in SHAPES:
            x = inputs[shape]
            ms, out = timed(run, x, args.repeats, device == "cuda")
            cells.append(ms)
            if name == "cpu t1":
                reference[shape] = out[0]
            else:
                worst = max(worst, float(np.max(np.abs(out[0] - reference[shape]))))
        print(f"{name:<20}" + "".join(f"{c:>12.2f}" for c in cells) + f"{worst:>11.2e}")

    print("\nms per call, numpy in / numpy out. 'cpu t1' is what a 16-server pool runs today.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
