# Models

Every network runs on PyTorch. There is one backend and one model format: a
`.pt` holding a JSON graph spec plus a state dict, loaded by `nn/torch_graph.py`.

## Regenerating the weights

`models/torch/*.pt` is gitignored and produced from the Keras archives BEN was
trained with:

```bash
python scripts/convert_keras_to_torch.py     # <weights>/*.keras -> models/torch/*.pt
```

The converter reads the `.keras` zip with `zipfile` and `h5py` — it does not
import Keras or TensorFlow, and neither does anything else in this repo. Point
`--src` at a directory of `.keras` files to bring in new upstream weights.

Two things it has to get right, both covered by `tests/test_converter.py`, which
builds its own miniature archive so the check needs no large models: Keras packs
LSTM gates as i,f,c,o — torch's i,f,g,o in the same order, so the conversion is a
transpose plus a zeroed second bias — and a `float16` input layer quantizes the
input before the graph's cast back to `float32`.

Keras also stores weights under an auto-generated `snake_case(class)_N` key in
layer order rather than the layer's own name, so `contract_output` is saved as
`dense_2`; resolving by name silently misses.

## Fidelity

`tests/test_torch_parity.py` checks all 16 networks against references recorded
from the original Keras models, on CPU and GPU. Worst relative error is 5.0e-6
on CPU and 5.7e-5 on GPU, where cuDNN's fused recurrence accumulates
differently. Tolerances are relative to each output's peak because the bidinfo
shape head predicts suit lengths while every other head is a probability.

Those references are frozen artifacts: they pin the answers the trained weights
were shipped with, and nothing in the repo can regenerate them.

Round-off is not free at the top level. The `[bidding]` thresholds are hard
cutoffs on these probabilities, so it flips a candidate set discretely and the
search then diverges — measured against a cache of the original engine's
answers, 2 changed bids in 2,000 on CPU and 6 on GPU. **A response cache is
valid for one build on one device.**

## Device and threads

Inference uses the GPU when one is present; `BEN_TORCH_DEVICE=cpu` forces it
off. On an RTX 5090 that is 8.5x on inference and takes a bidding board from
100.9 s to 79.8 s against the original engine, and a 16-server pool goes from
10.12 to 14.74 bids/s. It costs ~740 MB of VRAM per server process, so 16
servers hold ~11.8 GB and will contend with anything else on the card.

`BEN_TORCH_THREADS` (default 1) caps the intra-op pool. Leave it at 1 for a
server pool: one server handles one request at a time, so server count is the
concurrency dial, not threads.

## Do not spend more effort on inference

Measured on the real call shapes with `scripts/bench_bidder.py`:
`torch.compile` with `max-autotune` is within noise of eager (4.42 vs 4.45 ms on
the largest call) because cuDNN's fused LSTM already wins; TF32 buys 5% for 30x
the error; bfloat16 is not faster at these sizes. Inference is now 3.3% of a
board and double-dummy solving is 77.7%, so the whole remaining prize is 1.03x.
The only lever with real headroom is issuing fewer DD solves
(`sample_hands_auction`, the search thresholds), which changes how the engine
plays.
