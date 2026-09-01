# Torch backend

A third inference backend alongside `_tf2` and `_onnx`, selected by file
extension: a config whose model paths end in `.pt` loads `nn/*_torch.py` and
never imports TensorFlow, Keras or onnxruntime.

## Using it

```bash
python scripts/convert_keras_to_torch.py     # models/TF2models/*.keras -> models/torch/*.pt
cd src && python gameapi.py --config config/BEN-21GF-torch.conf --port 8085
```

`models/torch/` is gitignored: the `.pt` files are a deterministic derivative of
the committed `.keras` ones, and dropping the optimizer state makes them about a
third the size. `requirements-torch.txt` is the runtime; the converter also
needs `h5py`, but a serving box never runs it.

`BEN_TORCH_THREADS` (default 1) caps torch's intra-op pool. Leave it at 1 for a
server pool — TensorFlow sizing its own pool from the host's core count, not the
cgroup's, is what once cost 11x.

## Fidelity

The converter reads the `.keras` zip with h5py, so no TensorFlow is needed to
produce a `.pt` either. Two things it must get right, both covered by tests:
Keras packs LSTM gates as i,f,c,o (torch's i,f,g,o in the same order, so the
conversion is a transpose and a zeroed second bias), and a `float16` input layer
quantizes the input before the graph's cast back to `float32`.

Agreement with Keras is 5.0e-6 worst case across the 16 benchmark models
(`tests/test_torch_parity.py`, references recorded by
`scripts/make_parity_fixtures.py`).

That is float32 round-off, but it is not free: on 2,000 replayed positions the
torch backend returned **2 different bids (0.10%)** where TensorFlow returned 0
on the identical set. The `[bidding]` thresholds are hard cutoffs on these
probabilities, so round-off flips a candidate set discretely and the search then
diverges. **Numbers measured against a torch server are not comparable to
TensorFlow-served ones** — re-baseline any cached benchmark results.

What it buys: neural-network time down 1.63x, end-to-end wall down 1.06x
(6.34 vs 5.98 bids/s over those 2,000 positions, 8 servers each). Double-dummy
solving is ~62% of a bid and inference ~22%, so the backend is a small lever.
