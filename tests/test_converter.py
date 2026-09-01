"""Converter tests that carry their own model.

The parity fixtures cover the real networks but need `models/torch/*.pt`; this
builds a miniature `.keras` archive in-process, so the layer semantics that are
easy to get wrong - LSTM gate order, BatchNormalization at inference, the
float16 input cast - are checked on every run.
"""

import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

h5py = pytest.importorskip("h5py")

import torch  # noqa: E402

from convert_keras_to_torch import build  # noqa: E402
from nn.torch_graph import KerasGraph  # noqa: E402

UNITS, FEATURES, OUT = 4, 5, 3


def keras_tensor(name, shape, dtype):
    return {
        "class_name": "__keras_tensor__",
        "config": {"shape": shape, "dtype": dtype, "keras_history": [name, 0, 0]},
    }


def layer(class_name, name, config, inputs=None):
    node = {"class_name": class_name, "name": name, "config": dict(config, name=name)}
    node["inbound_nodes"] = [{"args": [inputs], "kwargs": {}}] if inputs else []
    return node


def make_archive(path, rng):
    """A float16 input -> LSTM -> BatchNormalization -> Dense(softmax) model."""
    layers = [
        layer("InputLayer", "input_layer", {"batch_shape": [None, None, FEATURES], "dtype": "float16"}),
        layer("Cast", "cast", {"dtype": "float32"},
              keras_tensor("input_layer", [None, None, FEATURES], "float16")),
        layer("LSTM", "lstm", {"units": UNITS, "return_sequences": True, "dtype": "float32"},
              keras_tensor("cast", [None, None, FEATURES], "float32")),
        layer("BatchNormalization", "batch_normalization", {"axis": -1, "epsilon": 1e-3, "dtype": "float32"},
              keras_tensor("lstm", [None, None, UNITS], "float32")),
        layer("Dense", "out", {"units": OUT, "activation": "softmax", "use_bias": True, "dtype": "float32"},
              keras_tensor("batch_normalization", [None, None, UNITS], "float32")),
    ]
    config = {
        "class_name": "Functional",
        "config": {
            "layers": layers,
            "input_layers": [["input_layer", 0, 0]],
            "output_layers": [["out", 0, 0]],
        },
    }

    weights = {
        "kernel": rng.standard_normal((FEATURES, 4 * UNITS)).astype(np.float32),
        "recurrent": rng.standard_normal((UNITS, 4 * UNITS)).astype(np.float32),
        "lstm_bias": rng.standard_normal(4 * UNITS).astype(np.float32),
        "gamma": rng.random(UNITS).astype(np.float32) + 0.5,
        "beta": rng.standard_normal(UNITS).astype(np.float32),
        "mean": rng.standard_normal(UNITS).astype(np.float32),
        "var": rng.random(UNITS).astype(np.float32) + 0.5,
        "dense_w": rng.standard_normal((UNITS, OUT)).astype(np.float32),
        "dense_b": rng.standard_normal(OUT).astype(np.float32),
    }

    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as f:
        cell = f.create_group("/layers/lstm/cell/vars")
        cell["0"], cell["1"], cell["2"] = weights["kernel"], weights["recurrent"], weights["lstm_bias"]
        bn = f.create_group("/layers/batch_normalization/vars")
        bn["0"], bn["1"] = weights["gamma"], weights["beta"]
        bn["2"], bn["3"] = weights["mean"], weights["var"]
        dense = f.create_group("/layers/dense/vars")
        dense["0"], dense["1"] = weights["dense_w"], weights["dense_b"]

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", json.dumps(config))
        z.writestr("model.weights.h5", buffer.getvalue())
    return weights


def reference(x, w, quantize=True):
    """Keras semantics by hand: gates i,f,c,o; BatchNormalization in inference."""
    x = x.astype(np.float16).astype(np.float32) if quantize else x.astype(np.float32)
    batch, steps, _ = x.shape
    h = np.zeros((batch, UNITS), np.float32)
    c = np.zeros((batch, UNITS), np.float32)
    seq = []
    sigmoid = lambda v: 1.0 / (1.0 + np.exp(-v))
    for t in range(steps):
        z = x[:, t] @ w["kernel"] + h @ w["recurrent"] + w["lstm_bias"]
        i, f, g, o = np.split(z, 4, axis=-1)
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
        h = sigmoid(o) * np.tanh(c)
        seq.append(h)
    y = np.stack(seq, axis=1)
    y = (y - w["mean"]) / np.sqrt(w["var"] + 1e-3) * w["gamma"] + w["beta"]
    logits = y @ w["dense_w"] + w["dense_b"]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    rng = np.random.default_rng(0)
    path = tmp_path_factory.mktemp("keras") / "mini.keras"
    weights = make_archive(str(path), rng)
    spec, state = build(str(path))
    graph = KerasGraph(spec)
    graph.load_state_dict(state)
    graph.eval()
    return graph, weights, rng


def test_matches_hand_computed_keras(converted):
    graph, weights, rng = converted
    x = rng.standard_normal((3, 6, FEATURES)).astype(np.float32)
    with torch.no_grad():
        got = graph(torch.from_numpy(x))[0].numpy()
    assert np.max(np.abs(got - reference(x, weights))) < 1e-5


def test_input_is_quantized_to_float16(converted):
    """The Cast chain is load-bearing: skipping it moves the answers."""
    graph, weights, rng = converted
    x = rng.standard_normal((2, 4, FEATURES)).astype(np.float32)
    with torch.no_grad():
        got = graph(torch.from_numpy(x))[0].numpy()
    assert np.max(np.abs(got - reference(x, weights, quantize=True))) < 1e-5
    assert np.max(np.abs(got - reference(x, weights, quantize=False))) > 1e-7


def test_spec_records_the_graph(converted):
    graph, _, _ = converted
    ops = [n["op"] for n in graph.spec["nodes"]]
    assert ops == ["input", "cast", "lstm", "batchnorm", "dense"]
    assert graph.spec["inputs"] == ["input_layer"]
    assert graph.spec["outputs"] == ["out"]


def test_rejects_unsupported_layer(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "bad.keras"
    make_archive(str(path), rng)
    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("config.json"))
        weights = z.read("model.weights.h5")
    config["config"]["layers"][2]["class_name"] = "GRU"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("config.json", json.dumps(config))
        z.writestr("model.weights.h5", weights)
    with pytest.raises(NotImplementedError):
        build(str(path))
