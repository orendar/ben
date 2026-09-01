"""Torch runtime for the converted Keras models.

A `.pt` produced by `scripts/convert_keras_to_torch.py` holds a JSON graph spec
plus a state dict; nothing here reads Keras, HDF5 or TensorFlow.
"""

import json

import torch
from torch import nn

ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": torch.relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "softmax": lambda x: torch.softmax(x, dim=-1),
    "elu": torch.nn.functional.elu,
    "selu": torch.selu,
}

DTYPES = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}


class LastAxisBatchNorm(nn.Module):
    """Keras BatchNormalization(axis=-1) at inference, for any rank of input.

    torch's BatchNorm1d normalizes axis 1, so it cannot serve a (batch, time,
    features) tensor without a transpose; the affine form is exact and cheaper.
    """

    def __init__(self, features, epsilon):
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("gamma", torch.ones(features))
        self.register_buffer("beta", torch.zeros(features))
        self.register_buffer("moving_mean", torch.zeros(features))
        self.register_buffer("moving_var", torch.ones(features))

    def forward(self, x):
        scale = self.gamma * torch.rsqrt(self.moving_var + self.epsilon)
        return (x - self.moving_mean) * scale + self.beta


class DenseLayer(nn.Module):
    def __init__(self, in_features, units, activation, use_bias):
        super().__init__()
        self.linear = nn.Linear(in_features, units, bias=use_bias)
        self.activation = activation

    def forward(self, x):
        return ACTIVATIONS[self.activation](self.linear(x))


class LSTMLayer(nn.Module):
    def __init__(self, in_features, units, return_sequences):
        super().__init__()
        self.lstm = nn.LSTM(in_features, units, batch_first=True)
        self.return_sequences = return_sequences

    def forward(self, x):
        out, _ = self.lstm(x)
        return out if self.return_sequences else out[:, -1, :]


class KerasGraph(nn.Module):
    """Executes a converted Keras functional graph."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.input_names = spec["inputs"]
        self.output_names = spec["outputs"]
        self.nodes = spec["nodes"]
        layers = {}
        for node in self.nodes:
            op, args = node["op"], node["args"]
            if op == "dense":
                layers[node["name"]] = DenseLayer(
                    args["in_features"], args["units"], args["activation"], args["use_bias"]
                )
            elif op == "lstm":
                layers[node["name"]] = LSTMLayer(
                    args["in_features"], args["units"], args["return_sequences"]
                )
            elif op == "batchnorm":
                layers[node["name"]] = LastAxisBatchNorm(args["features"], args["epsilon"])
        self.layers = nn.ModuleDict(layers)
        self.eval()

    def forward(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(f"expected {len(self.input_names)} inputs, got {len(inputs)}")
        values = dict(zip(self.input_names, inputs))
        for node in self.nodes:
            op, name, args = node["op"], node["name"], node["args"]
            if op == "input":
                # Keras declares the input dtype on the layer; a float16 input
                # layer quantizes before the graph's Cast back to float32, and
                # dropping that changes the model's answers.
                values[name] = values[name].to(DTYPES[args["dtype"]])
                continue
            src = [values[i] for i in node["inputs"]]
            if "compute" in args:
                src = [s.to(DTYPES[args["compute"]]) for s in src]
            if op == "cast":
                values[name] = src[0].to(DTYPES[args["dtype"]])
            elif op == "concat":
                values[name] = torch.cat(src, dim=args["axis"])
            elif op == "softmax":
                values[name] = torch.softmax(src[0], dim=args["axis"])
            elif op == "activation":
                values[name] = ACTIVATIONS[args["activation"]](src[0])
            elif op == "identity":  # Dropout, and other inference no-ops
                values[name] = src[0]
            else:
                values[name] = self.layers[name](src[0])
        return tuple(values[o] for o in self.output_names)


def load_graph(path, threads=None):
    blob = torch.load(path, map_location="cpu", weights_only=True)
    graph = KerasGraph(json.loads(blob["spec"]))
    graph.load_state_dict(blob["state"])
    graph.eval()
    for p in graph.parameters():
        p.requires_grad_(False)
    if threads is not None:
        torch.set_num_threads(threads)
    return graph
