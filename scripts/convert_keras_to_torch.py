"""Convert BEN's Keras models to the torch format `nn/torch_graph.py` loads.

Reads `.keras` archives directly (zip + HDF5), so TensorFlow is never imported
and never needs to be installed. Optimizer state is dropped, which is why the
outputs are roughly a third the size of the inputs.

    python scripts/convert_keras_to_torch.py                 # all of models/TF2models
    python scripts/convert_keras_to_torch.py --only BEN-21GF # substring filter
"""

import argparse
import io
import json
import os
import sys
import zipfile

import h5py
import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keras 3 stores weights under an auto-generated `snake_case(class)_N` key in
# layer order, NOT under the layer's own name, and omits Cast layers entirely.
# So `contract_output` is saved as `dense_2`; resolving by name silently misses.
H5_BASENAME = {
    "InputLayer": "input_layer",
    "Dense": "dense",
    "LSTM": "lstm",
    "BatchNormalization": "batch_normalization",
    "TimeDistributed": "time_distributed",
    "Dropout": "dropout",
    "Concatenate": "concatenate",
    "Softmax": "softmax",
    "Activation": "activation",
}


def h5_keys(layers):
    """Map each layer name to the key Keras saved its weights under."""
    keys, counts = {}, {}
    for layer in layers:
        cls = layer["class_name"]
        if cls == "Cast":
            continue
        if cls not in H5_BASENAME:
            raise NotImplementedError(f"no weight-key rule for layer class {cls}")
        base = H5_BASENAME[cls]
        n = counts.get(cls, 0)
        counts[cls] = n + 1
        keys[layer["name"]] = base if n == 0 else f"{base}_{n}"
    return keys


def _history(arg):
    """Source layer name(s) of one entry in a Keras 3 `inbound_nodes` arg list."""
    if isinstance(arg, list):
        return [h for a in arg for h in _history(a)]
    if isinstance(arg, dict) and arg.get("class_name") == "__keras_tensor__":
        return [arg["config"]["keras_history"][0]]
    return []


def _inputs_of(layer):
    nodes = layer.get("inbound_nodes") or []
    if not nodes:
        return []
    return [n for a in nodes[0].get("args", []) for n in _history(a)]


def _dtype_name(cfg, default="float32"):
    dtype = cfg.get("dtype", default)
    if isinstance(dtype, dict):
        return dtype["config"]["name"]
    return dtype or default


def _vars(weights, path):
    group = weights[path]
    return [np.asarray(group[str(i)]) for i in range(len(group))]


def build(keras_path):
    archive = zipfile.ZipFile(keras_path)
    config = json.loads(archive.read("config.json"))["config"]
    weights = h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r")

    nodes, state = [], {}
    keys = h5_keys(config["layers"])
    for layer in config["layers"]:
        cls, cfg = layer["class_name"], layer["config"]
        name = layer["name"]
        key = keys.get(name)
        srcs = _inputs_of(layer)
        args = {}

        if cls == "InputLayer":
            op, args = "input", {"dtype": _dtype_name(cfg), "shape": cfg["batch_shape"]}
        elif cls == "Cast":
            op, args = "cast", {"dtype": _dtype_name(cfg)}
        elif cls == "Dropout":
            op = "identity"
        elif cls == "Concatenate":
            op, args = "concat", {"axis": cfg.get("axis", -1)}
        elif cls == "Softmax":
            op, args = "softmax", {"axis": cfg.get("axis", -1)}
        elif cls == "Activation":
            op, args = "activation", {"activation": cfg["activation"]}
        elif cls in ("Dense", "TimeDistributed"):
            inner = cfg["layer"]["config"] if cls == "TimeDistributed" else cfg
            if cls == "TimeDistributed" and cfg["layer"]["class_name"] != "Dense":
                raise NotImplementedError(f"TimeDistributed({cfg['layer']['class_name']})")
            path = f"/layers/{key}/layer/vars" if cls == "TimeDistributed" else f"/layers/{key}/vars"
            var = _vars(weights, path)
            kernel = var[0]
            op = "dense"
            args = {
                "in_features": int(kernel.shape[0]),
                "units": int(inner["units"]),
                "activation": inner.get("activation", "linear"),
                "use_bias": bool(inner.get("use_bias", True)),
            }
            if kernel.shape[1] != args["units"]:
                raise ValueError(f"{name}: weights {kernel.shape} do not match units {args['units']}")
            state[f"layers.{name}.linear.weight"] = torch.from_numpy(kernel.T.copy())
            if args["use_bias"]:
                state[f"layers.{name}.linear.bias"] = torch.from_numpy(var[1].copy())
        elif cls == "LSTM":
            kernel, recurrent, bias = _vars(weights, f"/layers/{key}/cell/vars")
            units = int(cfg["units"])
            if kernel.shape[1] != 4 * units:
                raise ValueError(f"{name}: kernel {kernel.shape} does not match 4x{units}")
            op = "lstm"
            args = {
                "in_features": int(kernel.shape[0]),
                "units": units,
                "return_sequences": bool(cfg.get("return_sequences", False)),
            }
            # Keras packs the gates i,f,c,o along axis 1; torch uses i,f,g,o in
            # the same order, so the transpose is the whole conversion. Torch
            # applies two biases and Keras one, hence the zeros.
            state[f"layers.{name}.lstm.weight_ih_l0"] = torch.from_numpy(kernel.T.copy())
            state[f"layers.{name}.lstm.weight_hh_l0"] = torch.from_numpy(recurrent.T.copy())
            state[f"layers.{name}.lstm.bias_ih_l0"] = torch.from_numpy(bias.copy())
            state[f"layers.{name}.lstm.bias_hh_l0"] = torch.zeros(4 * units)
            for unsupported in ("activation", "recurrent_activation"):
                expected = {"activation": "tanh", "recurrent_activation": "sigmoid"}[unsupported]
                if cfg.get(unsupported, expected) != expected:
                    raise NotImplementedError(f"{name}: {unsupported}={cfg[unsupported]}")
            if cfg.get("go_backwards") or cfg.get("stateful"):
                raise NotImplementedError(f"{name}: go_backwards/stateful unsupported")
        elif cls == "BatchNormalization":
            gamma, beta, mean, var = _vars(weights, f"/layers/{key}/vars")
            if cfg.get("axis", -1) not in (-1, len(gamma.shape)):
                raise NotImplementedError(f"{name}: BatchNormalization axis={cfg['axis']}")
            op = "batchnorm"
            args = {"features": int(gamma.shape[0]), "epsilon": float(cfg.get("epsilon", 1e-3))}
            state[f"layers.{name}.gamma"] = torch.from_numpy(gamma.copy())
            state[f"layers.{name}.beta"] = torch.from_numpy(beta.copy())
            state[f"layers.{name}.moving_mean"] = torch.from_numpy(mean.copy())
            state[f"layers.{name}.moving_var"] = torch.from_numpy(var.copy())
        else:
            raise NotImplementedError(f"unsupported layer {cls} ({name})")

        if op not in ("input", "cast"):
            # Keras casts to the layer's own policy dtype on entry. Most models
            # spell that out with a Cast layer after a float16 input; righty_nt
            # and righty_suit do not, and rely on this implicit cast instead.
            args = dict(args, compute=_dtype_name(cfg))
        nodes.append({"name": name, "op": op, "args": args, "inputs": srcs})

    spec = {
        "inputs": [n[0] for n in config["input_layers"]],
        "outputs": [n[0] for n in config["output_layers"]],
        "nodes": nodes,
        "source": os.path.basename(keras_path),
    }
    return spec, state


def convert(keras_path, out_path):
    spec, state = build(keras_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"spec": json.dumps(spec), "state": state}, out_path)
    return spec, os.path.getsize(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=os.path.join(BASE, "models", "TF2models"))
    parser.add_argument("--dst", default=os.path.join(BASE, "models", "torch"))
    parser.add_argument("--only", default="", help="substring filter on the filename")
    parsed = parser.parse_args()

    names = sorted(f for f in os.listdir(parsed.src) if f.endswith(".keras") and parsed.only in f)
    if not names:
        sys.exit(f"no .keras models matching {parsed.only!r} in {parsed.src}")

    failed = 0
    for name in names:
        out = os.path.join(parsed.dst, name.replace(".keras", ".pt"))
        try:
            spec, size = convert(os.path.join(parsed.src, name), out)
        except Exception as exc:  # keep going so one odd model does not stop the batch
            print(f"  FAIL {name}: {exc}")
            failed += 1
            continue
        ops = ",".join(sorted({n["op"] for n in spec["nodes"]} - {"identity", "input"}))
        print(f"  ok   {name} -> {os.path.basename(out)}  {size / 1e6:.1f} MB  [{ops}]")

    print(f"\n{len(names) - failed}/{len(names)} converted into {parsed.dst}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
