"""The 2024 lead models were trained with their two inputs swapped.

Keras 3.6 and older fed `{'X_input', 'B_input'}` to the concat in the wrong order,
and the weights learned that order; the converter (and Keras 3.7+) uses the declared
one, which leaves the model leading cards it does not hold. The `-v2` files restore
the training order, as upstream's `-v2` Keras files do.
"""

import json
import os

import numpy as np
import pytest
import torch

from conftest import FIXTURES, TORCH_MODELS
from nn.torch_graph import load_graph

MODELS = {"nt": "Lead-NT_2024-11-04-E200", "suit": "Lead-Suit_2024-11-04-E200"}


def held_mass(path, kind):
    data = np.load(os.path.join(FIXTURES, "lead", "real_lead_inputs.npz"))
    with torch.no_grad():
        (p,) = load_graph(path)(torch.from_numpy(data[f"{kind}_x"]), torch.from_numpy(data[f"{kind}_b"]))
    held = data[f"{kind}_held"]
    p = p.numpy()
    return float((p * held).sum(axis=1).mean()), float(held[np.arange(len(p)), p.argmax(axis=1)].mean())


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_v2_leads_cards_it_holds(kind):
    mass, top_held = held_mass(os.path.join(TORCH_MODELS, MODELS[kind] + "-v2.pt"), kind)
    assert mass > 0.95 and top_held == 1.0, (mass, top_held)


@pytest.mark.parametrize("name", ["LeadV3-NT-Both_2026-10-16", "LeadV3-Suit-Both_2026-10-16"])
def test_leadv3_leads_cards_it_holds_in_declared_order(name):
    mass, top_held = held_mass(os.path.join(TORCH_MODELS, name + ".pt"), "nt" if "NT" in name else "suit")
    assert mass > 0.95 and top_held == 1.0, (mass, top_held)


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_declared_order_does_not(kind):
    mass, top_held = held_mass(os.path.join(TORCH_MODELS, MODELS[kind] + ".pt"), kind)
    assert mass < 0.92 and top_held < 0.95, (mass, top_held)


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_v2_differs_only_in_concat_order(kind):
    base, v2 = (torch.load(os.path.join(TORCH_MODELS, MODELS[kind] + s + ".pt"), weights_only=False)
                for s in ("", "-v2"))
    for key, tensor in base["state"].items():
        assert torch.equal(tensor, v2["state"][key]), key
    a, b = json.loads(base["spec"]), json.loads(v2["spec"])
    for na, nb in zip(a["nodes"], b["nodes"]):
        if na["op"] == "concat":
            assert nb["inputs"] == na["inputs"][::-1]
        else:
            assert na == nb
