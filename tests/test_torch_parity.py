"""Every converted model must reproduce Keras's answers.

The references in `tests/fixtures/` are frozen recordings of the answers the
trained weights shipped with; nothing in the repo regenerates them.
"""

import os

import numpy as np
import pytest
import torch

from conftest import FIXTURES, TORCH_MODELS
from nn.torch_graph import load_graph

# Relative to each output's peak, because the bidinfo shape head predicts suit
# lengths (0-13) while every other head is a probability. Measured worst: 5.0e-6
# on CPU, 5.7e-5 on CUDA, where cuDNN's fused recurrence accumulates differently.
# A wrong LSTM gate order or a missed BatchNormalization lands orders over both.
TOLERANCE = 1e-4
CUDA_TOLERANCE = 5e-4

NAMES = sorted(f[: -len(".npz")] for f in os.listdir(FIXTURES) if f.endswith(".npz"))


def load_fixture(name):
    data = np.load(os.path.join(FIXTURES, name + ".npz"))
    inputs = [data[k] for k in sorted(data) if k.startswith("in")]
    outputs = [data[k] for k in sorted(data) if k.startswith("out")]
    return inputs, outputs


@pytest.fixture(scope="module", params=NAMES)
def model_case(request):
    path = os.path.join(TORCH_MODELS, request.param + ".pt")
    if not os.path.exists(path):
        pytest.skip(f"{path} missing - run scripts/convert_keras_to_torch.py")
    inputs, outputs = load_fixture(request.param)
    return request.param, load_graph(path), inputs, outputs


def run(graph, inputs):
    with torch.no_grad():
        return graph(*[torch.from_numpy(x) for x in inputs])


def relative(got, expected):
    return np.max(np.abs(got - expected)) / max(float(np.max(np.abs(expected))), 1e-9)


def test_matches_keras_reference(model_case):
    name, graph, inputs, expected = model_case
    got = run(graph, inputs)
    assert len(got) == len(expected), f"{name}: {len(got)} outputs, expected {len(expected)}"
    for i, (a, b) in enumerate(zip(got, expected)):
        a = a.numpy()
        assert a.shape == b.shape, f"{name} out{i}: {a.shape} != {b.shape}"
        diff = relative(a, b)
        assert diff < TOLERANCE, f"{name} out{i}: relative diff {diff:.3g}"


def test_probabilities_stay_normalized(model_case):
    """A softmax head that lost its activation still looks plausible elementwise."""
    name, graph, inputs, expected = model_case
    for i, (got, ref) in enumerate(zip(run(graph, inputs), expected)):
        if not np.allclose(ref.sum(axis=-1), 1.0, atol=1e-3):
            continue
        sums = got.numpy().sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=1e-3), f"{name} out{i}: sums {sums.min()}..{sums.max()}"


def test_deterministic(model_case):
    name, graph, inputs, _ = model_case
    first = [t.numpy() for t in run(graph, inputs)]
    second = [t.numpy() for t in run(graph, inputs)]
    for i, (a, b) in enumerate(zip(first, second)):
        assert np.array_equal(a, b), f"{name} out{i}: repeated call differs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_cuda_matches_keras_reference(model_case):
    """BEN_TORCH_DEVICE=cuda must not cost accuracy - cudnn.allow_tf32 would."""
    name, graph, inputs, expected = model_case
    matmul_tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        graph.to("cuda")
        with torch.no_grad():
            got = graph(*[torch.from_numpy(x).to("cuda") for x in inputs])
        for i, (a, b) in enumerate(zip(got, expected)):
            diff = relative(a.cpu().numpy(), b)
            assert diff < CUDA_TOLERANCE, f"{name} out{i}: relative diff {diff:.3g} on cuda"
    finally:
        graph.to("cpu")
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def test_rows_are_independent(model_case):
    """Dropout left on, or a batch-statistics BatchNormalization, breaks this."""
    name, graph, inputs, _ = model_case
    batched = [t.numpy() for t in run(graph, inputs)]
    single = [t.numpy() for t in run(graph, [x[:1] for x in inputs])]
    for i, (whole, row) in enumerate(zip(batched, single)):
        diff = relative(whole[:1], row)
        assert diff < TOLERANCE, f"{name} out{i}: row 0 depends on the batch ({diff:.3g})"
