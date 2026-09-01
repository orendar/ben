"""Torch-backed Leader - API compatible with leader_tf2.Leader."""

from nn.timing import ModelTimer
from nn.torch_config import create_model, run


class Leader:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = create_model(model_path)

    def pred_fun(self, x, b):
        """x: [batch, 42], b: [batch, 15] -> card probabilities [batch, 32]."""
        with ModelTimer.time_call('leader'):
            return run(self.model, x, b)[0]


Leader.pred_fun_tf = Leader.pred_fun
