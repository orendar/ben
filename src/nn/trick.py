"""Trick: BEN's trick network."""

from nn.timing import ModelTimer
from nn.torch_config import create_model, run


class Trick:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = create_model(model_path)

    def pred_fun(self, x):
        """x: [batch, 55] -> trick distribution."""
        with ModelTimer.time_call('trick'):
            return run(self.model, x)[0]
