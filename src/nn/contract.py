"""Contract: BEN's contract network."""

from nn.timing import ModelTimer
from nn.torch_config import create_model, run


class Contract:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = create_model(model_path)

    def pred_fun(self, x):
        """x: [batch, 50] -> contract probabilities."""
        with ModelTimer.time_call('contract'):
            return run(self.model, x)[0]
