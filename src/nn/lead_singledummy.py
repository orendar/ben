"""LeadSingleDummy: BEN's single-dummy trick estimator."""

from nn.timing import ModelTimer
from nn.torch_config import create_model, run


class LeadSingleDummy:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = create_model(model_path)

    def pred_fun(self, x):
        """x: [batch, 165] (SD) or [batch, 133] (RPDD) -> tricks [batch, 14]."""
        with ModelTimer.time_call('single_dummy'):
            return run(self.model, x)[0]
