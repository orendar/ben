"""Torch-backed BatchPlayer - API compatible with player_tf2.BatchPlayer."""

from nn.timing import ModelTimer
from nn.torch_config import create_model, run


class BatchPlayer:

    def __init__(self, name, model_path):
        self.name = name
        self.model_path = model_path
        self.model = create_model(model_path)

    def pred_fun(self, x):
        """x: [batch, seq_len, 298] -> card logits [batch, seq_len, 32]."""
        with ModelTimer.time_call(f'player_{self.name}'):
            return run(self.model, x)[0]

    def next_cards_softmax(self, x):
        with ModelTimer.time_call(f'player_{self.name}'):
            return run(self.model, x)[0][:, -1, :]
