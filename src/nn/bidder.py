"""Bidder: BEN's bidder network."""

import numpy as np

from nn.timing import ModelTimer
from nn.torch_config import create_model, run


class Bidder:

    def __init__(self, name, model_path, alert_supported):
        self.alert_supported = alert_supported
        self.name = name
        self.model_path = model_path
        self.model = create_model(model_path)

    def pred_fun_seq(self, x):
        """x: [batch, seq_len, 193] -> bids [batch, seq_len, 40], alerts."""
        with ModelTimer.time_call('bidder'):
            out = run(self.model, x)
        bids = out[0]
        if self.alert_supported:
            alerts = out[1] if len(out) > 1 else np.zeros_like(bids[:, :, :1])
        else:
            alerts = 0
        return bids, alerts
