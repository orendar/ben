"""BidInfo: BEN's bid info network."""

from nn.timing import ModelTimer
from nn.torch_config import create_model, run


class BidInfo:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = create_model(model_path)

    def pred_fun(self, x):
        """x: [batch, seq_len, 193] -> hcp [.., 3], shape [.., 12]."""
        with ModelTimer.time_call('bidinfo'):
            out_hcp_seq, out_shape_seq = run(self.model, x)
        return out_hcp_seq, out_shape_seq
