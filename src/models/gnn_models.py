import torch.nn as nn

from src.models.encoders import GraphSAGEEncoder
from src.models.heads import CoxHead, DeepHitHead


class GraphSAGECoxModel(nn.Module):
    """
    GraphSAGE encoder + Cox survival head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.head = CoxHead(
            input_dim=self.encoder.output_dim,
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        log_risk = self.head(h)
        return log_risk
    
class GraphSAGEDeepHitModel(nn.Module):
    """
    GraphSAGE encoder + DeepHit discrete-time survival head.

    Output:
        logits [num_nodes, n_events, n_time_bins]
    """

    def __init__(
        self,
        input_dim: int,
        n_time_bins: int,
        n_events: int = 1,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.head = DeepHitHead(
            input_dim=self.encoder.output_dim,
            n_time_bins=n_time_bins,
            n_events=n_events,
        )

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        logits = self.head(h)
        return logits