import torch.nn as nn


class CoxHead(nn.Module):
    """
    Cox survival prediction head.

    Input:
        node/sample embeddings [N, hidden_dim]

    Output:
        log_risk [N]
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, h):
        log_risk = self.linear(h)
        return log_risk.view(-1)
    
class DeepHitHead(nn.Module):
    """
    DeepHit discrete-time survival prediction head.

    Input:
        embeddings [N, hidden_dim]

    Output:
        logits [N, n_events, n_time_bins]
    """

    def __init__(self, input_dim: int, n_time_bins: int, n_events: int = 1):
        super().__init__()

        self.n_time_bins = n_time_bins
        self.n_events = n_events

        self.linear = nn.Linear(
            input_dim,
            n_events * n_time_bins,
        )

    def forward(self, h):
        logits = self.linear(h)

        logits = logits.view(
            -1,
            self.n_events,
            self.n_time_bins,
        )

        return logits