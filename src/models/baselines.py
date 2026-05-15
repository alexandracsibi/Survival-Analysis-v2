import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    """
    Simple MLP backbone used by DeepSurv and later DeepHit.

    Input:
        x: [batch_size, n_features]

    Output:
        hidden representation: [batch_size, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x):
        return self.net(x)


class DeepSurv(nn.Module):
    """
    DeepSurv-style Cox model.

    Output:
        log_risk: [batch_size]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.backbone = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.risk_head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        log_risk = self.risk_head(h)
        return log_risk.view(-1)
    
class DeepHit(nn.Module):
    """
    DeepHit-style discrete-time survival model.

    Output:
        logits: [batch_size, n_events, n_time_bins]

    For single-event survival:
        n_events = 1

    For competing risks:
        n_events > 1
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

        self.n_time_bins = n_time_bins
        self.n_events = n_events

        self.backbone = MLPBackbone(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        self.output_layer = nn.Linear(
            self.backbone.output_dim,
            n_events * n_time_bins,
        )

    def forward(self, x):
        h = self.backbone(x)
        logits = self.output_layer(h)

        logits = logits.view(
            -1,
            self.n_events,
            self.n_time_bins,
        )

        return logits