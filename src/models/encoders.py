import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for node/sample embeddings.

    Input:
        x: node features [num_nodes, input_dim]
        edge_index: graph edges [2, num_edges]

    Output:
        node embeddings [num_nodes, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.convs.append(SAGEConv(prev_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.output_dim = prev_dim

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        return x