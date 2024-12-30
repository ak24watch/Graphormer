import torch as th
import torch.nn as nn

class EdgeEncoder(nn.Module):
    """
    Edge Encoder for encoding edge features along the shortest path.

    Args:
        max_len (int): Maximum length of the shortest path.
        feat_dim (int): Dimension of the edge features.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, max_len, feat_dim, num_heads=1):
        super().__init__()
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.embedding_table = nn.Embedding(max_len * num_heads, feat_dim)

    def forward(self, dist, path_data):
        """
        Forward pass for the edge encoder.

        Args:
            dist (Tensor): Shortest path distance tensor.
            path_data (Tensor): Edge feature tensor along the shortest path.

        Returns:
            Tensor: Path encoding tensor.
        """
        shortest_distance = th.clamp(dist, min=1, max=self.max_len)
        edge_embedding = self.embedding_table.weight.reshape(
            self.max_len, self.num_heads, -1
        )
        path_encoding = th.div(
            th.einsum("bxyld,lhd->bxyh", path_data, edge_embedding).permute(
                3, 0, 1, 2
            ),
            shortest_distance,
        ).permute(1, 2, 3, 0)
        return path_encoding