import torch
import torch.nn as nn
from fancy_einsum import einsum
import einops


class EdgeEncoder(nn.Module):
    """
    Edge Encoder for encoding edge features along the shortest path.

    Args:
        max_len (int): Maximum length of the shortest path.
        feat_dim (int): Dimension of the edge features.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.embedding_table = nn.Embedding(
            cfg.max_path_length * cfg.n_heads, cfg.d_model
        )

    def forward(self, dist, path_data):
        """
        Forward pass for the edge encoder.

        Args:
            dist (Tensor): [batch_size, num_nodes, num_nodes]=> Shortest path distance tensor.
            path_data (Tensor): [batch_size, num_nodes, num_nodes, max_path_length, edge_dim]=> Edge feature tensor along the shortest path.


        Returns:
            Tensor: Path encoding tensor.
        """
        shortest_distance = torch.clamp(dist, min=1, max=self.cfg.max_path_length)
        shortest_distance = shortest_distance.unsqueeze(-1) # [batch_size, num_nodes, num_nodes, 1]
        path_dim = self.cfg.max_path_length
        n_heads = self.cfg.n_heads
        edge_embedding = einops.rearrange(
            self.embedding_table.weight,
            "(path_dim n_heads) edge_dim -> path_dim n_heads edge_dim",
            path_dim=path_dim, n_heads=n_heads
        )

        path_encodeing = einsum(
            "batch pos1 pos2 path_dim edge_dim, path_dim n_heads edge_dim -> batch pos1 pos2 n_heads",
            path_data,
            edge_embedding,
        )
        avg_path_encoding = torch.div(path_encodeing, shortest_distance)
        return avg_path_encoding
