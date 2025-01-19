import torch
import torch.nn as nn
from fancy_einsum import einsum
import einops


class EdgeEncoder(nn.Module):
    """
    Edge Encoder for encoding edge features along the shortest path.

    Args:
        cfg (object): Configuration object containing model parameters.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_table = nn.Embedding(
            cfg.max_path_length * cfg.n_heads,
            cfg.d_model,
            device=cfg.device,
        )

    def forward(self, dist, path_data):
        """
        Forward pass for the edge encoder.

        Args:
            dist (Tensor): Shortest path distance tensor.
            path_data (Tensor): Edge feature tensor along the shortest path.

        Returns:
            Tensor: Path encoding tensor.
        """
        shortest_distance = torch.clamp(dist, min=1, max=self.cfg.max_path_length)
        shortest_distance = shortest_distance.unsqueeze(-1)
        path_dim = self.cfg.max_path_length
        n_heads = self.cfg.n_heads
        edge_embedding = einops.rearrange(
            self.embedding_table.weight,
            "(path_dim n_heads) edge_dim -> path_dim n_heads edge_dim",
            path_dim=path_dim,
            n_heads=n_heads,
        )

        path_encodeing = einsum(
            "batch pos1 pos2 path_dim edge_dim, path_dim n_heads edge_dim -> batch pos1 pos2 n_heads",
            path_data,
            edge_embedding,
        )
        avg_path_encoding = torch.div(path_encodeing, shortest_distance)
        return avg_path_encoding
