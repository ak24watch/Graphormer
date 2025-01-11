import torch.nn as nn
import torch as th

class SpatialEncoder(nn.Module):
    """
    Spatial Encoder for encoding shortest path distances.

    Args:
        max_dist (int): Maximum distance for the shortest path.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_table = nn.Embedding(cfg.max_num_nodes + 2, cfg.n_heads, padding_idx=0)

    def forward(self, dist):
        """
        Forward pass for the spatial encoder.

        Args:
            dist (Tensor): Shortest path distance tensor.

        Returns:
            Tensor: Spatial encoding tensor.
        """
        spatial_encoding = self.embedding_table(
            th.clamp(
                dist,
                min=-1,
                max=self.cfg.max_dist,
            )
            + 1
        )
        return spatial_encoding
