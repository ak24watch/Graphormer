import torch.nn as nn
import torch

class CentralityEncoder(nn.Module):
    """
    Centrality Encoder for encoding node centrality features.

    Args:
        cfg (object): Configuration object containing model parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.in_degree_embedding_table = nn.Embedding(
            cfg.max_degrees + 1, cfg.d_model, padding_idx=0, device=cfg.device,
        )
        self.out_degree_embedding_table = nn.Embedding(
            cfg.max_degrees + 1, cfg.d_model, padding_idx=0, device=cfg.device,
        )
        self.cfg = cfg

    def forward(self, in_degrees, out_degrees):
        """
        Forward pass for the centrality encoder.

        Args:
            in_degrees (Tensor): In-degree tensor.
            out_degrees (Tensor): Out-degree tensor.

        Returns:
            Tensor: Centrality encoding tensor.
        """
        in_degrees = torch.clamp(in_degrees, min=0, max=self.cfg.max_degrees)
        out_degrees = torch.clamp(out_degrees, min=0, max=self.cfg.max_degrees)
        z_in_degree = self.in_degree_embedding_table(in_degrees)
        z_out_degree = self.out_degree_embedding_table(out_degrees)
        z = z_in_degree + z_out_degree
        return z
