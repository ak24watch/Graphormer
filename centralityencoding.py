import torch.nn as nn

class CentralityEncoder(nn.Module):
    """
    Centrality Encoder for encoding node centrality features.

    Args:
        max_in_degree (int): Maximum in-degree of nodes.
        max_out_degree (int): Maximum out-degree of nodes.
        embedding_dim (int): Dimension of the embedding.
    """
    def __init__(self, max_in_degree, max_out_degree, embedding_dim):
        super().__init__()
        self.in_degree_embedding_table = nn.Embedding(max_in_degree+1, embedding_dim, padding_idx=0)
        self.out_degree_embedding_table = nn.Embedding(max_out_degree+1, embedding_dim, padding_idx=0)

    def forward(self, in_degrees, out_degrees):
        """
        Forward pass for the centrality encoder.

        Args:
            in_degrees (Tensor): In-degree tensor.
            out_degrees (Tensor): Out-degree tensor.

        Returns:
            Tensor: Centrality encoding tensor.
        """
        z_in_degree = self.in_degree_embedding_table(in_degrees)
        z_out_degree = self.out_degree_embedding_table(out_degrees)
        z = z_in_degree + z_out_degree
        return z
