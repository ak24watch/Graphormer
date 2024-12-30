import torch as th
import torch.nn as nn
from centralityencoding import CentralityEncoder
from spaceencoding import SpatialEncoder
from edgeencoding import EdgeEncoder
from encoder import Encoder

class Graphormer(nn.Module):
    """
    Graphormer model for graph representation learning.

    Args:
        regrees_output_dim (int): Regression output dimension.
        edge_dim (int): Edge dimension.
        num_atoms (int): Maximum number of atoms in batch graphs.
        max_in_degree (int): Maximum in-degree in batch graphs.
        max_out_degree (int): Maximum out-degree in batch graphs.
        num_spatial (int): Maximum distance in batch graphs between two nodes.
        multi_hop_max_dist (int): Maximum multi-hop distance in batch graphs.
        num_encoder_layers (int): Number of encoder layers.
        embedding_dim (int): Embedding dimension.
        ffn_embedding_dim (int): Feed-forward network embedding dimension.
        num_attention_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        pre_layernorm (bool): Whether to use pre-layer normalization.
        activation_fn (nn.Module): Activation function.
    """
    def __init__(
        self,
        regrees_output_dim=1,
        edge_dim=1,
        num_atoms=0,
        max_in_degree=0,
        max_out_degree=0,
        num_spatial=0,
        multi_hop_max_dist=0,
        num_encoder_layers=12,
        embedding_dim=80,
        ffn_embedding_dim=80,
        num_attention_heads=8,
        dropout=0.1,
        pre_layernorm=True,
        activation_fn=nn.GELU(),
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.atom_encoder = nn.Embedding(num_atoms + 1, embedding_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, embedding_dim)
        self.degree_encoder = CentralityEncoder(
            max_in_degree=max_in_degree,
            max_out_degree=max_out_degree,
            embedding_dim=embedding_dim,
        )
        self.path_encoder = EdgeEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_attention_heads,
        )
        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial, num_heads=num_attention_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_attention_heads)
        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                Encoder(
                    hidden_size=embedding_dim,
                    ffn_out_size=ffn_embedding_dim,
                    attention_dropout=dropout,
                    num_heads=num_attention_heads,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.lm_head_transform_weight = nn.Linear(
            self.embedding_dim, self.embedding_dim
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.activation_fn = activation_fn
        self.embed_out = nn.Linear(self.embedding_dim, regrees_output_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(th.zeros(regrees_output_dim))

    def reset_output_layer_parameters(self):
        """
        Reset the parameters of the output layer.
        """
        self.lm_output_learned_bias = nn.Parameter(th.zeros(1))
        self.embed_out.reset_parameters()

    def forward(
        self,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
        attn_mask=None,
    ):
        """
        Forward pass for the Graphormer model.

        Args:
            node_feat (Tensor): Node feature tensor.
            in_degree (Tensor): In-degree tensor.
            out_degree (Tensor): Out-degree tensor.
            path_data (Tensor): Path data tensor.
            dist (Tensor): Distance tensor.
            attn_mask (Tensor, optional): Attention mask tensor.

        Returns:
            Tensor: Graph representation tensor.
        """
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(in_degree, out_degree)
        node_feat = self.atom_encoder(node_feat.int()).sum(dim=-2)
        node_feat = node_feat + deg_emb
        graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(num_graphs, 1, 1)
        x = th.cat([graph_token_feat, node_feat], dim=1)
        attn_bias = th.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding
        t = self.graph_token_virtual_distance.weight.reshape(1, 1, self.num_heads)
        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t
        x = self.emb_layer_norm(x)
        for layer in self.layers:
            x = layer(
                x,
                att_mask=attn_mask,
                att_bias=attn_bias,
            )
        graph_rep = x[:, 0, :]
        graph_rep = self.layer_norm(
            self.activation_fn(self.lm_head_transform_weight(graph_rep))
        )
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return graph_rep
