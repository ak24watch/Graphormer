import torch
import torch.nn as nn
from centralityencoding import CentralityEncoder
from spaceencoding import SpatialEncoder
from edgeencoding import EdgeEncoder
from encoder import Encoder
from signet import Signet
import einops


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

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.signet = Signet(cfg)
        if not cfg.node_classification:
            self.atom_encoder = nn.Embedding(
                cfg.num_atom_types + 1, cfg.d_model, padding_idx=0
            )
        else:
            self.atom_encoder = nn.Sequential(
                nn.Linear(cfg.num_node_features, cfg.d_model),
                nn.ReLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
        if cfg.edge_encoding:
            self.bond_encoder = nn.Embedding(
                cfg.num_bond_types + 1, cfg.d_model, padding_idx=0
            )
        if not cfg.node_classification:
            self.graph_node_enoceder = nn.Embedding(
                1, cfg.d_model if cfg.add_pos_emb else 2 * cfg.d_model
            )
            self.graph_node_virtual_distance_encoder = nn.Embedding(1, cfg.n_heads)

        self.degree_encoder = CentralityEncoder(cfg)
        if cfg.edge_encoding:
            self.path_encoder = EdgeEncoder(cfg)
        self.spatial_encoder = SpatialEncoder(cfg)

        self.emb_layer_norm = nn.LayerNorm(
            cfg.d_model if cfg.add_pos_emb else 2 * cfg.d_model
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([Encoder(cfg) for _ in range(cfg.n_layers)])
        self.lm_head_transform_weight = nn.Linear(
            cfg.d_model if cfg.add_pos_emb else 2 * cfg.d_model,
            cfg.d_model if cfg.add_pos_emb else 2 * cfg.d_model,
            bias=False,
        )
        self.layer_norm = nn.LayerNorm(
            cfg.d_model if cfg.add_pos_emb else 2 * cfg.d_model
        )

        self.embed_out = nn.Linear(
            cfg.d_model if cfg.add_pos_emb else 2 * cfg.d_model,
            cfg.num_classes if cfg.node_classification else cfg.regression_output_dim,
            bias=False,
        )
        self.lm_output_learned_bias = nn.Parameter(
            torch.zeros(
                cfg.num_classes
                if cfg.node_classification
                else cfg.regression_output_dim
            )
        )

    def reset_output_layer_parameters(self):
        """
        Reset the parameters of the output layer.
        """
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(
        self,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
        eigenvecs,
        eigen_value,
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

        if not self.cfg.node_classification:
            node_emb = self.atom_encoder(
                node_feat
            )  # [batch, max_num_nodes, node_type, d_model]
            node_emb = einops.reduce(
                node_emb,
                "batch pos node_type d_model -> batch pos d_model",
                "mean",
            )  # [batch, max_num_nodes, d_model
        else:
            node_emb = self.atom_encoder(node_feat)
        if self.cfg.edge_encoding:
            path_edata_emb = self.bond_encoder(
                path_data
            )  # [batch, max_num_nodes, max_num_nodes, max_path_length, edge_type_dim, edge_dim]
            path_edata_emb = einops.reduce(
                path_edata_emb,
                "batch pos1 pos2 path_len edge_type_dim edge_dim -> batch pos1 pos2 path_len edge_dim",
                "mean",
            )

        if self.cfg.deg_emb:
            deg_emb = self.degree_encoder(in_degree, out_degree)
            node_emb += deg_emb

        if self.cfg.pos_emb:
            pos_emb = self.signet(eigenvecs, eigen_value)
            if self.cfg.add_pos_emb:
                node_emb += pos_emb  # [batch, max_num_nodes, d_model]
            else:
                node_emb = torch.cat(
                    (pos_emb, node_emb), dim=-1
                )  # [batch, max_num_nodes, 2*d_model]
        # print("node_emb is", node_emb.shape)
        spatial_encoding = self.spatial_encoder(dist)
        if self.cfg.edge_encoding:
            path_encoding = self.path_encoder(dist, path_edata_emb)
        if not self.cfg.node_classification:
            attn_bias = torch.zeros(
                num_graphs,
                max_num_nodes + 1,
                max_num_nodes + 1,
                self.cfg.n_heads,
            )
            attn_bias[:, 1:, 1:, :] = (
                path_encoding if self.cfg.edge_encoding else 0
            ) + spatial_encoding

            graph_node_saptial_bias = (
                self.graph_node_virtual_distance_encoder.weight.reshape(
                    1, 1, self.cfg.n_heads
                )
            )
            attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + graph_node_saptial_bias
            attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + graph_node_saptial_bias
            graph_node_emb = einops.repeat(
                self.graph_node_enoceder.weight,
                "1 d_model -> num_graphs 1 d_model",
                num_graphs=num_graphs,
            )

            x = torch.cat(
                (graph_node_emb, node_emb), dim=1
            )  # [batch, max_num_nodes + 1, (d_model or 2*d_model)]

            x = self.emb_layer_norm(x)
            for layer in self.layers:
                x = layer(
                    x,
                    att_mask=attn_mask,
                    att_bias=attn_bias,
                )
            graph_rep = x[:, 0, :]
            # print("graph_rep is", graph_rep.shape)
            graph_rep = self.layer_norm(
                self.cfg.out_activation(self.lm_head_transform_weight(graph_rep))
            )
            graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias
        else:
            attn_bias = (
                path_encoding if self.cfg.edge_encoding else 0
            ) + spatial_encoding
            x = node_emb
            for layer in self.layers:
                x = layer(
                    x,
                    att_mask=attn_mask,
                    att_bias=attn_bias,
                )
            nodes_rep = self.layer_norm(
                self.cfg.out_activation(self.lm_head_transform_weight(x))
            )
            nodes_rep = self.embed_out(nodes_rep) + self.lm_output_learned_bias

        return nodes_rep if self.cfg.node_classification else graph_rep
